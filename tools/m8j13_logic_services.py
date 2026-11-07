"""Separate IO, phase supervisor and synthetic worker for one no-model test.

IO ticks ONLY in response to an explicit owner request. Watchdog never ticks.
The synthetic worker runs in its own network namespace and imports no CARLA.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from tools.m8j13_common import atomic,append,write,config
from simulator.m8j13_logic_runtime import STOP,TargetCapture,supervision_reason,frozen_equal


def load(path):return json.loads(path.read_text()) if path.exists() else None


def batch_control(client, ego_id, control):
    import carla
    responses=client.apply_batch_sync([carla.command.ApplyVehicleControl(ego_id,carla.VehicleControl(**control))],False)
    if len(responses)!=1 or responses[0].has_error() or responses[0].actor_id!=ego_id:
        raise RuntimeError('BatchProcessingError:'+str([r.error for r in responses]))
    return dict(server_processed=True,physics_confirmed=False,due_tick_cue=False,actor_id=ego_id)


def io_service(root,cfg):
    import carla
    import numpy as np
    from types import SimpleNamespace
    from tools.m8j13_compatibility import Owner,Inputs
    from simulator.m8j13_carla import measurement
    from simulator.m8j13_execution import control_record,feedback_matches
    from simulator.m8j13_navigation import load_planner,set_navigation_route
    from datasets.m8j12_contract import assert_observation
    owner=None;ego=None;inputs=None;target=None;last=None;seen=set();stats={};ticks=0;active=None;confirmed=None
    def drain():
        if inputs is None:return
        # Reuse raw callback copies and identity check; only retain selected payloads.
        for payload in inputs.inbox.drain():
            row=payload.stamp.record();row['copied_monotonic_s']=payload.copied_monotonic_s
            if (payload.width,payload.height,len(payload.bgra))!=(704,396,704*396*4):raise RuntimeError('CameraPayloadShapeMismatch')
            outcome=target.camera(row);row['disposition']=outcome
            append(root/'camera_metadata.jsonl',row)
            if outcome in ('PendingSnapshot','Selected'):
                inputs.groups[row['frame']].append(row);inputs.payloads[(row['camera'],row['frame'])]=payload.bgra
        while not inputs.imu_queue.empty():
            x=inputs.imu_queue.get_nowait()
            if x.frame==target.target:
                if x.frame in inputs.imu:raise RuntimeError('DuplicateIMU')
                inputs.imu[x.frame]=dict(frame=x.frame,timestamp=x.timestamp,values=[x.accelerometer.x,x.accelerometer.y,x.accelerometer.z,x.gyroscope.x,x.gyroscope.y,x.gyroscope.z,x.compass])
        while not inputs.collision.empty():
            x=inputs.collision.get_nowait();r=dict(frame=x.frame,other=x.other_actor.type_id);inputs.collisions.append(r);append(root/'collisions.jsonl',r)
        if inputs.inbox.overflow:raise RuntimeError('InputQueueOverflow')
        if inputs.collisions:raise RuntimeError('CollisionObserved')
    atomic(root/'io_ready.json',dict(pid=os.getpid(),python=__import__('sys').executable,carla_path=carla.libcarla.__file__))
    while not (root/'io_stop').exists():
        requests=sorted((root/'io_requests').glob('*.json'))
        request=next((p for p in requests if p.name not in seen),None)
        if request is None:time.sleep(.005);continue
        seen.add(request.name);r=load(request);start=time.monotonic();reply=dict(sequence=r['sequence'],kind=r['kind'],call_start_wall_s=start)
        try:
            kind=r['kind'];args=r.get('args',{})
            if kind=='setup':
                owner=Owner.__new__(Owner);owner.__init__(root,cfg)
                owner.client.set_timeout(.5);ego=owner.ego();inputs=Inputs(owner,ego,root)
                target=TargetCapture(inputs.identity,{n:a.id for n,a in inputs.sensor_map.items()})
                map_=owner.world.get_map();planner=load_planner(Path(owner.c['runtime_root'])/'official-source/leaderboard/team_code/planner.py',map_)
                gps=map_.transform_to_geolocation(owner.target)
                set_navigation_route(planner,[({'lat':gps.latitude,'lon':gps.longitude,'alt':gps.altitude},SimpleNamespace(value=4))],coordinate_type='gps',client_version='0.9.16',carla_map=map_)
                inputs.navigation=planner
                reply.update(ego_id=ego.id,identity=list(inputs.identity),frame=owner.last,clients=owner.clients,geometry=load(root/'geometry.json'))
            elif kind=='register':
                drain();target.register(args['frame']);inputs.groups.clear();inputs.states.clear();inputs.imu.clear();inputs.payloads.clear()
                append(root/'events.jsonl',dict(event='target_registered',frame=target.target,wall_s=time.monotonic()))
                reply['registered']=target.target
            elif kind=='control':
                cmd=args['command'];fault=load(root/'watchdog_fault.json')
                if fault and cmd['source']!='FaultBrake':raise RuntimeError('FaultLockedBeforeRPC')
                before=owner.world.get_snapshot().frame;owner.client.set_timeout(.2)
                reply.update(batch_control(owner.client,ego.id,cmd['control']));owner.client.set_timeout(.5)
                after=owner.world.get_snapshot().frame
                if before!=after:raise RuntimeError('BatchUnexpectedWorldAdvance')
                active=cmd;reply.update(command=cmd,frame_before=before,frame_after=after)
            elif kind=='tick':
                cmd=args['command'];fault=load(root/'watchdog_fault.json')
                if fault and cmd['source']!='FaultBrake':raise RuntimeError('FaultLockedBeforeTick')
                if cmd['source']=='PlannedHoldBrake':raise RuntimeError('HoldCannotTick')
                if cmd['source']=='FaultBrake':
                    try:drain()
                    except Exception as exc:append(root/'safety_input_faults.jsonl',dict(error=repr(exc),wall_s=time.monotonic()))
                else:drain()
                before=owner.world.get_snapshot().frame
                if before!=owner.last:raise RuntimeError('UnexpectedWorldAdvance')
                tick_start=time.monotonic();target.start_tick(before+1,tick_start)
                # Count attempted ticks BEFORE the blocking call; never silently retry.
                ticks+=1;owner.tick_count=ticks
                append(root/'tick_requests.jsonl',dict(index=ticks,phase=args['phase'],expected_frame=before+1,wall_s=tick_start,command=cmd))
                frame=owner.world.tick(.5);rpc_return=time.monotonic();snap=owner.world.get_snapshot()
                if frame!=snap.frame or frame!=before+1:raise RuntimeError('CompetingTickOwner')
                owner.last=frame;last=measurement(snap,ego);last.update(input_wall_s=tick_start,tick_end_wall_s=time.monotonic(),delta_s=snap.timestamp.delta_seconds,full_control=control_record(ego.get_control()),phase=args['phase'],command=cmd)
                if abs(last['delta_s']-.05)>1e-6:raise RuntimeError('FixedDeltaChanged')
                inputs.snapshot(snap,last);target.confirm(last)
                if cmd['source']=='FaultBrake':
                    try:drain()
                    except Exception as exc:append(root/'safety_input_faults.jsonl',dict(error=repr(exc),wall_s=time.monotonic()))
                else:drain()
                matched=feedback_matches(cmd['control'],last['full_control']);last['feedback_matches']=matched
                append(root/'snapshots.jsonl',last)
                if not matched:raise RuntimeError('ControlFeedbackMismatch')
                reply.update(snapshot=last,rpc_return_wall_s=rpc_return,explicit_ticks=ticks)
            elif kind=='collect':
                until=target.start+.5
                while time.monotonic()<until:
                    drain()
                    if len(target.groups)==6 and target.target in inputs.imu:break
                    time.sleep(.002)
                current=measurement(owner.world.get_snapshot(),ego)
                if last is None or not frozen_equal(last,current):raise RuntimeError('UnexpectedWorldBeforeCollect')
                target.validate(current,time.monotonic())
                row,error=inputs.row(target.target,time.monotonic())
                if error:raise RuntimeError(error)
                images={name:np.frombuffer(inputs.payloads[(name,target.target)],np.uint8).reshape(396,704,4)[:,:,:3] for name in target.actors}
                assert_observation(row,images)
                completed=time.monotonic()
                if completed>until:raise RuntimeError('CaptureDeadlineAfterChecks')
                folder=root/'inputs'/str(target.target);folder.mkdir(parents=True)
                # Off critical contract check, private-only archival; charged in wall flow.
                for name in target.actors:(folder/(name+'.bgra')).write_bytes(inputs.payloads[(name,target.target)])
                write(folder/'observation.json',row)
                reply.update(observation=inputs.states[target.target],frozen=last,capture_complete_wall_s=completed,capture_wall_s=completed-target.start,stats=dict(target.stats))
            elif kind=='frozen':
                drain();current=measurement(owner.world.get_snapshot(),ego)
                if not frozen_equal(args['expected'],current):raise RuntimeError('FrozenWorldChanged')
                reply['snapshot']=current
            elif kind=='cleanup':
                if inputs is not None:inputs.close();target.close()
                # Cleanup allowed only after all IO requests resolved; never a stop proof.
                reply['cleanup']=owner.close() if owner is not None else {'status':'NoWorldOwned'}
                reply['explicit_ticks']=ticks
            else:raise ValueError('UnknownIOOperation')
            reply['success']=True
        except BaseException as exc:
            reply.update(success=False,error=repr(exc),traceback=traceback.format_exc(),explicit_ticks=ticks)
        reply['call_end_wall_s']=time.monotonic();atomic(root/'io_responses'/request.name,reply)
        if r['kind']=='cleanup':break


def test_worker(root,cfg):
    # The same one-task structured request/response convention as future workers.
    seen=set();pending=None
    atomic(root/'test_worker_ready.json',dict(pid=os.getpid(),network_namespace=os.readlink('/proc/self/ns/net'),source='IndependentTestControl',model_forwards=0))
    while not (root/'test_worker_stop').exists():
        now=time.monotonic();atomic(root/'test_worker_heartbeat.json',dict(wall_s=now,task=None if pending is None else pending['sequence']))
        if pending is None:
            p=next((p for p in sorted((root/'tasks').glob('*.json')) if p.name not in seen),None)
            if p:
                pending=load(p);seen.add(p.name);pending['worker_received_wall_s']=now
        if pending is not None:
            due=now-pending['worker_received_wall_s']>=cfg['normal_task_delay_s']
            late=pending['mode']=='blocked' and (root/'release_late_result').exists()
            if (pending['mode']=='normal' and due) or late:
                reply={k:pending[k] for k in ('sequence','identity','frame','source','control')}
                reply.update(worker_received_wall_s=pending['worker_received_wall_s'],worker_completed_wall_s=now,late=late,model_forward=False)
                atomic(root/'task_results'/('%06d.json'%pending['sequence']),reply);pending=None
        time.sleep(.02)


def watchdog(root,cfg):
    import carla
    c=config();client=carla.Client('127.0.0.1',c['port']);client.set_timeout(.2)
    versions=[client.get_client_version(),client.get_server_version()]
    if versions!=['0.9.16']*2:raise RuntimeError('WatchdogVersionMismatch')
    atomic(root/'logic_watchdog_ready.json',dict(pid=os.getpid(),versions=versions,carla_path=carla.libcarla.__file__,tick_calls=0))
    latched=None
    while not (root/'logic_watchdog_stop').exists():
        state=load(root/'owner_state.json');now=time.monotonic()
        if not state:time.sleep(.02);continue
        reason=supervision_reason(state,now,(root/'EMERGENCY_STOP').exists())
        if reason and latched is None:
            latched=dict(reason=reason,wall_s=now,phase=state['phase'],task=state.get('task'),owner_wall_s=state['owner_wall_s'],epoch=state['epoch']+1)
            atomic(root/'watchdog_fault.json',latched)
            # No cross-RPC lock. Wait only for an already submitted normal batch's
            # bounded processing response; never assume a timeout cancels it.
            operation=state.get('operation');uncertain=False
            if operation and operation['kind']=='control':
                until=min(now+.2,operation['deadline'])
                response=None
                while time.monotonic()<=until:
                    response=load(root/'io_responses'/('%06d.json'%operation['sequence']))
                    if response:break
                    time.sleep(.002)
                uncertain=not response or not response.get('success') or not response.get('server_processed')
            result=dict(reason=reason,source='FaultBrake',control=STOP,submission_wall_s=time.monotonic(),handoff_uncertain=uncertain)
            try:
                # One bounded safety attempt even if ordinary ordering is unknown;
                # no physical continuation is admitted in that case.
                result.update(batch_control(client,state['ego_id'],STOP))
                result['status']='HandoffUncertain' if uncertain else 'ServerProcessed_PhysicsUnverified'
            except Exception as exc:result.update(status='ResponseUnverified',error=repr(exc),server_processed=False)
            result['return_wall_s']=time.monotonic();atomic(root/'fault_brake.json',result)
        time.sleep(.02)


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['io','test','watchdog'],required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    cfg=load(a.root/'frozen_config.json');os.umask(0o077)
    {'io':io_service,'test':test_worker,'watchdog':watchdog}[a.role](a.root,cfg)

if __name__=='__main__':main()
