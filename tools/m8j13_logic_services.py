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
from tools.m8j13_logic_shutdown import first_fault,last_snapshot
from simulator.m8j13_parked_handoff import enabled as parked_handoff_enabled,verify_io_proof,uncertainty
from simulator.m8j13_logic_shadow import require_parked_execution,PROTOCOL
from simulator.m8j13_logic_faults import CameraSuppression,receive_camera,collect_input,synthetic_reply,scenario,validate_manual_request
from simulator.m8j13_logic_runtime import STOP,TargetCapture,supervision_reason,frozen_equal,OperationDeadline


def load(path):return json.loads(path.read_text()) if path.exists() else None


class ServerAwareDeadline(OperationDeadline):
    def __init__(self,operation,clock,root):
        super().__init__(operation,clock);self.root=root
    def remaining(self):
        if (self.root/'server_exited.json').exists():raise RuntimeError('ServerExited_NoFurtherRPC')
        return super().remaining()


def retain_handoff_uncertainty(root,request,error):
    """Single I/O writer publishes the first failure atomically for watchdog."""
    path=root/'io_handoff_uncertain.json'
    if not path.exists():
        atomic(path,dict(operation=request['operation'],command=request['args']['command'],error=error,wall_s=time.monotonic(),remote_order_confirmed=False))


def batch_control(client, ego_id, control, budget=None):
    import carla
    args=([carla.command.ApplyVehicleControl(ego_id,carla.VehicleControl(**control))],False)
    responses=client.apply_batch_sync(*args) if budget is None else budget.call(client.apply_batch_sync,*args,client=client)
    if len(responses)!=1 or responses[0].has_error() or responses[0].actor_id!=ego_id:
        raise RuntimeError('BatchProcessingError:'+str([r.error for r in responses]))
    return dict(server_processed=True,physics_confirmed=False,due_tick_cue=False,actor_id=ego_id)


def io_service(root,cfg):
    import carla
    import numpy as np
    from types import SimpleNamespace
    from tools.m8j13_compatibility import Inputs
    from tools.m8j13_logic_io import DeadlineOwner,measured_tick,cleanup_owner
    from simulator.m8j13_carla import measurement
    from simulator.m8j13_execution import control_record,feedback_matches
    from simulator.m8j13_navigation import load_planner,set_navigation_route
    from datasets.m8j12_contract import assert_observation
    owner=None;ego=None;inputs=None;target=None;last=None;seen=set();stats={};ticks=0;active=None;confirmed=None;hook=None
    def drain():
        if inputs is None:return
        # Reuse raw callback copies and identity check; only retain selected payloads.
        for payload in inputs.inbox.drain():
            receive_camera(payload,target,inputs,hook,lambda row:append(root/'camera_metadata.jsonl',row))
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
        seen.add(request.name);r=load(request);start=time.monotonic();reply=dict(sequence=r['sequence'],kind=r['kind'],call_start_wall_s=start,io_received_wall_s=start,operation=r['operation'],owner_submitted_wall_s=r['owner_submitted_wall_s'])
        budget=ServerAwareDeadline(r['operation'],time.monotonic,root)
        try:
            if (root/'server_exited.json').exists():raise RuntimeError('ServerExited_NoFurtherRPC')
            budget.remaining();kind=r['kind'];args=r.get('args',{})
            if (root/'shutdown_request.json').exists() and kind!='cleanup' and args.get('command',{}).get('source')!='FaultBrake':
                raise RuntimeError('TerminalNoOrdinaryOperation')
            if owner is not None:owner.budget=budget
            if kind=='setup':
                owner=DeadlineOwner(root,cfg,budget);owner.setup()
                ego=owner.ego();inputs=Inputs(owner,ego,root);budget.remaining()
                target=TargetCapture(inputs.identity,{n:a.id for n,a in inputs.sensor_map.items()})
                hook=CameraSuppression(cfg,inputs.identity,target.actors)
                map_=owner.rpc(owner.world.get_map);planner=load_planner(Path(owner.c['runtime_root'])/'official-source/leaderboard/team_code/planner.py',map_)
                gps=map_.transform_to_geolocation(owner.target)
                set_navigation_route(planner,[({'lat':gps.latitude,'lon':gps.longitude,'alt':gps.altitude},SimpleNamespace(value=4))],coordinate_type='gps',client_version='0.9.16',carla_map=map_)
                inputs.navigation=planner
                reply.update(ego_id=ego.id,identity=list(inputs.identity),frame=owner.last,clients=owner.clients,geometry=load(root/'geometry.json'),sensor_actors=target.actors)
            elif kind=='register':
                drain();target.register(args['frame']);inputs.groups.clear();inputs.states.clear();inputs.imu.clear();inputs.payloads.clear()
                append(root/'events.jsonl',dict(event='target_registered',frame=target.target,wall_s=time.monotonic()))
                if args.get('injection'):
                    hook.arm(args['injection']);write(root/'camera_injection_armed.json',dict(key=hook.key,wall_s=time.monotonic()))
                reply['registered']=target.target
            elif kind=='control':
                cmd=args['command'];require_parked_execution(cfg,cmd,tick=kind=='tick');fault=load(root/'watchdog_fault.json')
                if fault and cmd['source']!='FaultBrake':raise RuntimeError('FaultLockedBeforeRPC')
                before_snapshot=owner.rpc(owner.world.get_snapshot);before=before_snapshot.frame
                reply['pre_control_snapshot_wall_s']=time.monotonic()
                if args.get('parked_handoff'):
                    current=measurement(before_snapshot,ego)
                    current.update(full_control=control_record(owner.rpc(ego.get_control)),identity=list(inputs.identity),ego_id=ego.id,epoch=cmd['epoch'])
                    state=load(root/'owner_state.json')
                    verify_io_proof(cfg,args['parked_handoff'],cmd,current,state,r['operation'],inputs.identity,ego.id,load(root/'io_handoff_uncertain.json') is not None,last=last,active=active)
                    reply['parked_proof_checked_wall_s']=time.monotonic()
                elif r['operation']['operation_limit_s']>.2:raise RuntimeError('UncertifiedControlDeadline')
                reply['rpc_start_wall_s']=time.monotonic()
                try:reply.update(batch_control(owner.client,ego.id,cmd['control'],budget))
                finally:reply['rpc_return_wall_s']=time.monotonic()
                after=owner.rpc(owner.world.get_snapshot).frame;reply['post_control_snapshot_wall_s']=time.monotonic()
                if before!=after:raise RuntimeError('BatchUnexpectedWorldAdvance')
                active=cmd;reply.update(command=cmd,frame_before=before,frame_after=after)
            elif kind=='tick':
                cmd=args['command'];require_parked_execution(cfg,cmd,tick=kind=='tick');fault=load(root/'watchdog_fault.json')
                if fault and cmd['source']!='FaultBrake':raise RuntimeError('FaultLockedBeforeTick')
                if cmd['source']=='PlannedHoldBrake':raise RuntimeError('HoldCannotTick')
                if cmd['source']=='FaultBrake':
                    try:drain()
                    except Exception as exc:append(root/'safety_input_faults.jsonl',dict(error=repr(exc),wall_s=time.monotonic()))
                else:drain()
                before=owner.rpc(owner.world.get_snapshot).frame
                if before!=owner.last:raise RuntimeError('UnexpectedWorldAdvance')
                budget.remaining();tick_start=time.monotonic();target.start_tick(before+1,tick_start)
                # Count attempted ticks BEFORE the blocking call; never silently retry.
                ticks+=1;owner.tick_count=ticks
                append(root/'tick_requests.jsonl',dict(index=ticks,phase=args['phase'],expected_frame=before+1,wall_s=tick_start,command=cmd))
                snap,last=measured_tick(owner,ego,budget,reply,before)
                frame=snap.frame;owner.last=frame
                if parked_handoff_enabled(cfg):last.update(identity=list(inputs.identity),ego_id=ego.id,epoch=cmd['epoch'])
                last.update(input_wall_s=tick_start,tick_end_wall_s=time.monotonic(),delta_s=snap.timestamp.delta_seconds,phase=args['phase'],command=cmd)
                if abs(last['delta_s']-.05)>1e-6:raise RuntimeError('FixedDeltaChanged')
                inputs.snapshot(snap,last);target.confirm(last)
                if cmd['source']=='FaultBrake':
                    try:drain()
                    except Exception as exc:append(root/'safety_input_faults.jsonl',dict(error=repr(exc),wall_s=time.monotonic()))
                else:drain()
                matched=feedback_matches(cmd['control'],last['full_control']);last['feedback_matches']=matched
                append(root/'snapshots.jsonl',last)
                if not matched:raise RuntimeError('ControlFeedbackMismatch')
                budget.remaining();reply.update(snapshot=last,explicit_ticks=ticks)
            elif kind=='collect':
                until=min(target.start+.5,budget.operation['deadline'])
                def confirmed_current():
                    current=measurement(owner.rpc(owner.world.get_snapshot),ego)
                    if last is None or not frozen_equal(last,current):raise RuntimeError('UnexpectedWorldBeforeCollect')
                    return current
                try:
                    collect_input(target,inputs,confirmed_current,drain,budget,time.monotonic,time.sleep,cfg.get('collect_return_reserve_s',0.))
                except ValueError as exc:
                    write(root/'capture_rejection.json',dict(reason=str(exc),wall_s=time.monotonic(),target=target.target,
                        groups=list(target.groups.values()),snapshot=target.snapshot,current=last,imu=inputs.imu.get(target.target),
                        suppression=hook.hit,original_start_wall_s=target.start,operation=r['operation']))
                    if str(exc)=='MissingCamera':reply['rejection_reason']='MissingCamera'
                    raise
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
                if cfg.get('shadow_protocol')==PROTOCOL:
                    reply['shadow_capture']=dict(row=row,camera_metadata=list(target.groups.values()),archive_done_wall_s=time.monotonic())
                reply.update(observation=inputs.states[target.target],frozen=last,capture_complete_wall_s=completed,capture_wall_s=completed-target.start,stats=dict(target.stats))
            elif kind=='frozen':
                drain();current=measurement(owner.rpc(owner.world.get_snapshot),ego)
                if not frozen_equal(args['expected'],current):raise RuntimeError('FrozenWorldChanged')
                if parked_handoff_enabled(cfg):current.update(full_control=control_record(owner.rpc(ego.get_control)),identity=list(inputs.identity),ego_id=ego.id,epoch=args['expected']['epoch'])
                reply['snapshot']=current;reply['frozen_checked_wall_s']=time.monotonic()
            elif kind=='cleanup':
                if inputs is not None:inputs.close();target.close()
                reply['cleanup']=cleanup_owner(owner,budget,root,args.get('advance_unknown',False))
                reply['explicit_ticks']=ticks
            else:raise ValueError('UnknownIOOperation')
            budget.remaining();reply['success']=True
        except BaseException as exc:
            reply.update(success=False,error=repr(exc),traceback=traceback.format_exc(),explicit_ticks=ticks)
        reply['processing_complete_wall_s']=time.monotonic()
        reply['call_end_wall_s']=time.monotonic()
        if reply['call_end_wall_s']>r['operation']['deadline']:
            reply['success']=False;reply.setdefault('error','OperationDeadlineAtPublication')
        if r['kind']=='control' and not reply['success']:
            retain_handoff_uncertainty(root,r,reply.get('error'))
        reply['publication_started_wall_s']=time.monotonic()
        atomic(root/'io_responses'/request.name,reply)
        append(root/'io_publications.jsonl',dict(sequence=r['sequence'],publication_done_wall_s=time.monotonic()))
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
                if pending.get('injection'):
                    state=load(root/'owner_state.json')
                    if not state or pending['identity']!=state.get('identity') or pending['sequence']!=state.get('task',{}).get('sequence'):raise ValueError('WorkerInjectionIdentityMismatch')
                    if (root/'worker_injection.json').exists():raise ValueError('DuplicateWorkerInjection')
                    write(root/'worker_injection.json',dict(sequence=pending['sequence'],identity=pending['identity'],frame=pending['frame'],wall_s=now,scenario=scenario(cfg)))
        if pending is not None:
            due=now-pending['worker_received_wall_s']>=cfg['normal_task_delay_s']
            late=pending['mode']=='blocked' and (root/'release_late_result').exists()
            if (pending['mode']=='normal' and due) or late:
                reply=synthetic_reply(pending,now,cfg,late)
                atomic(root/'task_results'/('%06d.json'%pending['sequence']),reply);pending=None
        time.sleep(.02)


def watchdog(root,cfg):
    if (root/'server_exited.json').exists():
        state=load(root/'owner_state.json') or {}
        fault=first_fault(root,'ServerExited','watchdog',state,last_snapshot(root))
        atomic(root/'watchdog_fault.json',fault)
        atomic(root/'fault_brake.json',dict(reason=fault['reason'],status='CommandNotDelivered_ServerExited',server_processed=False,physics_confirmed=False,handoff_uncertain=bool(state.get('operation'))))
        return
    import carla
    c=config();client=carla.Client('127.0.0.1',c['port']);client.set_timeout(.2)
    versions=[client.get_client_version(),client.get_server_version()]
    if versions!=['0.9.16']*2:raise RuntimeError('WatchdogVersionMismatch')
    atomic(root/'logic_watchdog_ready.json',dict(pid=os.getpid(),versions=versions,carla_path=carla.libcarla.__file__,tick_calls=0))
    latched=None
    while not (root/'logic_watchdog_stop').exists():
        state=load(root/'owner_state.json');now=time.monotonic()
        if not state:time.sleep(.02);continue
        dead=(root/'server_exited.json').exists()
        reason='ServerExited' if dead else supervision_reason(state,now,(root/'EMERGENCY_STOP').exists())
        if reason and latched is None:
            latched=first_fault(root,reason,'watchdog',state,last_snapshot(root),now)
            reason=latched['reason']
            atomic(root/'watchdog_fault.json',latched)
            if dead:
                atomic(root/'fault_brake.json',dict(reason=reason,status='CommandNotDelivered_ServerExited',server_processed=False,physics_confirmed=False,submission_wall_s=None,return_wall_s=now,handoff_uncertain=bool(state.get('handoff_uncertain') or state.get('operation'))))
                continue
            # No cross-RPC lock. Wait only for an already submitted normal batch's
            # bounded processing response; never assume a timeout cancels it.
            operation=state.get('operation');uncertain,failed_operation=uncertainty(state,load(root/'io_handoff_uncertain.json'))
            if operation and operation['kind']=='control':
                until=min(now+.2,operation['deadline'])
                response=None
                while time.monotonic()<=until:
                    response=load(root/'io_responses'/('%06d.json'%operation['sequence']))
                    if response:break
                    time.sleep(.002)
                uncertain=uncertain or not response or not response.get('success') or not response.get('server_processed') or response.get('call_end_wall_s',operation['deadline']+1)>operation['deadline']
            result=dict(reason=reason,source='FaultBrake',control=STOP,submission_wall_s=time.monotonic(),handoff_uncertain=uncertain,ordinary_handoff_confirmed=not uncertain,failed_operation=failed_operation)
            result['operation']=dict(sequence='fault:'+str(latched['epoch']),kind='fault_control',started=result['submission_wall_s'],deadline=result['submission_wall_s']+.2)
            safety_budget=OperationDeadline(result['operation'],time.monotonic)
            try:
                if (root/'server_exited.json').exists():raise RuntimeError('ServerExited_NoFaultRPC')
                # One bounded safety attempt even if ordinary ordering is unknown;
                # no physical continuation is admitted in that case.
                result.update(batch_control(client,state['ego_id'],STOP,safety_budget))
                result['status']='HandoffUncertain' if uncertain else 'ServerProcessed_PhysicsUnverified'
            except Exception as exc:result.update(status='ResponseUnverified',error=repr(exc),server_processed=False)
            # Owner may discover late consumption while this independent RPC runs.
            # Refresh persistent ordering evidence; brake processing never clears it.
            latest_uncertain,latest_failure=uncertainty(load(root/'owner_state.json') or {},load(root/'io_handoff_uncertain.json'))
            uncertain=uncertain or latest_uncertain
            result.update(handoff_uncertain=uncertain,ordinary_handoff_confirmed=not uncertain,failed_operation=failed_operation or latest_failure)
            if uncertain and result.get('server_processed'):result['status']='HandoffUncertain'
            result['return_wall_s']=time.monotonic()
            if result['return_wall_s']>result['operation']['deadline']:
                result.update(status='ResponseUnverified',error='FaultBrakeDeadlineAtReturn',server_processed=False)
            atomic(root/'fault_brake.json',result)
        time.sleep(.02)


def manual_driver(root,cfg):
    if scenario(cfg)!='manual_emergency_stop':raise ValueError('ManualDriverDisabled')
    atomic(root/'manual_ready.json',dict(pid=os.getpid(),tick_calls=0))
    while not (root/'manual_stop').exists():
        request=load(root/'manual_request.json')
        if request:
            now=time.monotonic()
            if now>=request['task']['requested_wall_s']+cfg['fault_injection']['manual_delay_s']:
                state=load(root/'owner_state.json');hb=load(root/'test_worker_heartbeat.json')
                validate_manual_request(request,state,hb,now)
                if (root/'EMERGENCY_STOP').exists():raise ValueError('DuplicateManualInjection')
                record=dict(request_wall_s=now,identity=request['identity'],task=request['task'],owner_heartbeat=state['owner_wall_s'],worker_heartbeat=hb,source='IndependentTestDriverViaManualEntry')
                atomic(root/'manual_injection.json',record)
                atomic(root/'EMERGENCY_STOP',record)
                while not (root/'manual_stop').exists():time.sleep(.01)
                return
        time.sleep(.005)


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['io','test','watchdog','manual'],required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    cfg=load(a.root/'frozen_config.json');os.umask(0o077)
    {'io':io_service,'test':test_worker,'watchdog':watchdog,'manual':manual_driver}[a.role](a.root,cfg)

if __name__=='__main__':main()
