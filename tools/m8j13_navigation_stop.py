"""Two-start bounded navigation and independent physical braking validation.

No model import or output. Owner alone ticks. Every round reserves a braking
window; a missing prerequisite is a failure even when its fallback stops.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from tools.m8j13_common import config,atomic,write,append,digest
from tools.m8j13_compatibility import Owner,Inputs
from simulator.m8j13_execution import STOP,control_record,feedback_matches,execution_reason,issue_command,round_exit_reason
from simulator.m8j13_safety import SafetyGate
from simulator.m8j13_navigation import load_planner,set_navigation_route
from datasets.m8j12_contract import measured_navigation_target,native_target

CFG=Path('configs/m8j13_navigation_stop.json')


def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def physics_record(ego,world):
    p=ego.get_physics_control();tf=ego.get_transform();wp=world.get_map().get_waypoint(tf.location)
    wheels=[{k:getattr(w,k) for k in ('tire_friction','damping_rate','radius','max_brake_torque','max_handbrake_torque')} for w in p.wheels]
    return dict(mass=p.mass,drag_coefficient=p.drag_coefficient,use_gear_autobox=p.use_gear_autobox,
        gear_switch_time=p.gear_switch_time,clutch_strength=p.clutch_strength,
        torque_curve=[[x.x,x.y] for x in p.torque_curve],forward_gears=[dict(ratio=g.ratio,down_ratio=g.down_ratio,up_ratio=g.up_ratio) for g in p.forward_gears],
        wheels=wheels,ego_pitch_deg=tf.rotation.pitch,ego_roll_deg=tf.rotation.roll,
        road_pitch_deg=None if wp is None else wp.transform.rotation.pitch,physics_modified=False)


class TestOwner(Owner):
    def tick(self,ego,phase,folder):
        snap,m=super().tick(ego,phase,folder)
        m['full_control']=control_record(ego.get_control());tf=snap.find(ego.id).get_transform()
        m['pitch_deg']=tf.rotation.pitch;m['roll_deg']=tf.rotation.roll
        a=snap.find(ego.id).get_acceleration();m['acceleration']=[a.x,a.y,a.z]
        wp=self.world.get_map().get_waypoint(tf.location)
        m['road_pitch_deg']=None if wp is None else wp.transform.rotation.pitch
        append(folder/'feedback.jsonl',m)
        return snap,m


def issue(owner,ego,folder,raw,source,state,gate,m,plan):
    import carla
    with (folder/'actuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        fault=load(folder/'watchdog_fault.json')
        if fault:gate.stop(fault['reason'],0 if m is None else m['sim_s'],time.monotonic())
        if gate.reason:raw=STOP;source='SafetyStop'
        now=time.monotonic()
        try:command=issue_command(raw,now=now,source=source,sequence=state.get('sequence',0)+1,
            plan=plan if state['armed'] and source=='IndependentTestControl' else None,sim_s=0 if m is None else m['sim_s'])
        except ValueError as exc:
            gate.stop(str(exc),0 if m is None else m['sim_s'],now)
            command=issue_command(STOP,now=now,source='SafetyStop',sequence=state.get('sequence',0)+1)
        state['sequence']=command['sequence'];state['issued']=command
        atomic(folder/'execution_state.json',state)
        try:ego.apply_control(carla.VehicleControl(**command['control']))
        except Exception as exc:
            append(folder/'actuation.jsonl',dict(command,status='CommandNotDelivered',error=repr(exc)));raise
        command=dict(command,status='Submitted_ResponseUnverified',call_end_wall_s=time.monotonic())
        append(folder/'actuation.jsonl',command)
        return command


def navigation_live(owner,inputs,m,planner,checked,folder):
    # Delayed IMU stays paired to its original snapshot; no current-state fill.
    for frame in sorted(inputs.imu):
        if frame in checked or frame not in inputs.states:continue
        checked.add(frame);state=inputs.states[frame];imu=inputs.imu[frame]
        if abs(imu['timestamp']-state['sim_s'])>1e-5:raise RuntimeError('IMUSnapshotMismatch')
        target,command=planner.run_step(np.array(state['position'][:2]))
        row=dict(ego_world_matrix=state['pose'],target_world_xy=target,imu=imu['values'])
        measured=measured_navigation_target(row);geometric=native_target(state['pose'],target)
        error=float(np.max(np.abs(measured-geometric)))
        yaw=state['yaw'];expected=(yaw+np.pi/2)%(2*np.pi)
        compass_error=abs((imu['values'][-1]-expected+np.pi)%(2*np.pi)-np.pi)
        append(folder/'navigation_live.jsonl',dict(frame=frame,target=target.tolist(),command=command.value,
            imu_timestamp=imu['timestamp'],snapshot_timestamp=state['sim_s'],native_measured=measured.tolist(),
            native_geometry=geometric.tolist(),error_m=error,compass_error_rad=compass_error))
        if error>.03 or compass_error>.005:raise RuntimeError('LiveNavigationDirectionConflict')


def run_round(owner,root,case,basic=False):
    import carla
    from types import SimpleNamespace
    cfg=owner.cfg;folder=root/case;folder.mkdir();ego=None;inputs=None;planner_proc=None;wd=None
    gate=SafetyGate(dict(max_speed_mps=2.,speed_cap_brake_threshold_mps=1.7,speed_cap_brake=.4))
    state=dict(armed=False,require_planner=True,issued=None,ack=None,sequence=0)
    atomic(folder/'execution_state.json',state)
    result=dict(case=case,status='Partial',model_forwards=0,model_controls=0)
    m=None;injected=None;decision=None;first_brake=None;first_stop=None;held=0;maxspeed=0.;checked=set();late_checked=False;precondition=False;rows=[];errors=[];expired_inputs=set()
    cap=cfg['basic_round_max_ticks'] if basic else cfg['round_max_ticks']
    try:
        ego=owner.ego();inputs=Inputs(owner,ego,folder)
        write(folder/'physics_initial.json',physics_record(ego,owner.world))
        # Real route-consumer boundary, GPS points derived only from the fixed
        # navigation route. No other actor state/future ego trajectory is used.
        map_=owner.world.get_map();planner=load_planner(Path(owner.c['runtime_root'])/'official-source/leaderboard/team_code/planner.py',map_)
        target=owner.target;gps=map_.transform_to_geolocation(target)
        set_navigation_route(planner,[({'lat':gps.latitude,'lon':gps.longitude,'alt':gps.altitude},SimpleNamespace(value=4))],
            coordinate_type='gps',client_version=cfg['version'],carla_map=map_)
        with (folder/'planner.log').open('x') as f:
            planner_proc=subprocess.Popen([sys.executable,'-m','tools.m8j13_test_planner','--root',str(folder)],stdout=f,stderr=subprocess.STDOUT)
        with (folder/'watchdog.log').open('x') as f:
            wd=subprocess.Popen(['taskset','-c',owner.c['process_cpus']['watchdog'],sys.executable,'-m','tools.m8j13_watchdog','--root',str(folder),'--ego',str(ego.id),'--expected-version',cfg['version'],'--execution-contract'],stdout=f,stderr=subprocess.STDOUT)
        until=time.monotonic()+10
        while not (folder/'planner_ready.json').exists() or not (folder/'watchdog_ready.json').exists():
            if planner_proc.poll() is not None or wd.poll() is not None:raise RuntimeError('IndependentProcessFailed')
            if time.monotonic()>until:raise TimeoutError('ProcessReadiness')
            time.sleep(.01)
        for index in range(cap):
            now=time.monotonic();plan=load(folder/'planner_status.json');raw=dict(steer=0.,throttle=cfg['test_throttle'],brake=0.)
            if index<cfg['settle_ticks']:raw=STOP.copy()
            if index==cfg['settle_ticks']:
                state['armed']=True;atomic(folder/'execution_state.json',state)
                write(folder/'physics_settled.json',physics_record(ego,owner.world))
            if state['armed'] and not gate.reason:
                current=dict(state,planner=plan)
                reason=execution_reason(current,now,manual=(folder/'EMERGENCY_STOP').exists())
                if reason:gate.stop(reason,m['sim_s'],now)
                if wd.poll() is not None:gate.stop('WatchdogProcessExited',m['sim_s'],now)
                reason=round_exit_reason(index,m['speed_norm'],injected=injected is not None,start_deadline_ticks=cfg['start_deadline_ticks'])
                if reason:
                    precondition=reason=='PreconditionsNotMet';gate.stop(reason,m['sim_s'],now)
            if state['armed'] and not gate.reason and injected is None and m['speed']>=cfg['fault_injection_speed_mps']:
                injected=dict(wall_s=now,sim_s=m['sim_s'],frame=m['frame'],speed=m['speed'],position=m['position'])
                append(folder/'fault_chain.jsonl',dict(event='basic_brake_requested' if basic else 'fault_injected',case=case,**injected))
                if basic:gate.stop('BasicBrakeCheck',m['sim_s'],now)
                elif case=='camera_missing':inputs.sensor_map['CAM_FRONT'].stop()
                elif case=='worker_blocked':(folder/'block_planner').touch()
                elif case=='nonfinite_control':raw=dict(steer=0.,throttle=float('nan'),brake=0.)
                elif case=='manual_estop':(folder/'EMERGENCY_STOP').touch()
            if state['armed']:
                # Independent test motion does not consume camera observations.
                # Count/reject stale groups throughout, but arm camera-dependent
                # protection in the designated injected case. Thus a preceding
                # zero-timeout camera streak is not an admission requirement.
                for frame,old in sorted(inputs.states.items()):
                    if frame in expired_inputs:continue
                    if len(inputs.groups[frame])!=6 and now-old['input_wall_s']>.25:
                        expired_inputs.add(frame);append(folder/'input_rejections.jsonl',dict(frame=frame,reason='InputWaitExpired',input_wall_s=old['input_wall_s'],decision_wall_s=now))
                        if injected and case=='camera_missing' and frame>injected['frame']:
                            gate.stop('InputWaitExpired',m['sim_s'],now)
            fault=load(folder/'watchdog_fault.json')
            if fault:gate.stop(fault['reason'],m['sim_s'],now)
            u=gate.command(raw,speed=0 if m is None else m['speed'],sim_s=0 if m is None else m['sim_s'],wall_s=now)
            if gate.reason and decision is None:
                decision=dict(wall_s=now,sim_s=m['sim_s'],frame=m['frame'],position=m['position'],reason=gate.reason)
                append(folder/'fault_chain.jsonl',dict(event='normal_invalidated_stop_generated',command=STOP,**decision))
            if gate.reason and (folder/'late_plan.json').exists() and not late_checked:
                stale=load(folder/'late_plan.json');check=gate.command(dict(steer=0.,throttle=.5,brake=0.),speed=m['speed'],sim_s=m['sim_s'],wall_s=now)
                assert check==STOP;late_checked=True
                append(folder/'fault_chain.jsonl',dict(event='late_plan_rejected',wall_s=now,original_capture_wall_s=stale['capture_wall_s']))
            # In two cases the separate watchdog must deliver the first brake.
            # Owner stays alive and keeps ticking, but does not renew throttle.
            suppress=injected is not None and not basic and case in ('worker_blocked','manual_estop')
            if suppress and not (folder/'watchdog_events.jsonl').exists():
                issued=state['issued']
            else:
                issued=issue(owner,ego,folder,u,'SafetyStop' if gate.reason else 'IndependentTestControl',state,gate,m,plan)
            snap,m=owner.tick(ego,case,folder);rows.append(m);inputs.snapshot(snap,m)
            matched=feedback_matches(issued['control'],m['full_control'])
            wd_events=load_watchdog(folder)
            if not matched and wd_events and m['full_control']['brake']>0 and m['full_control']['throttle']==0:
                feedback_source='IndependentWatchdog';matched=True
            else:feedback_source=issued['source']
            ack=dict(sequence=issued['sequence'],confirmed_wall_s=time.monotonic(),frame=m['frame'],sim_s=m['sim_s'],matched=matched,source=feedback_source)
            state['ack']=ack;atomic(folder/'execution_state.json',state);append(folder/'acknowledgements.jsonl',ack)
            if not matched:gate.stop('ControlFeedbackMismatch',m['sim_s'],time.monotonic())
            atomic(folder/'planner_request.json',dict(sequence=index,capture_wall_s=m['input_wall_s'],capture_sim_s=m['sim_s'],frame=m['frame']))
            try:
                inputs.drain();navigation_live(owner,inputs,m,planner,checked,folder)
                if inputs.collisions:raise RuntimeError('CollisionObserved')
            except Exception as exc:
                errors.append(repr(exc));gate.stop(str(exc),m['sim_s'],time.monotonic())
            fault=load(folder/'watchdog_fault.json')
            if fault and decision is None:
                # The watchdog may have braked while world.tick was blocked.
                # Keep its first reason/time and this measured observation;
                # never substitute a later owner-side planner-expiry decision.
                gate.stop(fault['reason'],m['sim_s'],time.monotonic())
                decision=dict(reason=fault['reason'],wall_s=fault['wall_s'],sim_s=m['sim_s'],frame=m['frame'],position=m['position'],
                    simulation_time_semantics='First snapshot observed after independent decision; not exact fault simulation time')
                append(folder/'fault_chain.jsonl',dict(event='independent_stop_observed',**decision))
            if index>=cfg['settle_ticks']:maxspeed=max(maxspeed,m['speed_norm'])
            if index>=cfg['settle_ticks'] and m['speed_norm']>2.:gate.stop('TestSpeedLimitExceeded',m['sim_s'],time.monotonic())
            if decision and m['full_control']['brake']>0 and m['full_control']['throttle']==0:
                if first_brake is None:
                    first_brake=m;append(folder/'fault_chain.jsonl',dict(event='brake_feedback',frame=m['frame'],sim_s=m['sim_s'],wall_s=time.monotonic(),source=feedback_source))
                if m['speed_norm']<=.1:
                    if first_stop is None:first_stop=m
                    held+=1
                else:held=0;first_stop=None
            else:held=0
            inputs.prune(m['frame'])
            if held>=21 and (case!='worker_blocked' or late_checked or precondition or errors):break
        for _ in range(min(cfg['tail_ticks'],cap-len(rows))):
            issue(owner,ego,folder,STOP,'SafetyStop',state,gate,m,None)
            snap,m=owner.tick(ego,'brake_tail',folder);rows.append(m);inputs.snapshot(snap,m)
            try:inputs.drain()
            except Exception as exc:errors.append(repr(exc))
        if held>=21:
            assert gate.command(dict(steer=0.,throttle=.5,brake=0.),speed=m['speed'],sim_s=m['sim_s'],wall_s=time.monotonic())==STOP
        write(folder/'physics_final.json',physics_record(ego,owner.world))
        expected='BasicBrakeCheck' if basic else {'camera_missing':'InputWaitExpired','worker_blocked':'PlannerWatchdogExpired','nonfinite_control':'NonfiniteControl','manual_estop':'ManualEmergencyStop'}[case]
        expected_reasons=[expected]+(['PlanWallExpired','ControlCommandExpired'] if case=='worker_blocked' else [])
        success=bool(injected and decision and decision['reason'] in expected_reasons and first_brake and held>=21 and maxspeed<=2. and not errors)
        if case in ('worker_blocked','manual_estop') and not basic:success=success and bool(load_watchdog(folder))
        if case=='worker_blocked':success=success and late_checked
        result.update(status='Passed' if success else 'PreconditionsNotMet' if precondition else 'Partial',
            injected=injected,decision=decision,brake_observed=first_brake is not None,held_stop=held>=21,stop_hold_sim_s=max(0,held-1)*.05,
            stop_time_sim_s=None if first_stop is None or decision is None else first_stop['sim_s']-decision['sim_s'],
            stop_distance_m=None if first_stop is None or decision is None else float(np.linalg.norm(np.array(first_stop['position'][:2])-decision['position'][:2])),
            max_speed_mps=maxspeed,terminal_speed_mps=m['speed_norm'],ticks=len(rows),collisions=len(inputs.collisions),
            live_navigation_checks=len(checked),input_wait_rejections=len(expired_inputs),errors=errors,late_plan_rejected=late_checked,watchdog_brake_submissions=len(load_watchdog(folder)),
            physical_stop_verified=bool(first_brake and held>=21),fault_stop_credit=success and not basic)
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),delivery='ResponseUnverified')
        # Bounded best effort on interface/initialization failure. No unobserved
        # RPC or teardown is labelled a physical stop.
        if ego is not None:
            try:
                gate.stop('ExceptionExit',0 if m is None else m['sim_s'],time.monotonic())
                issue(owner,ego,folder,STOP,'SafetyStop',state,gate,m,None)
                for _ in range(min(40,cap-len(rows),owner.cfg['run_tick_cap']-owner.tick_count)):
                    snap,m=owner.tick(ego,'exception_brake',folder);rows.append(m)
            except Exception as brake_exc:result['exception_brake_error']=repr(brake_exc)
    finally:
        if inputs is not None:inputs.close()
        for name in ('planner_stop','watchdog_stop'):(folder/name).touch()
        for proc in (planner_proc,wd):
            if proc is not None:
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
        owner.destroy_owned();write(folder/'result.json',result)
    return result


def load_watchdog(folder):
    p=folder/'watchdog_events.jsonl'
    if not p.exists():return []
    # Another process may currently append the next line.
    return [json.loads(x) for x in p.read_text().splitlines() if x.endswith('}')]


def worker(root,phase):
    cfg=json.loads(CFG.read_text());cfg['run_tick_cap']=cfg[phase+'_total_tick_cap']-cfg['settings_tick_charge']
    owner=None;result=dict(status='Partial',phase=phase,model_forwards=0,model_controls=0,training=0)
    try:
        owner=TestOwner.__new__(TestOwner);owner.__init__(root,cfg)
        cases=['basic_start_brake'] if phase=='basic' else cfg['fault_cases'];rounds=[]
        for case in cases:
            r=run_round(owner,root,case,basic=phase=='basic');rounds.append(r)
            if r['status']!='Passed':break
        result.update(rounds=rounds,status='Passed' if len(rounds)==len(cases) and all(r['status']=='Passed' for r in rounds) else 'Partial')
    except BaseException as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if owner is not None:
            result['explicit_ticks']=getattr(owner,'tick_count',0);result['charged_ticks']=result['explicit_ticks']+30
            try:result['cleanup']=owner.close()
            except Exception as exc:result.update(status='Partial',cleanup_error=repr(exc))
        write(root/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def supervise(phase):
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    from tools.m8j13_supervisor import terminate
    cfg=json.loads(CFG.read_text());c=config();base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    if phase=='faults':
        basic=load(base/'basic'/'result.json');sup=load(base/'basic'/'supervisor.json')
        if not basic or basic['status']!='Passed' or not sup or not sup['cleanup_complete']:raise RuntimeError('BasicGateNotPassed')
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=base/'budget.json';ledger=load(path,dict(runs=[]))
        cap=cfg[phase+'_total_tick_cap']
        if len(ledger['runs'])>=2 or sum(x['reserved_ticks'] for x in ledger['runs'])+cap>2400:raise RuntimeError('BudgetExhausted')
        if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
        root=base/phase;root.mkdir()
        write(root/'request.json',dict(phase=phase,config=cfg,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            config_sha256=digest(CFG),planned_tick_upper_bound=cap,model_forwards=0,model_controls=0,
            purpose='Native GPS consumer + live IMU + independent start/brake' if phase=='basic' else 'Four fixed moving faults with acknowledged execution and held stop',
            stop_conditions=['Budget','NavigationConflict','ControlSourceConflict','NoPhysicalBraking','First dependent system fault']))
        ledger['runs'].append(dict(phase=phase,reserved_ticks=cap));atomic(path,ledger)
        server=None;child=None;report={};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=cfg['python_deps']+':'+str(Path.cwd()))
        try:
            cmd=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic']
            with (root/'server.log').open('x') as f:server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=time.monotonic()+150
            while time.monotonic()<until:
                if server.poll() is not None:raise RuntimeError('ServerEarlyExit')
                client=carla.Client('127.0.0.1',c['port']);client.set_timeout(2.)
                try:
                    versions=[client.get_client_version(),client.get_server_version()]
                    if versions!=[cfg['version']]*2:raise ValueError('ClientServerVersionMismatch')
                    world=client.get_world();report['client']=dict(python=sys.executable,path=carla.libcarla.__file__,versions=versions,port=c['port'],episode=world.id);break
                except RuntimeError:time.sleep(1.)
            else:raise TimeoutError('ServerReadiness')
            with (root/'worker.log').open('x') as f:
                child=subprocess.Popen(['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_navigation_stop','--worker-root',str(root),'--phase',phase],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                until=time.monotonic()+900
                while child.poll() is None:
                    if server.poll() is not None:raise RuntimeError('ServerExited')
                    if time.monotonic()>until:raise TimeoutError('WorkerDeadline')
                    time.sleep(.2)
            report['worker_exit_code']=child.returncode;report['result']=load(root/'result.json')
        except BaseException as exc:report['error']=repr(exc)
        finally:
            terminate(child);terminate(server);until=time.monotonic()+8
            while time.monotonic()<until and (any(listening(p) for p in range(c['port'],c['port']+3)) or (server and group_members(server.pid))):time.sleep(.1)
            report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)}
            report['groups_after']=[group_members(p.pid) for p in (server,child) if p]
            report['cleanup_complete']=not any(report['ports_after'].values()) and not any(report['groups_after'])
            ledger['runs'][-1]['charged_ticks']=(report.get('result') or {}).get('charged_ticks',cap);atomic(path,ledger)
            write(root/'supervisor.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['basic','faults']);p.add_argument('--worker-root',type=Path);a=p.parse_args();os.umask(0o077)
    if a.worker_root:worker(a.worker_root,a.phase)
    else:supervise(a.phase)


if __name__=='__main__':main()
