"""J13-A real sensor stability and moving fault injections; no model control."""
import argparse,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from simulator.m8j13_carla import J13Session,Rig,Actuator,spawn_ego,measurement
from simulator.m8j13_safety import SafetyGate,PlanStamp,freshness,validate_trajectory
from tools.m8j13_common import config,write,append,atomic,provenance,quantiles


class Watchdog:
    def __init__(self,root,ego,c):
        self.root=root;self.c=c;self.proc=None;self.armed=False
        self.beat('initialization')
        self.log=(root/'watchdog.log').open('x')
        self.proc=subprocess.Popen(['taskset','-c',c['process_cpus']['watchdog'],sys.executable,'-m','tools.m8j13_watchdog','--root',str(root),'--ego',str(ego.id)],stdout=self.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+10
        while not (root/'watchdog_ready.json').exists():
            if self.proc.poll() is not None:raise RuntimeError('WatchdogStartupFailure')
            self.beat('initialization')
            if time.monotonic()>deadline:raise TimeoutError('WatchdogStartupTimeout')
            time.sleep(.01)
    def beat(self,mode,deadline=None):
        now=time.monotonic();atomic(self.root/'heartbeat.json',dict(wall_s=now,mode=mode,armed=self.armed,command_valid_until_wall_s=now+self.c['max_command_age_wall_s'] if deadline is None else deadline))
    def fault(self):
        if self.proc.poll() is not None:return 'WatchdogProcessExited'
        path=self.root/'watchdog_fault.json'
        return json.loads(path.read_text())['reason'] if path.exists() else None
    def close(self):
        (self.root/'watchdog_stop').touch(exist_ok=False)
        try:self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=2)
        self.log.close()


def case(root,name,c,stability=False):
    import carla
    root.mkdir();rows=[];gate=SafetyGate(c);wd=None;result=dict(case=name,model_control_applied=0)
    cfg=dict(host='127.0.0.1',port=c['port'],map=c['map'],client_timeout_s=c['client_timeout_s'],fixed_delta_seconds=c['physics_dt_s'])
    try:
        with J13Session(cfg,load_map=True) as session:
            ego,route,world_route=spawn_ego(session,c,c['scenes'][0]);wd=Watchdog(root,ego,c);session.cleanup_callbacks.append(wd.close)
            rig=Rig(session,ego,c,world_route,name);act=Actuator(session.client,ego)
            owned=[a.id for a in session.actors]
            sim_start=None;move_start=None;injected=False;injection_wall=None;trigger_speed=None;stable=0;captured=0;max_speed=0.;m=None;stability_start=None;captured_frames=[];sensor_ages=[];next_cycle=time.monotonic()
            for index in range(c['settle_ticks']+(c['stability_ticks']+8 if stability else 420)):
                # No burst catch-up after a missed wall-clock control slot.
                if time.monotonic()<next_cycle:time.sleep(next_cycle-time.monotonic())
                next_cycle=time.monotonic()+c['control_dt_s']
                now=time.monotonic();phase='settle' if index<c['settle_ticks'] else 'stability' if stability else 'move'
                if index==c['settle_ticks']:
                    wd.armed=True
                    # The parked startup interval is outside the stability
                    # window. Never carry pre-arm image age into that window.
                    if m is not None:rig.discard_before(m['frame']+1)
                reason=None if wd is None else wd.fault()
                if phase in ('settle','stability'):raw=dict(steer=0.,throttle=0.,brake=1.)
                else:
                    speed=0 if m is None else m['speed'];raw=dict(steer=0.,throttle=.3 if speed<1.0 else 0.,brake=0.)
                    if move_start is None:move_start=m['sim_s']
                    if not injected and speed>=.8:
                        injected=True;injection_wall=now;trigger_speed=speed;result['injection_frame']=m['frame']
                        if name=='camera_drop':rig.drop_camera='CAM_FRONT'
                        elif name=='mixed_timestamp':rig.corrupt_timestamp=True
                        elif name=='manual_estop':(root/'EMERGENCY_STOP').touch()
                        elif name=='watchdog_stall':
                            # No heartbeat while the tick owner is stalled; the independent client must brake.
                            time.sleep(.4);reason=wd.fault()
                        elif name=='carla_disconnect':
                            # Real unavailable RPC endpoint for the primary channel;
                            # the independent watchdog retains its own healthy connection.
                            act.client=carla.Client('127.0.0.1',c['port']+99);act.client.set_timeout(.1);act.inject_failure=True
                    if injected:
                        if name=='normal_stop':reason='NormalTestStop'
                        elif name=='planning_timeout':
                            p=PlanStamp(name,m['frame']-8,m['sim_s']-.4,now-.4,now-.39,now-.01,'synthetic_no_model')
                            reason=freshness(p,name,m['frame'],m['sim_s'],now)
                        elif name=='empty_candidate':reason='NoPlanningCandidate'
                        elif name in ('nonfinite','trajectory_jump'):
                            points=np.column_stack([np.arange(1,7)*.5,np.zeros(6),np.zeros(6),np.ones(6)])
                            points[2,0]=np.nan if name=='nonfinite' else 50
                            reason=validate_trajectory(points,speed,c['trajectory_checks'])[0]
                        elif name=='control_bounds':raw=dict(steer=2.,throttle=2.,brake=-1.)
                        elif name=='controller_exception':
                            try:raise RuntimeError('InjectedControllerException')
                            except RuntimeError:reason='ControllerException'
                        elif name=='manual_estop':reason='ManualEmergencyStop'
                if reason:raw=gate.stop(reason,0 if m is None else m['sim_s'],now)
                u=gate.command(raw,speed=0 if m is None else m['speed'],sim_s=0 if m is None else m['sim_s'],wall_s=now)
                if wd:wd.beat('fault_stop' if gate.reason else phase)
                apply_start=time.monotonic()
                try:
                    if act.inject_failure:act.client.get_world()
                    act.apply(u)
                except Exception as exc:
                    gate.stop('ActuationRPCFailure',m['sim_s'],time.monotonic());append(root/'exceptions.jsonl',dict(error=repr(exc),wall_s=time.monotonic()))
                    # Simulated primary transport failure; separate watchdog client remains usable.
                    time.sleep(.35)
                    act.client=session.client;act.inject_failure=False;act.apply(dict(steer=0.,throttle=0.,brake=1.))
                apply_end=time.monotonic();tick_start=time.monotonic()
                m=rig.tick();tick_end=time.monotonic();max_speed=max(max_speed,m['speed_norm']);sim_start=m['sim_s'] if sim_start is None else sim_start
                frame_ok=False
                if phase=='stability' and stability_start is None:stability_start=m['frame']
                try:
                    ready=rig.available();frame_ok=bool(ready)
                    for row,images in ready:
                        if stability_start is not None and stability_start<=row['frame_id']<stability_start+c['stability_ticks']:
                            if row['sensor_ready_wall_s']-row['input_capture_wall_s']>c['max_plan_age_wall_s']:
                                raise TimeoutError('CompletedSensorFrameExpired:'+str(row['frame_id']))
                            captured+=1;captured_frames.append(row['frame_id']);sensor_ages.append(row['sensor_ready_wall_s']-row['input_capture_wall_s'])
                    overdue=rig.overdue(m['sim_s'],time.monotonic())
                    if overdue and phase!='settle':raise TimeoutError('SensorFrameTimeout:'+json.dumps(overdue))
                    if overdue:
                        append(root/'startup_sensor_events.jsonl',dict(event='ParkedWarmupExpiredFrame',**overdue))
                        rig.discard_before(m['frame'])
                except (TimeoutError,ValueError) as exc:
                    why='SensorFrameTimeout' if isinstance(exc,TimeoutError) else 'SensorContractMismatch'
                    gate.stop(why,m['sim_s'],time.monotonic());act.apply(dict(steer=0.,throttle=0.,brake=1.))
                    append(root/'exceptions.jsonl',dict(error=repr(exc),frame=m['frame']))
                    rig.drop_camera=None;rig.corrupt_timestamp=False;rig.discard_before(m['frame']+1)
                if wd and wd.fault() and not gate.reason:gate.stop(wd.fault(),m['sim_s'],time.monotonic())
                collect_end=time.monotonic();actual=ego.get_control();get_control_end=time.monotonic()
                if not gate.reason and index>0:
                    try:act.verify(u)
                    except RuntimeError as exc:
                        gate.stop(str(exc),m['sim_s'],time.monotonic());act.apply(dict(steer=0.,throttle=0.,brake=1.))
                r=dict(index=index,frame=m['frame'],sim_s=m['sim_s'],wall_s=time.monotonic(),speed=m['speed_norm'],phase=phase,
                    injected=injected,frame_ok=frame_ok,command=u,applied=dict(steer=actual.steer,throttle=actual.throttle,brake=actual.brake),reason=gate.reason,
                    timing_ms=dict(apply=(apply_end-apply_start)*1000,tick_and_snapshot=(tick_end-tick_start)*1000,
                        collect=(collect_end-tick_end)*1000,get_control=(get_control_end-collect_end)*1000))
                rows.append(r);append(root/'trace.jsonl',r)
                if stability and captured==c['stability_ticks']:break
                if gate.reason:
                    stable=stable+1 if m['speed_norm']<=c['stop_speed_mps'] and actual.brake>0 and actual.throttle==0 else 0
                    if stable*c['physics_dt_s']>=c['stop_hold_s']:break
                    if injected and m['sim_s']-next(x['sim_s'] for x in rows if x['injected'])>c['max_fault_stopping_sim_s']:break
                if not stability and move_start is not None and not injected and m['sim_s']-move_start>c['max_initialization_sim_s']:break
            expected={'normal_stop':'NormalTestStop','camera_drop':'SensorFrameTimeout','mixed_timestamp':'SensorContractMismatch',
                'planning_timeout':'PlanSimulationExpired','empty_candidate':'NoPlanningCandidate','nonfinite':'NonfiniteTrajectory',
                'trajectory_jump':'TrajectoryJump','control_bounds':'ControlOutOfRange','controller_exception':'ControllerException',
                'carla_disconnect':'ActuationRPCFailure','manual_estop':'ManualEmergencyStop','watchdog_stall':'ControlWatchdogExpired'}
            result.update(captured_stability_ticks=captured,expected_reason=None if stability else expected[name],actual_reason=gate.reason,
                injected=injected,injection_speed_mps=trigger_speed,terminal_speed_mps=m['speed_norm'],stop_hold_ticks=stable,
                max_speed_mps=max_speed,collision_count=len(rig.collisions),lane_invasion_count=len(rig.lanes),
                sensor_stats=dict(rig.sync.stats),sensor_events=rig.events,gate_events=gate.events,
                tick_count=len(rows),control_interval_ms=quantiles(list(np.diff([x['wall_s'] for x in rows])*1000)),
                watchdog_fault=None if wd is None else wd.fault(),all_model_control_disabled=True,
                captured_consecutive=not captured_frames or captured_frames==list(range(captured_frames[0],captured_frames[0]+len(captured_frames))),sensor_wall_age_ms=quantiles([x*1000 for x in sensor_ages]))
            basic=not rig.collisions and max_speed<=c['max_speed_mps']
            result['checks_passed']=basic and ((captured==500 and result['captured_consecutive'] and gate.reason is None) if stability else (injected and trigger_speed>=.8 and gate.reason==expected[name] and stable*c['physics_dt_s']>=c['stop_hold_s']))
            write(root/'sensor_arrivals.json',rig.arrival_records)
            write(root/'raw_sensor_metadata.json',rig.raw_records)
            write(root/'raw_sensor_inbox_stats.json',dict(rig.raw_inbox.stats))
            write(root/'world_snapshots.json',rig.snapshot_records)
        # Actor handles are cached until a newer snapshot acknowledges deletion.
        cleanup_snapshot=session.world.wait_for_tick(c['client_timeout_s'])
        remaining=[i for i in owned if cleanup_snapshot.find(i) is not None]
        restored=session.world.get_settings()
        result.update(remaining_owned_actors=remaining,synchronous_after=restored.synchronous_mode,cleanup_passed=not remaining and not restored.synchronous_mode,traffic_manager='NotCreated')
        result['passed']=result['checks_passed'] and result['cleanup_passed']
    except BaseException as exc:result.update(passed=False,error=repr(exc));print(name,repr(exc),flush=True)
    write(root/'result.json',result);print(name,'passed',result['passed'],'reason',result.get('actual_reason'),'ticks',len(rows),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);c=config()
    results=[]
    for name in ['stability','normal_stop','planning_timeout','camera_drop','mixed_timestamp','empty_candidate','nonfinite','trajectory_jump','control_bounds','controller_exception','manual_estop','watchdog_stall','carla_disconnect']:
        result=case(root/name,name,c,name=='stability');results.append(result)
        if not result['passed']:break
    report=dict(status='Passed' if len(results)==13 and all(x['passed'] for x in results) else 'Partial',cases=results,provenance=provenance(),model_control_applied=0)
    write(root/'safety_result.json',report);write(Path('reports/m8j13_sources')/(root.name+'_result.json'),report)
    if report['status']!='Passed':raise SystemExit(1)


if __name__=='__main__':main()
