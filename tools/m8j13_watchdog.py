"""Independent braking client. Never ticks the world or imports the model."""
import argparse,json,os,time,fcntl
from pathlib import Path
from simulator.m8j13_safety import watchdog_reason
from tools.m8j13_common import config,append,atomic


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--ego',type=int,required=True)
    p.add_argument('--expected-version',default='0.9.15')
    p.add_argument('--execution-contract',action='store_true');a=p.parse_args()
    c=config();root=Path(a.root);client=carla.Client('127.0.0.1',c['port']);client.set_timeout(.2)
    # Establish the separate RPC connection before sensors and timed acceptance.
    versions=(client.get_client_version(),client.get_server_version())
    if versions!=(a.expected_version,a.expected_version):raise RuntimeError('WatchdogVersionMismatch')
    world=client.get_world();ego=world.get_actor(a.ego)
    if ego is None:raise RuntimeError('WatchdogEgoMissing')
    import sys
    latched=None;last_brake=0;armed=False;atomic(root/'watchdog_ready.json',dict(pid=os.getpid(),wall_s=time.monotonic(),
        python=sys.executable,carla_path=carla.libcarla.__file__,client_version=versions[0],server_version=versions[1],
        host='127.0.0.1',port=c['port'],episode=world.id,ego_id=ego.id,tick_calls=0))
    while not (root/'watchdog_stop').exists():
        now=time.monotonic()
        try:
            if a.execution_contract:
                from simulator.m8j13_execution import execution_reason
                state=json.loads((root/'execution_state.json').read_text())
                planner=root/'planner_status.json'
                state['planner']=json.loads(planner.read_text()) if planner.exists() else None
                reason=execution_reason(state,now,manual=(root/'EMERGENCY_STOP').exists())
            else:
                heartbeat=json.loads((root/'heartbeat.json').read_text())
                armed=armed or heartbeat.get('armed',False)
                reason=watchdog_reason(heartbeat,now,c['watchdog_heartbeat_timeout_s'],(root/'EMERGENCY_STOP').exists()) if armed else None
        except (OSError,ValueError,KeyError):reason='InvalidHeartbeat'
        # Serialize latch publication and actuation with the opt-in owner. An
        # owner cannot race a previously checked normal command past this stop.
        with (root/'actuation.lock').open('a') as lock:
            if a.execution_contract:fcntl.flock(lock,fcntl.LOCK_EX)
            if reason and latched is None:
                latched=reason;atomic(root/'watchdog_fault.json',dict(reason=reason,wall_s=now))
                if a.execution_contract:
                    append(root/'watchdog_decisions.jsonl',dict(reason=reason,decision_wall_s=now,execution_state=state))
            if latched and now-last_brake>=.05:
                result=dict(reason=latched,wall_s=now)
                try:
                    ego.apply_control(carla.VehicleControl(throttle=0.,brake=1.,steer=0.))
                    result.update(submitted=True,call_end_wall_s=time.monotonic(),source='IndependentWatchdog',
                        command=dict(throttle=0.,brake=1.,steer=0.))
                except Exception as exc:result.update(submitted=False,status='CommandNotDelivered',error=repr(exc))
                append(root/'watchdog_events.jsonl',result);last_brake=now
        time.sleep(c['watchdog_poll_wall_s'])


if __name__=='__main__':main()
