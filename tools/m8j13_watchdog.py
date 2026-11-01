"""Independent braking client. Never ticks the world or imports the model."""
import argparse,json,os,time
from pathlib import Path
from simulator.m8j13_safety import watchdog_reason
from tools.m8j13_common import config,append,atomic


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--ego',type=int,required=True);a=p.parse_args()
    c=config();root=Path(a.root);client=carla.Client('127.0.0.1',c['port']);client.set_timeout(.2)
    # Establish the separate RPC connection before sensors and timed acceptance.
    client.get_server_version();ego=client.get_world().get_actor(a.ego)
    if ego is None:raise RuntimeError('WatchdogEgoMissing')
    latched=None;last_brake=0;armed=False;atomic(root/'watchdog_ready.json',dict(pid=os.getpid(),wall_s=time.monotonic()))
    while not (root/'watchdog_stop').exists():
        now=time.monotonic()
        try:
            heartbeat=json.loads((root/'heartbeat.json').read_text())
            armed=armed or heartbeat.get('armed',False)
            reason=watchdog_reason(heartbeat,now,c['watchdog_heartbeat_timeout_s'],(root/'EMERGENCY_STOP').exists()) if armed else None
        except (OSError,ValueError,KeyError):reason='InvalidHeartbeat'
        if reason and latched is None:
            latched=reason;atomic(root/'watchdog_fault.json',dict(reason=reason,wall_s=now))
        if latched and now-last_brake>=.05:
            result=dict(reason=latched,wall_s=now)
            try:
                ego.apply_control(carla.VehicleControl(throttle=0.,brake=1.,steer=0.))
                result['submitted']=True
            except Exception as exc:result.update(applied=False,error=repr(exc))
            append(root/'watchdog_events.jsonl',result);last_brake=now
        time.sleep(c['watchdog_poll_wall_s'])


if __name__=='__main__':main()
