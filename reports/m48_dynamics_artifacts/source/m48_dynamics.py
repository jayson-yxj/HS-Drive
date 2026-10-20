"""Repeat low-speed CARLA probes with original physics and isolated gear commands."""
import argparse,json,math
from pathlib import Path
import numpy as np,yaml
import carla
from simulator.session import CarlaSession
from simulator.oracle_driver import PurePursuitPID,local_trajectory
from tools.run_oracle_closedloop import find_routes,state_of
from tools.m47_common import physics_dict,sample_vehicle,dynamics_metrics,jsonl,freeze_sources
from simulator.recorder import atomic_json,file_hash


def cases():
    result=[]
    for repeat in range(3):
        for brake in [0.,.0001,.001,.005,1.]:
            result.append(dict(name=f'v1_b{brake}_r{repeat}',v=1.,brake=brake,kind='constant',duration=10))
    for speed in [.1,.2,.5,1.5]:
        result.append(dict(name=f'low_v{speed}',v=speed,brake=.001,kind='constant',duration=6))
    for kind in ['neutral','manual_first','throttle_overlap','hold_at_002','hold_at_008']:
        result.append(dict(name=kind,v=1.,brake=.001,kind=kind,duration=12))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m48_dynamics_feasibility.json');p.add_argument('--artifacts',default='reports/m48_dynamics_artifacts');a=p.parse_args()
    if Path(a.output).exists() or Path(a.artifacts).exists():raise FileExistsError('new paths required')
    root=Path(a.artifacts);root.mkdir();sources=freeze_sources(root/'source',[__file__,'tools/m47_common.py','simulator/session.py'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03')
    results=[];baseline=json.loads(Path('reports/m47_dynamics_v3_artifacts/physics.json').read_text())
    for case in cases():
        with CarlaSession(cfg,load_map=True) as session:
            w=session.world;w.set_weather(carla.WeatherParameters.ClearNoon)
            _,spawn,path=find_routes(w.get_map(),80)['straight'];route=np.array([[p.transform.location.x,-p.transform.location.y,-math.radians(p.transform.rotation.yaw)] for p in path])
            ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),spawn)
            ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(30):w.tick()
            physics=physics_dict(ego);assert physics==baseline
            ctl=PurePursuitPID();warm=[]
            # For very low speeds use an actual measured launch crossing, retain it.
            for tick in range(600):
                st=state_of(w.get_snapshot(),ego);tr,_=local_trajectory(route,st,max_speed=case['v']);u=ctl.command(tr,st)
                if case['v']<.5:u['throttle']=.15;u['brake']=0
                if (case['v']<.5 and st.speed_mps>=case['v']) or (case['v']>=.5 and len(warm)>12 and all(abs(r['speed']-case['v'])<.3 for r in warm[-10:])):break
                r=sample_vehicle(w,ego);r['command']=u;warm.append(r)
                ego.apply_control(carla.VehicleControl(throttle=u['throttle'],brake=u['brake'],steer=u['steer']));w.tick()
            jsonl(root/(case['name']+'_warmup.jsonl'),warm)
            rows=[];start=w.get_snapshot().timestamp.elapsed_seconds;hold=False
            for tick in range(int(case['duration']/.05)+1):
                r=sample_vehicle(w,ego);st=state_of(w.get_snapshot(),ego);tr,_=local_trajectory(route,st,max_speed=2);u=ctl.command(tr,st)
                command=dict(throttle=0.,brake=case['brake'],steer=u['steer'],hand_brake=False,manual_gear_shift=False,gear=1)
                if case['kind']=='neutral':command.update(manual_gear_shift=True,gear=0)
                if case['kind']=='manual_first':command.update(manual_gear_shift=True,gear=1)
                if case['kind']=='throttle_overlap':command['throttle']=.02
                if case['kind'].startswith('hold_at'):
                    threshold=.02 if case['kind'].endswith('002') else .08
                    hold|=abs(r['long_speed'])<=threshold
                    command['brake']=.1 if hold else 0.
                r.update(elapsed=r['t']-start,command=command,hold=hold);rows.append(r)
                ego.apply_control(carla.VehicleControl(**command));w.tick()
            log=root/(case['name']+'.jsonl');jsonl(log,rows);atomic_json(root/(case['name']+'_physics.json'),physics)
            metrics=dynamics_metrics(rows);v=np.array([r['long_speed'] for r in rows]);pos=np.array([r['position'] for r in rows]);holdrow=next((r for r in rows if r['hold']),None)
            result=dict(case=case,initial_speed=float(v[0]),final_speed=float(v[-1]),travel_m=float(np.linalg.norm(pos[-1]-pos[0])),metrics=metrics,
                first_hold=None if holdrow is None else dict(speed=holdrow['long_speed'],t=holdrow['elapsed']),
                raw_comfort_pass=metrics['max_decel_raw']<=1.8 and metrics['jerk_p95_moving']<=2.5 and metrics['jerk_peak_raw']<=5,
                final_1s_stable=bool((abs(v[-20:])<.02).all()),log=str(log),sha256=file_hash(log),physics_unchanged=True)
            results.append(result);atomic_json(root/(case['name']+'_result.json'),result);print(case['name'],result['initial_speed'],metrics['max_decel_raw'],metrics['jerk_peak_raw'],flush=True)
    atomic_json(a.output,dict(kind='actual_carla_low_speed_feasibility_probes',results=results,sources=sources,original_46_report_sha256=file_hash('reports/m47_vehicle_dynamics.json'),
        physics_sha256=file_hash('reports/m47_dynamics_v3_artifacts/physics.json'),fixed_dt=.05,physics_substep=.01,
        formal_feasible_stopping_demonstrated=False,impossibility_proven=False,
        limitations=['Finite actuator probes cannot prove nonexistence of all feasible controls.','No wheel contact force or slip telemetry exposed in current Python recording; contact inferred from settled pose only.','Gear commands alter operation, not original VehiclePhysicsControl.','No PID retuning or modified vehicle physics.']))
if __name__=='__main__':main()
