"""Finite natural-initialization Tesla response probes, original Physics unchanged."""
import json,math
from pathlib import Path
import numpy as np,yaml,carla
from simulator.session import CarlaSession
from simulator.m6c_geometry import road_grade
from tools.m47_common import sample_vehicle,physics_dict,dynamics_metrics,jsonl,freeze_sources
from tools.m6c_common import write,provenance,asset


def tasks():
    for split,positions in [('development',[27,126]),('validation',[66,170])]:
        for spawn in positions:
            for seed in [101,102]:
                for gear in ['auto','manual1']:
                    for throttle in [.10,.15]:
                        yield dict(split=split,spawn=spawn,seed=seed,mode='startup',gear=gear,throttle=throttle)
                for speed in [1.,2.5]:
                    for brake in [0.,.01,.05,1.]:
                        yield dict(split=split,spawn=spawn,seed=seed,mode='stop',gear='auto',speed=speed,brake=brake)
                for mode in ['warm_start','maintain']:
                    yield dict(split=split,spawn=spawn,seed=seed,mode=mode,gear='auto',throttle=.15 if mode=='warm_start' else .10)


def main():
    plan=list(tasks());write('reports/m6c_probe_plan.json',dict(**provenance('predeclared_real_CARLA_probe_plan'),tasks=plan))
    root=Path('reports/m6c_probe_artifacts');root.mkdir();freeze_sources(root/'source',[__file__,'simulator/m6c_geometry.py','tools/m47_common.py','configs/m6c/protocol.json'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03');summaries=[]
    for number,task in enumerate(plan):
        key=f"{task['split']}_p{task['spawn']}_s{task['seed']}_{task['mode']}_{task['gear']}_{task.get('throttle',task.get('speed'))}_{task.get('brake','x')}"
        with CarlaSession(cfg,load_map=True) as session:
            w=session.world;m=w.get_map();w.set_weather(carla.WeatherParameters.ClearNoon)
            ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),m.get_spawn_points()[task['spawn']])
            ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(30):w.tick()
            rows=[];phase='probe' if task['mode']=='startup' else 'prepare';phase_tick=0;settled=0;probe_start=None
            initial=sample_vehicle(w,ego);prepared=False
            for tick in range(850):
                r=sample_vehicle(w,ego);speed=r['speed'];u=dict(throttle=0.,brake=0.,steer=0.,manual_gear_shift=task['gear']=='manual1',gear=1 if task['gear']=='manual1' else 0)
                if phase=='prepare':
                    u['throttle']=.2
                    target=task.get('speed',1.)
                    if speed>=target:
                        phase='warm_stop' if task['mode']=='warm_start' else 'probe';phase_tick=0;prepared=True
                if phase=='warm_stop':
                    u.update(throttle=0.,brake=1.)
                    settled=settled+1 if speed<.02 else 0
                    if settled>=25:phase='probe';phase_tick=0
                if phase=='probe':
                    if probe_start is None:probe_start=len(rows)
                    mode=task['mode']
                    if mode in ['startup','warm_start']:u.update(throttle=task['throttle'],brake=0.)
                    elif mode=='maintain':u.update(throttle=.10 if phase_tick<80 else 0.,brake=0.)
                    else:u.update(throttle=0.,brake=task['brake'])
                    if mode in ['startup','warm_start'] and phase_tick>=120:phase='coast';phase_tick=0
                    elif mode=='stop' and speed<.02:phase='hold';phase_tick=0
                    elif mode=='maintain' and phase_tick>=80 and speed<.02:phase='hold';phase_tick=0
                if phase=='coast':
                    u.update(throttle=0.,brake=0.)
                    if speed<.02:phase='hold';phase_tick=0
                if phase=='hold':u.update(throttle=0.,brake=.12)
                r.update(command=u,phase=phase,phase_tick=phase_tick,source='Privileged',road=road_grade(m,r['position']))
                rows.append(r);ego.apply_control(carla.VehicleControl(**u));w.tick();phase_tick+=1
                if phase=='hold' and phase_tick>=40:break
                if probe_start is not None and len(rows)-probe_start>=400:break
                if probe_start is None and tick>=350:break
            physics=physics_dict(ego);write(root/(key+'_physics.json'),physics);jsonl(root/(key+'.jsonl'),rows)
            summary=dict(name=key,task=task,initial=initial,prepared=prepared,probe_start=probe_start,ticks=len(rows),
                reached_hold=phase=='hold',final_speed=rows[-1]['speed'],metrics=dynamics_metrics(rows),trace=asset(root/(key+'.jsonl')),physics=asset(root/(key+'_physics.json')))
            summaries.append(summary);print(number+1,len(plan),key,'start',probe_start,'hold',phase=='hold','v',round(rows[-1]['speed'],3),flush=True)
    write('reports/m6c_low_speed_dynamics.json',dict(**provenance('actual_CARLA_original_Physics_natural_actuator_probes'),source='Privileged',trials=summaries,
        limits='Natural preparation avoids set_target_velocity. Manual1 is only diagnostic control setting, not Physics modification or default controller change. Two seeds do not create independent physical worlds. Max400 probe ticks; non-stopping observations are censored, never treated as zero distance.'))

if __name__=='__main__':main()
