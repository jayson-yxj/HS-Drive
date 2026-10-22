"""Actual Tesla actuator probes with unchanged physics; measurements not filtered away."""
import json
from pathlib import Path
import numpy as np, yaml, carla
from simulator.session import CarlaSession
from tools.m47_common import sample_vehicle,physics_dict,dynamics_metrics,jsonl,freeze_sources
from tools.m6b_common import write,provenance,asset


def main():
    root=Path('reports/m6b_calibration_artifacts');root.mkdir()
    freeze_sources(root/'source',[__file__,'tools/m47_common.py','configs/m6b/protocol.json'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03')
    records=[]
    for repeat in [0,1]:
        for mode in ['constant_010','constant_015','ramp_coast_hold']:
            with CarlaSession(cfg,load_map=True) as session:
                w=session.world;ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),w.get_map().get_spawn_points()[27])
                ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
                for _ in range(30):w.tick()
                rows=[];hold=False;start=w.get_snapshot().timestamp.elapsed_seconds
                for tick in range(240):
                    r=sample_vehicle(w,ego);t=tick*.05
                    if mode.startswith('constant'):throttle=.1 if mode.endswith('010') else .15
                    else:throttle=min(.18,t*.025)
                    if t>=4:throttle=0
                    if t>=4 and abs(r['long_speed'])<.02:hold=True
                    u=dict(throttle=throttle,brake=.12 if hold else 0.,steer=0.)
                    r.update(command=u,source='Privileged',elapsed=r['t']-start,hold=hold);rows.append(r)
                    ego.apply_control(carla.VehicleControl(**u));w.tick()
                key=f'{mode}_r{repeat}';jsonl(root/(key+'.jsonl'),rows);write(root/(key+'_physics.json'),physics_dict(ego))
                records.append(dict(name=key,metrics=dynamics_metrics(rows),max_speed=max(r['speed'] for r in rows),final_speed=rows[-1]['speed'],log=asset(root/(key+'.jsonl')),physics=asset(root/(key+'_physics.json'))))
                print(key,records[-1]['max_speed'],records[-1]['metrics']['max_decel_raw'],flush=True)
    write('reports/m6b_longitudinal_calibration.json',dict(**provenance('actual_CARLA_controlled_actuator_probes'),source='Privileged',records=records,
        historical=[asset('reports/m47_dynamics_calibration.json'),asset('reports/m48_dynamics_feasibility.json')],
        limits='No PhysicsControl changes; raw and 250ms metrics separate. Finite probes do not prove infeasibility of all controls. Coast model is approximate, not exact inverse dynamics.'))

if __name__=='__main__':main()
