"""Measured coast-crossing low-speed probes; launch transients remain separate."""
import json,math
from pathlib import Path
import numpy as np,yaml,carla
from simulator.session import CarlaSession
from simulator.oracle_driver import PurePursuitPID,local_trajectory
from tools.run_oracle_closedloop import find_routes,state_of
from tools.m47_common import sample_vehicle,physics_dict,dynamics_metrics,jsonl,freeze_sources
from simulator.recorder import atomic_json,file_hash

def main():
    output=Path('reports/m48_low_speed_followup.json');root=Path('reports/m48_low_speed_artifacts')
    if root.exists() or output.exists():raise FileExistsError('new paths required')
    root.mkdir();sources=freeze_sources(root/'source',[__file__,'tools/m47_common.py'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03');results=[]
    for threshold in [.2,.1,.02]:
        for repeat in range(2):
            with CarlaSession(cfg,load_map=True) as session:
                w=session.world;_,spawn,path=find_routes(w.get_map(),80)['straight'];route=np.array([[p.transform.location.x,-p.transform.location.y,-math.radians(p.transform.rotation.yaw)] for p in path])
                ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),spawn);ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
                for _ in range(30):w.tick()
                assert physics_dict(ego)==json.loads(Path('reports/m47_dynamics_v3_artifacts/physics.json').read_text())
                control=PurePursuitPID();warm=[];launched=False
                for i in range(600):
                    r=sample_vehicle(w,ego);st=state_of(w.get_snapshot(),ego);tr,_=local_trajectory(route,st,max_speed=1.5);u=control.command(tr,st)
                    if r['long_speed']>1.2:launched=True
                    if launched and 0<=r['long_speed']<=threshold:break
                    command=dict(throttle=0 if launched else .15,brake=0,steer=u['steer']);r['command']=command;warm.append(r);ego.apply_control(carla.VehicleControl(**command));w.tick()
                if not launched or not 0<=r['long_speed']<=threshold:raise RuntimeError('coast crossing not reached')
                name=f'coast_to_{threshold}_r{repeat}';jsonl(root/(name+'_precondition.jsonl'),warm);rows=[]
                for i in range(41):
                    row=sample_vehicle(w,ego);row['command']=dict(throttle=0.,brake=.001,steer=u['steer']);rows.append(row);ego.apply_control(carla.VehicleControl(**row['command']));w.tick()
                log=root/(name+'.jsonl');jsonl(log,rows)
                metrics=dynamics_metrics(rows);first_a=(rows[1]['long_speed']-rows[0]['long_speed'])/.05;boundary_jerk=abs(first_a-rows[0]['long_acceleration_api'])/.05
                result=dict(name=name,initial_speed=rows[0]['long_speed'],metrics=metrics,control_boundary_jerk=boundary_jerk,final_speed=rows[-1]['long_speed'],log=str(log),sha256=file_hash(log));results.append(result);print(name,result,flush=True)
    atomic_json(output,dict(kind='actual_carla_coast_crossing_probes',records=results,sources=sources,physics_unchanged=True,scope='Local terminal braking feasibility only. Precondition includes launch/coast and is retained; cannot claim complete specified-distance comfortable stop.'))
if __name__=='__main__':main()
