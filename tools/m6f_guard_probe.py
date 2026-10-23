"""Frozen M6E G1 on predeclared independent 0915 positions, no threshold search."""
import json
import math
from pathlib import Path
import time
from dataclasses import asdict
import carla
import numpy as np
import yaml
from simulator.session import CarlaSession
from simulator.m6e_guard import DisplacementGuard
from simulator.m6d_state import signed_from_measurement
from simulator.m6c_geometry import road_grade
from tools.m47_common import physics_dict, sample_vehicle
from tools.m6f_probe import write,sha,resource
from tools.m6f_launch import inventory,verify_loaded


def main():
    protocol=json.loads(Path('configs/m6f/protocol.json').read_text());root=Path('reports/m6f_guard_0915');root.mkdir(exist_ok=False)
    write(root/'launch.json',dict(protocol=protocol,loaded=inventory(),resources=resource()))
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03')
    for pos in protocol['guard_0915_positions']:
        for rep in protocol['repeat_ids']:
            dest=root/f'p{pos}_{rep}';dest.mkdir();rows=[]
            with CarlaSession(cfg,load_map=True) as session:
                w=session.world;w.set_weather(carla.WeatherParameters.ClearNoon);m=w.get_map();spawn=m.get_spawn_points()[pos]
                ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),spawn);w.tick()
                p=physics_dict(ego);p0=json.loads(Path('reports/m6e_trials/P0_guard_p27_s301/physics.json').read_text())
                if p!=p0:raise ValueError('0915 P0 changed')
                write(dest/'physics.json',dict(carla_version='0.9.15',physics=p,sha256=sha(p),true_engine_wheel_api=False))
                yaw=-math.radians(spawn.rotation.yaw);axis=np.array([math.cos(yaw),math.sin(yaw)]);origin=np.array([spawn.location.x,-spawn.location.y]);route=np.array([np.r_[origin,yaw],np.r_[origin+100*axis,yaw]])
                grade=road_grade(m,[spawn.location.x,spawn.location.y,spawn.location.z]);write(dest/'road.json',grade)
                guard=DisplacementGuard();phase='initialize';phase_tick=0;stable=0;epoch=time.perf_counter();slot=0;collisions=[]
                sensor=session.spawn(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego);sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id)))
                with (dest/'trace.jsonl').open('x') as f:
                    for i in range(700):
                        scheduled=epoch+slot*.05
                        if time.perf_counter()<scheduled:time.sleep(scheduled-time.perf_counter())
                        r=sample_vehicle(w,ego);point=np.array(r['position'][:2])*[1,-1];s=signed_from_measurement(r,route,float((origin-point)@axis),grade['grade'])
                        transition=None;t=phase_tick*.05
                        if phase=='initialize' and t>=1.5:transition='prepare'
                        elif phase=='prepare' and (s.v_forward_mps>=1 or t>=10):transition='warm_stop'
                        elif phase=='warm_stop':
                            stable=stable+1 if s.speed_norm_mps<=.02 else 0
                            if stable>=5 or t>=6:transition='guard_hold'
                        elif phase=='guard_hold' and t>=3:transition='guard_release'
                        elif phase=='guard_release' and t>=6:transition='done'
                        elif phase=='protected_stop' and t>=2:transition='done'
                        if transition=='done':break
                        if transition:phase=transition;phase_tick=0;stable=0
                        g=guard.update(s,point,axis,s.timestamp_s)
                        u=dict(throttle=.2 if phase=='prepare' else .15 if phase=='guard_release' else 0.,
                               brake=1. if phase=='initialize' else .01 if phase=='warm_stop' else .12 if phase=='guard_hold' else 0.,
                               hand_brake=phase=='initialize',reverse=False,manual_gear_shift=False,gear=0,steer=0.)
                        protected=phase in ['guard_release','protected_stop'] and g['emergency']
                        if protected:
                            if phase!='protected_stop':phase='protected_stop';phase_tick=0
                            u.update(throttle=0.,brake=1.,hand_brake=False)
                        apply=time.perf_counter();ego.apply_control(carla.VehicleControl(**u));frame=w.tick();end=time.perf_counter()
                        next_slot=max(slot+1,int((end-epoch)/.05)+1)
                        row=dict(source='Privileged',carla_version='0.9.15',physics_sha256=sha(p),true_engine_wheel_api=False,
                                 measurement=r,signed=asdict(s),phase=phase,guard=g,command=u,protection_applied=protected,
                                 apply_time=apply,tick_end=end,next_frame=frame,scheduled=scheduled,deadline_missed=end>scheduled+.05)
                        f.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row);slot=next_slot;phase_tick+=1
                        if collisions:break
                write(dest/'result.json',dict(source='Privileged',carla_version='0.9.15',physics_sha256=sha(p),true_engine_wheel_api=False,
                     source_commit=inventory()['git_commit'],complete=transition=='done',collisions=collisions,guard_events=guard.events,rows=len(rows),
                     protocol='frozen M6E guard, no assigned velocity, new spawn positions; no sensitivity claim without natural positive'))
            verify_loaded();print(pos,rep,len(rows),flush=True)
    write(root/'completion.json',dict(loaded=inventory(),resources=resource()))
