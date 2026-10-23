"""Dedicated real CARLA finite actuator experiments with immutable versions."""
import argparse,json,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np,yaml,carla
from simulator.session import CarlaSession
from simulator.m5b_runtime import snapshot_state_measurement
from simulator.m6c_geometry import road_grade,body_box,polygon_gap
from simulator.m6b_bypass import body_samples
from simulator.oracle_driver import footprint
from simulator.m6d_state import signed_from_measurement
from simulator.m6e_guard import DisplacementGuard,GuardConfig
from simulator.m6e_actuation import cycle_command,apply_command,serialize_physics,apply_variant
from tools.m47_common import physics_dict,freeze_sources,jsonl,dynamics_metrics
from tools.m6e_common import write,asset,provenance
from diagnostics.m6a_oracle_metrics import quantiles


def run(task,protocol):
    root=Path('reports/m6e_trials')/task['name'];root.mkdir(parents=True,exist_ok=False);write(root/'config.json',task)
    files=[__file__,'simulator/m6e_guard.py','simulator/m6e_actuation.py','simulator/m6d_state.py','simulator/m5b_runtime.py','simulator/session.py','simulator/m6c_geometry.py','tools/m47_common.py','configs/m6e/protocol.json']
    if task['physics']=='P1':files+=['configs/m6e/physics_P1.json']
    freeze_sources(root/'source',files)
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03',seed=task['seed'])
    with CarlaSession(cfg,load_map=True) as session:
        w=session.world;m=w.get_map();w.set_weather(carla.WeatherParameters.ClearNoon);spawn=m.get_spawn_points()[task['spawn']]
        ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),spawn);w.tick()
        write(root/'physics_original.json',physics_dict(ego));assert asset(root/'physics_original.json')['sha256']==protocol['physics_P0_sha']
        if task['physics']=='P1':
            original_control=ego.get_physics_control()
            session.cleanup_callbacks.append(lambda:ego.apply_physics_control(original_control))
            apply_variant(ego,json.loads(Path('configs/m6e/physics_P1.json').read_text()))
            before=json.loads((root/'physics_original.json').read_text());after=physics_dict(ego)
            expected=json.loads(json.dumps(before))
            for wheel in expected['wheels']:wheel['max_brake_torque']*=.5
            if after!=expected:raise ValueError('Physics readback differs from sole declared wheel brake change')
        write(root/'physics.json',physics_dict(ego));write(root/'physics_full.json',serialize_physics(ego))
        if task['physics']=='P0':assert asset(root/'physics.json')['sha256']==protocol['physics_P0_sha']
        settings=ego.get_ackermann_controller_settings();write(root/'ackermann_settings.json',{k:getattr(settings,k) for k in ['speed_kp','speed_ki','speed_kd','accel_kp','accel_ki','accel_kd']})
        bb=ego.bounding_box;ref=dict(location=[bb.location.x,bb.location.y,bb.location.z],extent=[bb.extent.x,bb.extent.y,bb.extent.z],rotation=[bb.rotation.roll,bb.rotation.pitch,bb.rotation.yaw]);write(root/'body.json',ref)
        angle=-math.radians(spawn.rotation.yaw);axis=np.array([math.cos(angle),math.sin(angle)]);origin=np.array([spawn.location.x,-spawn.location.y]);route=np.array([np.r_[origin,angle],np.r_[origin+100*axis,angle]])
        guard=DisplacementGuard(GuardConfig(**protocol['guard']));collisions=[];lanes=[]
        sensor=session.spawn(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego);sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id)))
        sensor=session.spawn(w.get_blueprint_library().find('sensor.other.lane_invasion'),carla.Transform(),parent=ego);sensor.listen(lambda e:lanes.append(dict(frame=e.frame,markings=[str(v.type) for v in e.crossed_lane_markings])))
        mode=task.get('mode','cycle');phase='initialize';phase_tick=0;rows=[];epoch=time.perf_counter();slot=0;initial=None;normal_stable=0;held_anchor=None;emergency_started=None;stop_anchor=None;obstacle=None;boundary_count=0
        for tick in range(1300):
            scheduled=epoch+slot*.05;sleep_start=time.perf_counter()
            if sleep_start<scheduled:time.sleep(scheduled-sleep_start)
            begin=time.perf_counter();snap=w.get_snapshot();state,r=snapshot_state_measurement(snap,ego.id,ego.get_control());pos=np.array(r['position'][:2])*[1,-1]
            if initial is None:initial=r;held_anchor=pos.copy()
            grade=road_grade(m,r['position']);signed=signed_from_measurement(r,route,float((origin-pos)@axis),grade['grade']);snapshot_end=time.perf_counter()
            transition=None;elapsed=phase_tick*.05
            if phase=='initialize' and elapsed>=1.5:transition='prepare' if mode in ['guard','emergency','coast','obstacle'] else 'hold_initial'
            elif phase=='hold_initial' and elapsed>=2:transition='cold_run'
            elif phase=='cold_run' and elapsed>=6:transition='normal_stop'
            elif phase=='normal_stop' and elapsed>=6:transition='hold_warm'
            elif phase=='hold_warm' and elapsed>=2:transition='warm_run'
            elif phase=='warm_run' and elapsed>=6:transition='final_stop'
            elif phase=='final_stop' and elapsed>=6:transition='hold_final'
            elif phase=='hold_final' and elapsed>=2:transition='done'
            elif phase=='prepare' and (signed.v_forward_mps>= (1. if mode=='guard' else 2.5) or elapsed>=10):transition='warm_stop' if mode=='guard' else 'diagnostic_action'
            elif phase=='warm_stop':
                normal_stable=normal_stable+1 if signed.speed_norm_mps<=.02 else 0
                if normal_stable>=5 or elapsed>=6:transition='guard_hold'
            elif phase=='guard_hold' and elapsed>=3:transition='guard_release'
            elif phase=='guard_release' and elapsed>=6:transition='done'
            elif phase=='protected_stop' and elapsed>=2:transition='done'
            elif phase=='diagnostic_action':
                normal_stable=normal_stable+1 if signed.speed_norm_mps<=.02 else 0
                if elapsed>=10 or normal_stable>=25:transition='done'
            if transition:
                if transition=='done':break
                phase=transition;phase_tick=0;elapsed=0.;normal_stable=0
                if phase in ['normal_stop','final_stop','warm_stop','diagnostic_action']:stop_anchor=pos.copy()
                if phase.startswith('hold') or phase=='guard_hold':held_anchor=pos.copy()
                if phase=='diagnostic_action' and mode=='obstacle':
                    tf=ego.get_transform();f=tf.get_forward_vector();tf.location.x+=f.x*(bb.extent.x+1.86+5.);tf.location.y+=f.y*(bb.extent.x+1.86+5.);tf.location.z+=.8
                    obstacle=session.spawn(w.get_blueprint_library().find('vehicle.audi.a2'),tf);obstacle.set_simulate_physics(False)
            kind,u,target=cycle_command(task['interface'],phase,elapsed)
            if mode!='cycle' and phase!='initialize':
                kind='raw';u=dict(throttle=0.,brake=0.,steer=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0);target=None
                if phase=='prepare':u['throttle']=.2
                elif phase=='warm_stop':u['brake']=.01
                elif phase=='guard_hold':u['brake']=.12
                elif phase=='guard_release':u['throttle']=.15
                elif phase=='diagnostic_action' and mode in ['emergency','obstacle']:u['brake']=1.
            gd=guard.update(signed,pos,axis,signed.timestamp_s)
            # Cycle comparison observes guard only: no hidden actuator changes.
            # Existing emergency override is tested separately in guard mode.
            protection=mode=='guard' and phase in ['guard_release','protected_stop'] and gd['emergency']
            if protection:
                if phase!='protected_stop':phase='protected_stop';phase_tick=0;emergency_started=state.timestamp_s
                kind='raw';u=dict(throttle=0.,brake=1.,steer=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
            gap=None
            if obstacle is not None and snap.find(obstacle.id):
                from simulator.m6b_obstacles import snapshot_obstacle
                box=body_box(r,ref);o=snapshot_obstacle(snap,obstacle);gap=polygon_gap(footprint(*box['xy'],box['yaw'],box['half_length'],box['half_width']),footprint(*o['xy'],o['yaw'],o['half_length'],o['half_width']))
            # Sparse online corners; final dense full body below, never exact mesh proof.
            road_ok=all(m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=r['position'][2]),project_to_road=False,lane_type=carla.LaneType.Driving) is not None for x,y in footprint(state.x,state.y,state.yaw,bb.extent.x+.2,bb.extent.y+.15))
            boundary_count+=not road_ok
            safety_end=time.perf_counter();controller_end=time.perf_counter();apply=time.perf_counter()
            # Raw-to-Ackermann release explicitly clears lingering handbrake;
            # this is a single initialization boundary, not PID tuning.
            if kind=='ackermann' and (not rows or rows[-1]['kind']!='ackermann'):ego.apply_control(carla.VehicleControl())
            apply_command(ego,kind,u);apply_end=time.perf_counter();tick_start=time.perf_counter();next_frame=w.tick();tick_end=time.perf_counter()
            next_slot=max(slot+1,int(math.floor((tick_end-epoch)/.05))+1)
            rows.append(dict(source='Privileged',physics_version=task['physics'],frame=snap.frame,next_frame=next_frame,measurement=r,signed=asdict(signed),phase=phase,phase_tick=phase_tick,road=grade,kind=kind,command=u,reference_speed=target,guard=gd,protection_applied=protection,road_ok=road_ok,obstacle_obb_gap_m=gap,
                stop_travel_m=None if stop_anchor is None else float((pos-stop_anchor)@axis),hold_displacement_m=float(np.linalg.norm(pos-held_anchor)),
                planner_age=None,worker_queue_ms=None,plan_source='NotApplicable_fixed_actuator_schedule',
                timeline=dict(scheduled=scheduled,previous_sleep_start=sleep_start,wake_up=begin,snapshot_end=snapshot_end,safety_end=safety_end,controller_end=controller_end,apply_control=apply,apply_end=apply_end,tick_start=tick_start,tick_end=tick_end),deadline_missed=tick_end>scheduled+.05,skipped_slots=next_slot-slot-1))
            slot=next_slot;phase_tick+=1
            if collisions:break
        write(root/'physics_final.json',physics_dict(ego));assert asset(root/'physics_final.json')['sha256']==asset(root/'physics.json')['sha256']
        jsonl(root/'trace.jsonl',rows);write(root/'guard_events.json',guard.events)
        final=rows[-1];interval=np.diff([q['timeline']['apply_control'] for q in rows])*1000
        final_body_ok=all(m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=r['position'][2]),project_to_road=False,lane_type=carla.LaneType.Driving) is not None for x,y in body_samples(state.x,state.y,state.yaw,bb.extent.x+.2,bb.extent.y+.15))
        write(root/'result.json',dict(**provenance('actual_CARLA_paced_actuator_probe'),config=task,initial=initial,trace=asset(root/'trace.jsonl'),physics=asset(root/'physics.json'),physics_full=asset(root/'physics_full.json'),ticks=len(rows),complete=transition=='done',collisions=collisions,lane_events=lanes,online_boundary_ticks=boundary_count,final_body_road=final_body_ok,
            raw=dynamics_metrics([q['measurement'] for q in rows]),control_interval_ms=quantiles(interval),tick_ms=quantiles([(q['timeline']['tick_end']-q['timeline']['tick_start'])*1000 for q in rows]),
            mean_hz=(len(rows)-1)/(rows[-1]['timeline']['apply_control']-rows[0]['timeline']['apply_control']),deadline_miss=sum(q['deadline_missed'] for q in rows),skipped_slots=sum(q['skipped_slots'] for q in rows),
            worker='NotApplicable_no_planner',expired_normal_plan='NotApplicable_no_planner',emergency_started=emergency_started,
            limits='Fixed actuator probes, not navigation Pilot; sparse online corner checks plus final dense body, no exact curb mesh. Target speed profile is desired reference, never physical velocity assignment.'))
    return root/'result.json'


def main():
    p=argparse.ArgumentParser();p.add_argument('batch',choices=['development','validation','guard','variant']);a=p.parse_args();cfg=json.loads(Path('configs/m6e/protocol.json').read_text());tasks=[]
    if a.batch in ['development','validation']:
        for pos in cfg[a.batch+'_positions']:
            for seed in cfg['seeds']:
                for interface in ['A','B','C']:tasks.append(dict(name=f'P0_{a.batch}_p{pos}_s{seed}_{interface}',physics='P0',split=a.batch,spawn=pos,seed=seed,interface=interface,mode='cycle'))
    elif a.batch=='guard':
        for pos in cfg['guard_live']['positions']:
            for seed in cfg['seeds']:tasks.append(dict(name=f'P0_guard_p{pos}_s{seed}',physics='P0',split='development' if pos==27 else 'validation',spawn=pos,seed=seed,interface='A',mode='guard'))
    else:tasks=json.loads(Path('configs/m6e/physics_P1.json').read_text())['tasks']
    write(f'reports/m6e_{a.batch}_plan.json',dict(**provenance('predeclared_finite_probe_batch'),tasks=tasks));results=[]
    for i,task in enumerate(tasks):
        r=run(task,cfg);results.append(asset(r));print(i+1,len(tasks),r,flush=True)
    write(f'reports/m6e_{a.batch}_execution.json',dict(**provenance('actual_CARLA_probe_batch_complete'),results=results))
if __name__=='__main__':main()
