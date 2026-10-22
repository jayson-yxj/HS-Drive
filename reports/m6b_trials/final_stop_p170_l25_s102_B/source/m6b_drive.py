"""Actual opt-in Oracle maneuver trial; one tick owner, frozen Safety, fixed wall slots."""
import argparse,json,math,time,os
from dataclasses import asdict
from pathlib import Path
import numpy as np,yaml,carla
from simulator.session import CarlaSession
from simulator.m6b_motion import StopGoal,goal_route,remaining_distance,LowSpeedController
from simulator.m6b_worker import ManeuverPlanner
from simulator.m6b_bypass import choose_bypass,body_samples
from simulator.motion_oracle import route_projection
from simulator.oracle_driver import footprint,wrap
from simulator.m48_motion import RouteController
from simulator.m5a_async import validate_result
from simulator.m47_motion import continuous_rectangles
from simulator.m5b_runtime import snapshot_state_measurement
from simulator.m6b_obstacles import snapshot_obstacle as obs_of
from tools.m47_stopping import traffic_lights
from tools.m47_common import physics_dict,freeze_sources,jsonl,dynamics_metrics
from tools.m6b_common import write,provenance,asset
from diagnostics.m6a_oracle_metrics import quantiles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--arm',choices=['A','B','C','tick_only'],default='C')
    p.add_argument('--scene',choices=['stop','straight','mild_curve','junction_turn','bypass','blocked','removed','lead','emergency','crash','timeout','invalid'],default='stop')
    p.add_argument('--spawn',type=int,default=27);p.add_argument('--length',type=float,default=9)
    p.add_argument('--speed',type=float,default=1);p.add_argument('--seed',type=int,default=101)
    p.add_argument('--ticks',type=int,default=1200);p.add_argument('--fixed-budget',action='store_true')
    p.add_argument('--tick-cpus',default='');p.add_argument('--worker-cpus',default='')
    a=p.parse_args();np.random.seed(a.seed)
    if a.tick_cpus:os.sched_setaffinity(0,set(map(int,a.tick_cpus.split(','))))
    root=Path('reports/m6b_trials')/a.name;root.mkdir(parents=True,exist_ok=False)
    write(root/'config.json',vars(a))
    freeze_sources(root/'source',[__file__,'simulator/m6b_motion.py','simulator/m6b_worker.py','simulator/m6b_bypass.py','simulator/m6b_obstacles.py','simulator/motion_oracle.py','simulator/m48_motion.py','simulator/m47_motion.py','simulator/m5a_async.py','configs/m6b/protocol.json'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03',seed=a.seed)
    with CarlaSession(cfg,load_map=True) as session:
        w=session.world;m=w.get_map();w.set_weather(carla.WeatherParameters.ClearNoon)
        with traffic_lights(w,True):
            spawn=m.get_spawn_points()[a.spawn]
            if a.scene in ['straight','mild_curve','junction_turn']:
                saved=json.loads(Path(f'reports/m45_oracle_driving_artifacts/{a.scene}_route.json').read_text());spawn=m.get_spawn_points()[saved['spawn_index']]
                route=np.asarray(saved['route_world_hs']);route=route[np.r_[True,np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)>1e-5]]
            else:
                wp=m.get_waypoint(spawn.location);route=[]
                for _ in range(int(a.length)+1):
                    t=wp.transform;route.append([t.location.x,-t.location.y,-math.radians(t.rotation.yaw)])
                    ns=wp.next(1.)
                    if not ns:break
                    wp=min(ns,key=lambda q:abs(wrap(math.radians(q.transform.rotation.yaw-t.rotation.yaw))))
                route=np.array(route)
            ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),spawn)
            ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(30):w.tick()
            physics=physics_dict(ego);write(root/'physics.json',physics)
            hl,hw=ego.bounding_box.extent.x,ego.bounding_box.extent.y
            z=ego.get_location().z
            goal=StopGoal(tuple(route[-1,:2]),float(route[-1,2]),timestamp=w.get_snapshot().timestamp.elapsed_seconds);goal.validate()
            def road(x,y):return m.get_waypoint(carla.Location(x=float(x),y=float(-y),z=z),project_to_road=False,lane_type=carla.LaneType.Driving) is not None
            if not all(road(x,y) for x,y in body_samples(*goal.target_position_xy,goal.target_yaw,hl+.2,hw+.15)):
                raise ValueError('stop pose body outside Driving area')
            write(root/'goal.json',dict(**asdict(goal),reference='ego_actor_origin',navigation_endpoint=route[-1].tolist(),body_road_audit=True))
            obstacles=[]
            if a.scene in ['bypass','blocked','removed','lead']:
                at=route[min(len(route)-1,40 if a.scene in ['bypass','blocked','removed'] else 20)]
                pose=m.get_waypoint(carla.Location(x=float(at[0]),y=float(-at[1]),z=z)).transform;pose.location.z+=.8
                obj=session.spawn(w.get_blueprint_library().find('vehicle.audi.a2'),pose)
                obj.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
                for _ in range(30):w.tick()
                if a.scene=='lead':
                    obj.apply_control(carla.VehicleControl());obj.enable_constant_velocity(carla.Vector3D(1.,0,0))
                else:obj.set_simulate_physics(False)
                obstacles=[obj]
                if a.scene=='blocked':
                    # Same physical obstacle in both same-direction lanes:
                    # refusal is caused by geometry, not an artificial flag.
                    here=m.get_waypoint(pose.location);neighbor=here.get_left_lane()
                    if neighbor is not None and neighbor.lane_type==carla.LaneType.Driving and neighbor.lane_id*here.lane_id>0:
                        pose2=neighbor.transform;pose2.location.z+=.8
                        other=session.spawn(w.get_blueprint_library().find('vehicle.audi.a2'),pose2)
                        other.set_simulate_physics(False);obstacles.append(other)
            selected=route.copy();bypass=dict(source='Privileged',status='NotRequested')
            if a.scene in ['bypass','blocked']:
                selected,bypass=choose_bypass(m,route,[obs_of(w.get_snapshot(),o) for o in obstacles],hl,hw,z)
            write(root/'bypass.json',bypass);write(root/'route.json',dict(source='Privileged',original=route.tolist(),selected=selected.tolist()))
            planning=selected if a.arm=='A' else goal_route(selected,goal)
            ctl=RouteController() if a.arm in ('A','B') else LowSpeedController()
            worker=None if a.arm=='tick_only' else ManeuverPlanner(m.to_opendrive(),planning,max_speed=a.speed,worker_cpus=tuple(map(int,a.worker_cpus.split(','))) if a.worker_cpus else ())
            collisions=[];lanes=[]
            sensor=session.spawn(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego)
            sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id,impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z],source='Privileged')))
            sensor=session.spawn(w.get_blueprint_library().find('sensor.other.lane_invasion'),carla.Transform(),parent=ego)
            sensor.listen(lambda e:lanes.append(dict(frame=e.frame,markings=[str(v.type) for v in e.crossed_lane_markings],source='Privileged')))
            rows=[];last_submit=-1.;epoch=time.perf_counter();slot=0;stable=0;fault_wall=None;first_brake=None;removed=False;stationary=0
            try:
                for tick in range(a.ticks):
                    scheduled=epoch+slot*.05;sleep_start=time.perf_counter()
                    if sleep_start<scheduled:time.sleep(scheduled-sleep_start)
                    begin=time.perf_counter();timeline=dict(scheduled=scheduled,wake_up=begin,previous_sleep_start=sleep_start)
                    snap=w.get_snapshot();applied=ego.get_control();state,measurement=snapshot_state_measurement(snap,ego.id,applied)
                    timeline['snapshot_end']=time.perf_counter()
                    if a.scene=='lead':
                        if tick==200:obstacles[0].disable_constant_velocity();obstacles[0].apply_control(carla.VehicleControl(brake=.3))
                        if tick==500:obstacles[0].apply_control(carla.VehicleControl());obstacles[0].enable_constant_velocity(carla.Vector3D(1.5,0,0))
                    if a.scene=='removed' and tick==400:
                        obstacles[0].destroy();session.actors.remove(obstacles[0]);obstacles=[];removed=True
                    obs=[obs_of(snap,o) for o in obstacles]
                    tr=reason=result=None
                    if worker:
                        if tick==65 and a.scene in ('emergency','crash','timeout','invalid'):fault_wall=begin
                        if state.timestamp_s-last_submit>=.2-1e-6:
                            job=dict(scene=a.name,frame=snap.frame,state=asdict(state),input_wall=time.perf_counter(),obstacles=obs,half_length=hl,half_width=hw,z=z)
                            if tick>=65 and a.scene=='timeout':job['delay']=.5
                            if tick>=65 and a.scene=='invalid':job['state'].pop('x')
                            if worker.submit(job):last_submit=state.timestamp_s
                        if tick==65 and a.scene=='crash':worker.worker.terminate()
                        tr,reason,result=worker.poll(a.name,snap.frame,state,time.perf_counter())
                        if tick>=65 and a.scene=='emergency':reason='emergency_command'
                    timeline['submit_poll_end']=time.perf_counter()
                    current_body=footprint(state.x,state.y,state.yaw,hl+.2,hw+.15)
                    boundary=not all(road(x,y) for x,y in current_body)
                    if bypass.get('status')=='FeasibleBypass':
                        chosen=next(v for v in bypass['candidates'] if v.get('side')==bypass['selected'])
                        legal_lanes={tuple(v) for v in chosen['allowed_lanes']}
                        for xx,yy in current_body:
                            wp=m.get_waypoint(carla.Location(x=float(xx),y=float(-yy),z=z),project_to_road=False,lane_type=carla.LaneType.Driving)
                            if wp is None or (wp.road_id,wp.lane_id) not in legal_lanes:boundary=True
                    if boundary:reason=reason or 'current_road_boundary'
                    for o in obs:
                        if continuous_rectangles([state.x,state.y],o['xy'],np.array([np.cos(state.yaw),np.sin(state.yaw)])*state.speed_mps,o['velocity'],[hl+.2,hw+.15],[o['half_length'],o['half_width']],state.yaw,o['yaw'],.35):reason='current_collision_guard'
                    timeline['safety_end']=time.perf_counter()
                    if result is not None and reason is None:reason=validate_result(result,a.name,snap.frame,state.timestamp_s,time.perf_counter(),.35)
                    remaining=remaining_distance(selected,state,obs,hl,hw)
                    control_remaining=remaining
                    if a.arm=='A':control_remaining=1000 if result is None else max(0,result['meta'].get('remaining_m',1000)-state.speed_mps*(state.timestamp_s-result['state']['timestamp_s']))
                    u=dict(throttle=0.,brake=1.,steer=0.,reason='tick_only') if a.arm=='tick_only' else ctl.command(tr,state,reason,remaining_m=control_remaining)
                    timeline['controller_end']=time.perf_counter();timeline['apply_control']=time.perf_counter()
                    ego.apply_control(carla.VehicleControl(**{k:u[k] for k in ('throttle','brake','steer')}));timeline['apply_end']=time.perf_counter()
                    if fault_wall is not None and first_brake is None and reason and u['brake']>0:first_brake=timeline['apply_control']
                    timeline['tick_start']=time.perf_counter();next_frame=w.tick();timeline['tick_end']=time.perf_counter()
                    miss=timeline['tick_end']>scheduled+.05
                    # Never reset epoch or burst catch-up ticks: retain missed slots.
                    next_slot=max(slot+1,int(math.floor((timeline['tick_end']-epoch)/.05))+1)
                    skipped=next_slot-slot-1
                    err=float(np.linalg.norm(np.array([state.x,state.y])-goal.target_position_xy));heading=abs(float(wrap(state.yaw-goal.target_yaw)))
                    stable=stable+1 if err<=.5 and heading<=goal.heading_tolerance_rad and state.speed_mps<=.08 and not boundary else 0
                    stationary=stationary+1 if state.speed_mps<=.02 else 0
                    arc,total,_,cte=route_projection(route,[state.x,state.y])
                    _,_,_,path_cte=route_projection(selected,[state.x,state.y])
                    rows.append(dict(source='Privileged',frame=snap.frame,next_frame=next_frame,sim_time=state.timestamp_s,measurement=measurement,obstacles=obs,
                        oracle_coordinate='hs_world_x_y_left_z_up_m',measurement_coordinate='carla_world_x_y_right_z_up_m',timeline=timeline,
                        deadline_missed=miss,skipped_wall_slots=skipped,command=u,reason=reason,boundary=boundary,
                        current_remaining_m=remaining,control_remaining_m=control_remaining,goal_error_m=err,heading_error_rad=heading,stable_ticks=stable,
                        completion=float(arc/total[-1]),cross_track_m=cte,path_cross_track_m=path_cte,
                        plan_sim_age=None if result is None else state.timestamp_s-result['state']['timestamp_s'],
                        plan_wall_age=None if result is None else timeline['apply_control']-result['input_wall'],
                        plan_speed0=None if tr is None else float(tr.points[0,3]),plan_end_speed=None if tr is None else float(tr.points[-1,3])))
                    slot=next_slot
                    if stable>=21 and not a.fixed_budget:break
                    # A failed endpoint plateau is an explicit failure stop rule,
                    # not successful goal completion. At least 5s continuously
                    # stopped after motion; never truncate recovery/fault trials.
                    if not a.fixed_budget and a.scene=='stop' and tick>200 and stationary>=100:break
                    if not a.fixed_budget and a.scene=='blocked' and tick>200 and stationary>=100:break
                    if collisions:break
            finally:
                if worker:worker.close()
            jsonl(root/'trace.jsonl',rows);write(root/'plans.json',[] if worker is None else [dict(source='Privileged',**q) for q in worker.history])
            write(root/'transitions.json',getattr(ctl,'transitions',[]))
            intervals=np.diff([r['timeline']['apply_control'] for r in rows])*1000
            terminal=rows[-1];raw=dynamics_metrics([r['measurement'] for r in rows]);body_final=all(road(x,y) for x,y in body_samples(state.x,state.y,state.yaw,hl+.2,hw+.15))
            forbidden_lanes=[e for e in lanes if any('Solid' in v for v in e['markings'])]
            status=dict(position=terminal['goal_error_m']<=.5,heading=terminal['heading_error_rad']<=goal.heading_tolerance_rad,
                speed=state.speed_mps<=.08,hold_1s=stable>=21,no_collision=not collisions,no_road_boundary=not any(r['boundary'] for r in rows),body_road=body_final,
                no_illegal_lane_crossing=not forbidden_lanes,route_completion=terminal['completion']>=.95)
            summary=dict(**provenance('actual_CARLA_closed_loop' if worker else 'actual_CARLA_tick_only'),source='Privileged',config=vars(a),ticks=len(rows),
                goal_checks=status,goal_passed=all(status.values()),final_goal_error_m=terminal['goal_error_m'],heading_error_rad=terminal['heading_error_rad'],
                terminal_speed_mps=state.speed_mps,completion=terminal['completion'],collisions=collisions,lane_events=lanes,illegal_lane_events=forbidden_lanes,
                cte_m=quantiles([r['cross_track_m'] for r in rows]),selected_path_cte_m=quantiles([r['path_cross_track_m'] for r in rows]),
                control_interval_ms=quantiles(intervals),over50=int((intervals>50).sum()),over55=int((intervals>55).sum()),over100=int((intervals>100).sum()),
                deadline_miss=sum(r['deadline_missed'] for r in rows),skipped_wall_slots=sum(r['skipped_wall_slots'] for r in rows),
                mean_hz=(len(rows)-1)/(rows[-1]['timeline']['apply_control']-rows[0]['timeline']['apply_control']),
                world_tick_ms=quantiles([(r['timeline']['tick_end']-r['timeline']['tick_start'])*1000 for r in rows]),
                controller_ms=quantiles([(r['timeline']['controller_end']-r['timeline']['safety_end'])*1000 for r in rows]),
                expired_normal_control=sum(r['reason'] is None and a.arm!='tick_only' and (r['plan_sim_age'] is None or r['plan_sim_age']>.35+1e-6 or r['plan_wall_age']>.35+1e-6) for r in rows),
                fault_to_brake_s=None if first_brake is None else first_brake-fault_wall,
                degradation={v:sum(r['reason']==v for r in rows) for v in sorted({r['reason'] for r in rows if r['reason']})},
                planner_rejected=[] if worker is None else worker.rejected,dynamics=raw,
                comfort_pass=raw['max_decel_raw']<=1.8 and raw['jerk_p95_moving']<=2.5 and raw['jerk_peak_raw']<=5,
                bypass=bypass,trace=asset(root/'trace.jsonl'),physics=asset(root/'physics.json'),
                tick_affinity=sorted(os.sched_getaffinity(0)),
                limits='Privileged single-map controlled trial. Broken-line crossings permitted only audited bypass. Dense footprint checks are not exact curb mesh proof. No reverse parking.')
            write(root/'result.json',summary);print(a.name,'goal',summary['goal_passed'],'error',round(terminal['goal_error_m'],3),'speed',round(state.speed_mps,3),'decel',round(raw['max_decel_raw'],2),'jerk',round(raw['jerk_peak_raw'],2),'reasons',summary['degradation'],flush=True)

if __name__=='__main__':main()
