"""Actual wall-paced CARLA control, one tick owner, independent planner process."""
import argparse,json,time,math
from dataclasses import asdict
from pathlib import Path
import numpy as np,yaml
import carla
from simulator.session import CarlaSession
from simulator.m5a_async import AsyncPlanner
from simulator.m48_motion import RouteController
from simulator.m47_motion import continuous_rectangles
from simulator.oracle_driver import footprint
from simulator.motion_oracle import route_projection
from tools.run_oracle_closedloop import state_of,obs_of
from tools.m47_common import sample_vehicle,physics_dict,jsonl,freeze_sources,dynamics_metrics
from tools.m47_stopping import traffic_lights
from simulator.recorder import atomic_json,file_hash


def stats(x):
    return dict(p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(np.max(x)),n=len(x)) if len(x) else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m5a_async_runtime.json');p.add_argument('--artifacts',default='reports/m5a_async_artifacts');p.add_argument('--scenarios',nargs='*',default=['straight','mild_curve','junction_turn','slow_worker','worker_crash','emergency','static_obstacle']);a=p.parse_args()
    root=Path(a.artifacts)
    if root.exists() or Path(a.output).exists():raise FileExistsError('new paths required')
    root.mkdir();sources=freeze_sources(root/'source',[__file__,'simulator/m5a_async.py','simulator/m48_schedule.py','simulator/m48_motion.py','simulator/m47_motion.py','simulator/motion_oracle.py'])
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town03');results=[]
    for scene in a.scenarios:
        route_name=scene if scene in ['straight','mild_curve','junction_turn'] else 'straight'
        saved=json.loads(Path('reports/m45_oracle_driving_artifacts',route_name+'_route.json').read_text());route=np.array(saved['route_world_hs']);route=route[np.r_[True,np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)>1e-5]]
        with CarlaSession(cfg,load_map=True) as session:
            w=session.world;cmap=w.get_map();w.set_weather(carla.WeatherParameters.ClearNoon)
            with traffic_lights(w,True):
                ego=session.spawn(w.get_blueprint_library().find('vehicle.tesla.model3'),cmap.get_spawn_points()[saved['spawn_index']]);ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
                for _ in range(30):w.tick()
                atomic_json(root/(scene+'_physics.json'),physics_dict(ego))
                obstacles=[]
                if scene=='static_obstacle':
                    at=route[np.argmin(abs(np.linalg.norm(route[:,:2]-route[0,:2],axis=1)-20))];wp=cmap.get_waypoint(carla.Location(x=float(at[0]),y=float(-at[1]),z=ego.get_location().z));pose=wp.transform;pose.location.z+=.8
                    target=session.spawn(w.get_blueprint_library().find('vehicle.audi.a2'),pose);target.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
                    for _ in range(30):w.tick()
                    target.set_simulate_physics(False);obstacles=[target]
                collisions=[];lanes=[]
                sensor=session.spawn(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego);sensor.listen(lambda e:collisions.append(e.frame))
                sensor=session.spawn(w.get_blueprint_library().find('sensor.other.lane_invasion'),carla.Transform(),parent=ego);sensor.listen(lambda e:lanes.append(e.frame))
                worker=AsyncPlanner(cmap.to_opendrive(),route,delay=.5 if scene=='slow_worker' else 0,max_age=.35)
                rows=[];controller=RouteController();deadline=time.perf_counter();wall_start=deadline;last_submit=-1;start_sim=w.get_snapshot().timestamp.elapsed_seconds;stable=0;faulted=False
                try:
                    for tick in range(1000 if scene in ['straight','mild_curve','junction_turn'] else (600 if scene=='static_obstacle' else 160)):
                        begin=time.perf_counter();snap=w.get_snapshot();state=state_of(snap,ego);measurement=sample_vehicle(w,ego);z=measurement['position'][2]
                        if state.timestamp_s-last_submit>=.2-1e-6:
                            job=dict(scene=scene,frame=snap.frame,state=asdict(state),input_wall=time.perf_counter(),obstacles=[obs_of(snap,o) for o in obstacles],half_length=ego.bounding_box.extent.x,half_width=ego.bounding_box.extent.y,z=z)
                            if worker.submit(job):last_submit=state.timestamp_s
                        if scene=='worker_crash' and tick==65:worker.worker.terminate();worker.worker.join(timeout=.1);faulted=True
                        tr,reason,result=worker.poll(scene,snap.frame,state,time.perf_counter())
                        if scene=='emergency' and tick>=65:reason='emergency_command';faulted=True
                        check_start=time.perf_counter();boundary=False
                        for x,y in footprint(state.x,state.y,state.yaw,ego.bounding_box.extent.x+.2,ego.bounding_box.extent.y+.15):
                            if cmap.get_waypoint(carla.Location(x=float(x),y=float(-y),z=z),project_to_road=False,lane_type=carla.LaneType.Driving) is None:boundary=True;reason=reason or 'current_road_boundary'
                        # Short independent emergency guard covers obstacle motion since plan snapshot.
                        for obj in [obs_of(snap,o) for o in obstacles]:
                            if continuous_rectangles([state.x,state.y],obj['xy'],np.array([math.cos(state.yaw),math.sin(state.yaw)])*state.speed_mps,obj.get('velocity',[0,0]),[ego.bounding_box.extent.x+.2,ego.bounding_box.extent.y+.15],[obj['half_length'],obj['half_width']],state.yaw,obj['yaw'],.35):reason='current_collision_guard'

                        remaining=1000 if result is None else max(0,result['meta'].get('remaining_m',1000)-state.speed_mps*(state.timestamp_s-result['state']['timestamp_s']))
                        command=controller.command(tr,state,reason,remaining_m=remaining)
                        apply_wall=time.perf_counter();ego.apply_control(carla.VehicleControl(throttle=command['throttle'],brake=command['brake'],steer=command['steer']))
                        compute_ms=(time.perf_counter()-begin)*1000
                        w.tick();finished=time.perf_counter();deadline+=.05;miss=finished>deadline
                        rows.append(dict(frame=snap.frame,sim_time=state.timestamp_s,wall_begin=begin,control_apply_wall=apply_wall,tick_end_wall=finished,deadline_wall=deadline,deadline_missed=miss,compute_ms=compute_ms,cycle_work_ms=(finished-begin)*1000,measurement=measurement,command=command,reason=reason,boundary=boundary,plan_frame=None if result is None else result['frame'],plan_input_wall=None if result is None else result['input_wall'],plan_output_wall=None if result is None else result['output_wall'],plan_sim_age=None if result is None else state.timestamp_s-result['state']['timestamp_s'],plan_wall_age=None if result is None else apply_wall-result['input_wall']))
                        if finished<deadline:time.sleep(deadline-finished)
                        elif finished-deadline>.05:deadline=finished  # record miss, do not burst ticks to hide it
                        arc,arc_all,_,_=route_projection(route,[state.x,state.y]);completion=arc/arc_all[-1]
                        stable=stable+1 if state.speed_mps<.08 and completion>.95 else 0
                        if stable>=20:break
                    wall_elapsed=time.perf_counter()-wall_start;sim_elapsed=w.get_snapshot().timestamp.elapsed_seconds-start_sim
                finally:
                    worker.close()
                log=root/(scene+'.jsonl');jsonl(log,rows);plans=root/(scene+'_plans.json');atomic_json(plans,worker.history)
                intervals=np.diff([r['control_apply_wall'] for r in rows]);lat=[(r['output_wall']-r['start_wall'])*1000 for r in worker.history]
                reasons={k:sum(r['reason']==k for r in rows) for k in set(r['reason'] for r in rows) if k}
                result=dict(scene=scene,route_completion=completion,route_passed=stable>=20,collisions=len(collisions),lane_invasions=len(lanes),road_boundary_frames=sum(r['boundary'] for r in rows),ticks=len(rows),sim_hz=len(rows)/sim_elapsed,wall_hz=len(rows)/wall_elapsed,planner_ms=stats(lat),controller_compute_ms=stats([r['compute_ms'] for r in rows]),control_interval_ms=stats(intervals*1000),deadline_miss_rate=float(np.mean([r['deadline_missed'] for r in rows])),interval_over_55ms_rate=float(np.mean(intervals>.055)),plan_sim_age_s=stats([r['plan_sim_age'] for r in rows if r['plan_sim_age'] is not None]),plan_wall_age_s=stats([r['plan_wall_age'] for r in rows if r['plan_wall_age'] is not None]),degradation_ticks=reasons,rejected=worker.rejected,worker_history=str(plans),log=str(log),log_sha256=file_hash(log),actual_dynamics=dynamics_metrics([r['measurement'] for r in rows]))
                results.append(result);atomic_json(root/(scene+'_result.json'),result);print(scene,result,flush=True)
    atomic_json(a.output,dict(kind='actual_CARLA_wall_paced_async_process',results=results,sources=sources,worker_world_tick_calls=0,full_safety_horizon_s=6.,control_period_s=.05,planner_period_s=.2,max_plan_age_s=.35,vision_inputs=False,limitations=['Privileged oracle route/state, not vision closed loop.','Road checks use corners, not exact swept road mesh.','No unsafe fast road certificates.','Route trials alone do not validate dynamic collision generalization.']))
if __name__=='__main__':main()
