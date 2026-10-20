"""M4.5 privileged planner/controller quantitative CARLA scenarios; no network."""
import argparse,json,time,sys,math
from pathlib import Path
import numpy as np,yaml
from simulator.session import CarlaSession
from simulator.oracle_driver import OracleState,PurePursuitPID,local_trajectory,safety_check,wrap
from simulator.recorder import atomic_json,file_hash

CARLA_AGENTS='/root/gpufree-data/simulators/CARLA_0.9.15/PythonAPI/carla'


def find_routes(cmap,length):
    routes={}
    for spawn_id,spawn in enumerate(cmap.get_spawn_points()):
        start=cmap.get_waypoint(spawn.location)
        for branch in (0,1,2):
            path=[start]
            for step in range(length):
                options=sorted(path[-1].next(1.),key=lambda w:(w.road_id,w.lane_id,w.s))
                if not options:break
                path.append(options[min(branch,len(options)-1)])
            if len(path)<length:continue
            yaw=np.unwrap(np.deg2rad([w.transform.rotation.yaw for w in path]));delta=float(np.max(np.abs(yaw-yaw[0])));junction=any(w.is_junction for w in path)
            if delta<.08 and not junction:key='straight'
            elif .12<delta<.65:key='mild_curve'
            elif .8<delta<2.2 and junction:key='junction_turn'
            else:continue
            if key not in routes:routes[key]=(spawn_id,spawn,path)
        if len(routes)==3:break
    if 'straight' not in routes:raise RuntimeError('No deterministic straight route')
    return routes


def state_of(snapshot,actor):
    a=snapshot.find(actor.id);t=a.get_transform();v=a.get_velocity()
    return OracleState(t.location.x,-t.location.y,-math.radians(t.rotation.yaw),math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z),snapshot.timestamp.elapsed_seconds)


def obs_of(snapshot,actor):
    s=state_of(snapshot,actor);v=snapshot.find(actor.id).get_velocity();bb=actor.bounding_box
    return {'xy':[s.x,s.y],'yaw':s.yaw,'half_length':bb.extent.x,'half_width':bb.extent.y,'velocity':[v.x,-v.y]}


def main():
    import carla
    sys.path.insert(0,CARLA_AGENTS)
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m45_oracle_driving.json');p.add_argument('--artifacts',default='reports/m45_oracle_driving_artifacts');p.add_argument('--scenarios',nargs='*');p.add_argument('--replay-routes',help='Directory containing saved scenario_route.json files');a=p.parse_args()
    root=Path(a.artifacts)
    if Path(a.output).exists() or root.exists():raise FileExistsError('New outputs required')
    root.mkdir();cfg=yaml.safe_load(Path('configs/m45_oracle_driving.yaml').read_text());session_cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());session_cfg.update(port=cfg['port'],map=cfg['map']);client=carla.Client('127.0.0.1',cfg['port']);client.set_timeout(60);world=client.get_world()
    if world.get_map().name.rsplit('/',1)[-1]!=cfg['map']:world=client.load_world(cfg['map'])
    cmap=world.get_map();route_options=find_routes(cmap,cfg['route_length_m']);grp=GlobalRoutePlanner(cmap,1.0)
    report={'schema_version':'hsd-m45-oracle-driving-v1','scope':'CARLA privileged state/map/actors, not vision driving. Low-speed fixed routes; no autopilot. Discrete densified footprint, not exact continuous swept volume.','config':cfg,'scenarios':{},'official_route_planner_sha256':file_hash(Path(CARLA_AGENTS)/'agents/navigation/global_route_planner.py')}
    for scenario in a.scenarios or cfg['scenarios']:
        key=scenario if scenario in ('straight','mild_curve','junction_turn') else 'straight'
        if key not in route_options:report['scenarios'][scenario]={'status':'Partial','reason':'No matching fixed route found'};continue
        spawn_id,spawn,path=route_options[key];trace=grp.trace_route(path[0].transform.location,path[-1].transform.location)
        route=np.array([[wp.transform.location.x,-wp.transform.location.y,-math.radians(wp.transform.rotation.yaw)] for wp,option in trace]);route_carla=np.array([[wp.transform.location.x,wp.transform.location.y,wp.transform.location.z] for wp,op in trace])
        if a.replay_routes:
            saved=json.loads((Path(a.replay_routes)/(scenario+'_route.json')).read_text())
            if saved['map']!=cmap.name or saved['spawn_index']!=spawn_id or not np.allclose(saved['route_world_hs'],route,atol=1e-6):raise ValueError('Recorded route differs from current CARLA map/router')
            route=np.array(saved['route_world_hs']);route_carla=np.array(saved['route_world_carla'])
        if len(route)<10:raise ValueError('Route too short')
        atomic_json(root/(scenario+'_route.json'),{'spawn_index':spawn_id,'route_world_hs':route.tolist(),'route_world_carla':route_carla.tolist(),'map':cmap.name,'source':'official CARLA GlobalRoutePlanner trace_route','options':[str(op) for _,op in trace]})
        collision=[];lane_events=[];rows=[];reason_counts={};latencies=[];wall_start=time.monotonic();success=False;failure='max_ticks';stop_stable=0
        with CarlaSession(session_cfg,load_map=False) as session:
            world=session.world;weather=carla.WeatherParameters.ClearNoon;world.set_weather(weather)
            bp=world.get_blueprint_library().find('vehicle.tesla.model3');bp.set_attribute('role_name','m45_oracle');ego=session.spawn(bp,spawn);ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(30):world.tick()
            sensor=session.spawn(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),parent=ego);sensor.listen(lambda event:collision.append({'frame':event.frame,'other':event.other_actor.type_id}))
            lane=session.spawn(world.get_blueprint_library().find('sensor.other.lane_invasion'),carla.Transform(),parent=ego);lane.listen(lambda event:lane_events.append(event.frame))
            target=None;lead_control=PurePursuitPID();control=PurePursuitPID();last_snapshot=world.get_snapshot();last_good=None
            if scenario in ('static_obstacle','lead_braking'):
                index=min(25 if scenario=='static_obstacle' else 15,len(trace)-1);tf=trace[index][0].transform;tf.location.z+=.5
                target=session.spawn(bp,tf);target.apply_control(carla.VehicleControl(brake=1,hand_brake=scenario=='static_obstacle'))
                for _ in range(20):world.tick()
            for tick in range(cfg['max_ticks']):
                snapshot=world.get_snapshot();s=state_of(snapshot,ego);obstacles=[] if target is None else [obs_of(snapshot,target)];stop_distance=None
                if obstacles:
                    rel=np.array(obstacles[0]['xy'])-[s.x,s.y];long=rel@np.array([math.cos(s.yaw),math.sin(s.yaw)]);lat=abs(rel@np.array([-math.sin(s.yaw),math.cos(s.yaw)]))
                    if long>0 and lat<3.:stop_distance=max(0,long-ego.bounding_box.extent.x-target.bounding_box.extent.x)
                t0=time.perf_counter();trajectory,index=local_trajectory(route,s,max_speed=cfg['max_speed_mps'],stop_distance=stop_distance)
                forced=None
                if scenario=='emergency' and tick>=65:forced='emergency_command'
                elif scenario=='controller_timeout' and tick>=65:trajectory.timestamp_s=s.timestamp_s-1
                elif scenario=='invalid_trajectory' and tick>=65:trajectory.points[0,0]=np.nan
                elif scenario=='no_trajectory' and tick>=65:trajectory=None
                def road_ok(x,y):
                    wp=cmap.get_waypoint(carla.Location(x=float(x),y=float(-y),z=ego.get_location().z),project_to_road=False,lane_type=carla.LaneType.Driving)
                    return wp is not None
                safety=forced or safety_check(trajectory,s,obstacles,half_length=ego.bounding_box.extent.x,half_width=ego.bounding_box.extent.y,road_predicate=road_ok)
                # Red/yellow traffic stop only; no right-of-way/yield completeness claimed.
                if ego.is_at_traffic_light() and str(ego.get_traffic_light_state()) in ('Red','Yellow'):safety=safety or 'traffic_signal_stop'
                command=control.command(trajectory,s,safety);latencies.append((time.perf_counter()-t0)*1000)
                if safety:reason_counts[safety]=reason_counts.get(safety,0)+1
                ego.apply_control(carla.VehicleControl(steer=command['steer'],throttle=command['throttle'],brake=command['brake']))
                if target is not None and scenario=='lead_braking':
                    ls=state_of(snapshot,target);lt,_=local_trajectory(route,ls,max_speed=2.5 if tick<100 else 0.);lc=lead_control.command(lt,ls);target.apply_control(carla.VehicleControl(steer=lc['steer'],throttle=lc['throttle'],brake=lc['brake']))
                lateral=float(np.min(np.linalg.norm(route[:,:2]-[s.x,s.y],axis=1)));target_speed=0 if trajectory is None else float(trajectory.points[0,3]);completion=index/(len(route)-1);distance_goal=float(np.linalg.norm(route[-1,:2]-[s.x,s.y]))
                clean_points=None if trajectory is None else [[float(v) if np.isfinite(v) else None for v in row] for row in trajectory.points]
                rows.append({'frame':snapshot.frame,'timestamp_s':s.timestamp_s,'x':s.x,'y':s.y,'yaw':s.yaw,'speed_mps':s.speed_mps,'target_speed_mps':target_speed,'trajectory':clean_points,'trajectory_timestamp_s':None if trajectory is None else trajectory.timestamp_s,'command':command,'safety_reason':safety,'route_completion':completion,'lateral_tracking_error_m':lateral,'stop_distance_m':stop_distance,'goal_distance_m':distance_goal,'planner_and_safety_ms':latencies[-1]})
                rows[-1]['obstacles']=obstacles
                stopped=s.speed_mps<.15
                if scenario in ('static_obstacle','lead_braking'):met=stopped and tick>130 and stop_distance is not None and 2.<stop_distance<15.
                elif scenario in ('emergency','controller_timeout','invalid_trajectory','no_trajectory'):met=stopped and tick>70 and safety is not None
                else:met=stopped and distance_goal<2.5 and completion>.94
                stop_stable=stop_stable+1 if met else 0
                if stop_stable>=12:success=True;failure=None;break
                if collision:failure='collision';break
                if lateral>3.:failure='route_departure';break
                world.tick();time.sleep(.003)
        # Cleanup and a small event callback drain occur before recording collision totals.
        time.sleep(.1);duration=rows[-1]['timestamp_s']-rows[0]['timestamp_s'];brakes=sum(1 for i,r in enumerate(rows) if r['safety_reason'] and (i==0 or rows[i-1]['safety_reason']!=r['safety_reason']))
        logfile=root/(scenario+'_controls.jsonl')
        with logfile.open('x') as f:
            for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
        result={'status':'Passed' if success and not collision else 'Failed','failure_reason':failure,'route_completion':max(r['route_completion'] for r in rows),'collisions':collision,'collision_count':len(collision),'lane_invasions':len(lane_events),
            'lateral_error_mean_m':float(np.mean([r['lateral_tracking_error_m'] for r in rows])),'lateral_error_max_m':max(r['lateral_tracking_error_m'] for r in rows),
            'speed_error_mae_mps':float(np.mean([abs(r['speed_mps']-r['target_speed_mps']) for r in rows])),'max_speed_mps':max(r['speed_mps'] for r in rows),
            'control_frequency_sim_hz':(len(rows)-1)/duration if duration else None,'control_frequency_wall_hz':len(rows)/(time.monotonic()-wall_start),'planner_safety_p50_ms':float(np.median(latencies)),'planner_safety_p95_ms':float(np.quantile(latencies,.95)),
            'emergency_stop_trigger_events':brakes,'safety_reason_ticks':reason_counts,'ticks':len(rows),'last_stop_distance_m':rows[-1]['stop_distance_m'],'log':str(logfile),'log_sha256':file_hash(logfile)}
        report['scenarios'][scenario]=result;atomic_json(root/(scenario+'_result.json'),result);print(scenario,json.dumps(result),flush=True)
    report['passed_count']=sum(r['status']=='Passed' for r in report['scenarios'].values());report['all_passed']=report['passed_count']==len(report['scenarios']);atomic_json(a.output,report)

if __name__=='__main__':main()
