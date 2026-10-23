"""Independent CARLA script collection. Imports no HS planning/control modules."""
import argparse,json,math,time
from pathlib import Path
import carla
import numpy as np
from tools.m7d_common import *
from simulator.m6b_obstacles import snapshot_obstacle
from tools.m47_common import physics_dict


def lane_start(m,road):
 ws=[w for w in m.generate_waypoints(1.) if w.road_id==road and not w.is_junction and w.lane_type==carla.LaneType.Driving]
 lane=min(w.lane_id for w in ws);ws=sorted([w for w in ws if w.lane_id==lane],key=lambda w:w.s,reverse=lane>0)
 return ws[7]


def forward(w,distance):
 for _ in range(int(distance/.5)):
  ns=w.next(.5)
  if not ns:break
  # Collection-only deterministic generator route, not exposed as target actor intent.
  w=min(ns,key=lambda n:abs((n.transform.rotation.yaw-w.transform.rotation.yaw+180)%360-180))
 return w


def script_command(actor,target,m):
 tf=actor.get_transform();vel=actor.get_velocity();v=vel.x*math.cos(math.radians(tf.rotation.yaw))+vel.y*math.sin(math.radians(tf.rotation.yaw))
 w=m.get_waypoint(tf.location);aim=forward(w,4.).transform.location;heading=math.degrees(math.atan2(aim.y-tf.location.y,aim.x-tf.location.x));error=(heading-tf.rotation.yaw+180)%360-180
 return carla.VehicleControl(throttle=float(np.clip(.22+.22*(target-v),0,.55)) if target>0.05 and v<target+.1 else 0.,brake=float(np.clip(.2*(v-target),0,.35)) if v>target+.1 else (.2 if target==0 else 0.),steer=float(np.clip(error*.025,-.5,.5)),hand_brake=False)


def record_actor(snapshot,actor,role):
 o=snapshot_obstacle(snapshot,actor);s=snapshot.find(actor.id);tf=s.get_transform();v=s.get_velocity();av=s.get_angular_velocity()
 o.update(blueprint=actor.type_id,role=role,position_carla=[tf.location.x,tf.location.y,tf.location.z],rotation_carla=[tf.rotation.roll,tf.rotation.pitch,tf.rotation.yaw],velocity_carla=[v.x,v.y,v.z],yaw_rate=-math.radians(av.z),category='Person' if actor.type_id.startswith('walker') else 'Vehicle')
 return o


def run(split):
 cfg=read('configs/m7d/protocol.json');client=carla.Client('127.0.0.1',cfg['port']);client.set_timeout(30);assert client.get_server_version()=='0.9.15'
 for spec in cfg['episodes']:
  if spec['split']!=split:continue
  root=Path('reports/m7d_episodes')/spec['id'];root.mkdir(parents=True,exist_ok=False);write(root/'config.json',dict(spec=spec,provenance=provenance(),control_source='IndependentCollectionScript_NoHSPlanner'))
  w=client.load_world('Town03');m=w.get_map();original=w.get_settings();settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=True;w.apply_settings(settings);w.set_weather(carla.WeatherParameters.ClearNoon)
  actors=[];rows=[];collisions=[];startwp=lane_start(m,spec['road'])
  try:
   for role,wp,bp in [('ego',startwp,'vehicle.tesla.model3'),('lead',forward(startwp,18.),'vehicle.audi.a2')]:
    tf=wp.transform;tf.location.z+=.8;a=w.spawn_actor(w.get_blueprint_library().find(bp),tf);actors.append(a);a.apply_control(carla.VehicleControl(brake=1.,hand_brake=True))
   ego,lead=actors;walker=None
   if spec['kind']=='crossing':
    wp=forward(startwp,14.);tf=wp.transform;ang=math.radians(tf.rotation.yaw);tf.location.x+=-math.sin(ang)*4;tf.location.y+=math.cos(ang)*4;tf.location.z+=1
    bp=w.get_blueprint_library().find('walker.pedestrian.0001');bp.set_attribute('is_invincible','true');walker=w.spawn_actor(bp,tf);actors.append(walker)
   for _ in range(40):w.tick()
   collision=w.spawn_actor(w.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=ego);actors.append(collision);collision.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.id)))
   write(root/'physics.json',dict(ego=physics_dict(ego),lead=physics_dict(lead)));write(root/'body.json',dict(half_extent=[ego.bounding_box.extent.x,ego.bounding_box.extent.y],offset=[ego.bounding_box.location.x,-ego.bounding_box.location.y]))
   route=[];cur=startwp
   for _ in range(161):
    tf=cur.transform;route.append([tf.location.x,-tf.location.y,-math.radians(tf.rotation.yaw)]);n=cur.next(.5)
    if not n:break
    if len(n)>1:break # no inferred intent from future at branch
    cur=n[0]
   write(root/'route.json',dict(source='Privileged_current_predeclared_ego_route',points=route,actor_intent=None))
   if not Path('reports/m7d_execution_logs/Town03.xodr').exists():Path('reports/m7d_execution_logs/Town03.xodr').write_text(m.to_opendrive())
   start=w.get_snapshot().timestamp.elapsed_seconds
   for tick in range(int(cfg['episode_seconds']/.05)+1):
    snap=w.get_snapshot();available=time.perf_counter();t=snap.timestamp.elapsed_seconds-start;es=snap.find(ego.id);tf=es.get_transform();v=es.get_velocity();yaw=math.radians(tf.rotation.yaw);signed=v.x*math.cos(yaw)+v.y*math.sin(yaw)
    obs=[record_actor(snap,lead,'static' if spec['kind']=='static' else 'lead')]
    if walker:obs.append(record_actor(snap,walker,'crossing'))
    leadtarget=0 if spec['kind']=='static' or (spec['kind']=='brake_restart' and 4<=t<9) else 2.2
    egotarget=spec['ego_speed'] if not(spec['kind']=='brake_restart' and 6<=t<10) else 0.
    ec=script_command(ego,egotarget,m);lc=script_command(lead,leadtarget,m);ego.apply_control(ec);lead.apply_control(lc)
    wc=None
    if walker:
     angle=math.radians(startwp.transform.rotation.yaw);wc=carla.WalkerControl(direction=carla.Vector3D(math.sin(angle),-math.cos(angle),0),speed=1.0 if 3<=t<12 else 0.);walker.apply_control(wc)
    applied=lambda c:dict(throttle=c.throttle,brake=c.brake,steer=c.steer,hand_brake=c.hand_brake)
    rows.append(dict(source='Privileged_ActualCARLAActorSnapshot',episode_id=spec['id'],frame=snap.frame,sim_time=snap.timestamp.elapsed_seconds,input_available_wall_s=available,valid=True,missing=False,ego=dict(actor_id=ego.id,xy=[tf.location.x,-tf.location.y],z=tf.location.z,yaw=-yaw,signed_speed=signed,velocity=[v.x,-v.y],position_carla=[tf.location.x,tf.location.y,tf.location.z],rotation_carla=[tf.rotation.roll,tf.rotation.pitch,tf.rotation.yaw]),actors=obs,commands=dict(source='IndependentCollectionScript',ego=applied(ec),lead=applied(lc),ego_target=egotarget,lead_target=leadtarget,walker_speed=None if wc is None else wc.speed)))
    if tick<int(cfg['episode_seconds']/.05):w.tick()
   with (root/'trace.jsonl').open('x') as f:
    for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
   write(root/'result.json',dict(kind='ActualCARLA0915IndependentScriptCollection',provenance=provenance(),episode_id=spec['id'],split=split,rows=len(rows),duration=rows[-1]['sim_time']-rows[0]['sim_time'],source='Privileged',carla_version=client.get_server_version(),map=m.name,collisions=collisions,trace=asset(root/'trace.jsonl'),ego_control_from_M7D_Planner=False,physics_modified=False,single_tick_owner=True))
   print(spec['id'],'recorded',len(rows),'collisions',len(collisions),flush=True)
  except Exception as e:
   write(root/'failure.json',dict(error=repr(e),rows_collected=len(rows)));raise
  finally:
   for a in reversed(actors):
    if a.is_alive:a.destroy()
   w.apply_settings(original)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['development','validation','fresh_final'],required=True);a=ap.parse_args();run(a.split)
