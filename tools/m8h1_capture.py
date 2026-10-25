"""Four-keyframe capture after actual settling gates. RGB-only packets, privileged sidecars."""
import json,queue,time
from pathlib import Path
import numpy as np
import cv2,yaml
from tools.m8h1_common import *
from tools.m8h1_preflight import tf
from simulator.m8f_placement import pose,box_record
from geometry.reference import intrinsics_from_hfov,camera_mount_to_hs_from_optical,decode_carla_depth_bgra
from datasets.m8e_reference import taxonomy,require_source_frame

def capture(session,group,reviews,contract,env,protocol,directory):
 import carla
 from tools.m8h1_actor_smoke import settle,evaluate_final,TemplateStop
 world=session.world;ego=reviews['ego'][0];targets={k:reviews[k][0] for k in ['vehicle','person']};rig=yaml.safe_load(Path('configs/cameras.yaml').read_text());rgbdir=directory/'rgb';side=directory/'evaluator';rgbdir.mkdir();side.mkdir();q=queue.Queue();lib=world.get_blueprint_library();records=[];ids={k:a.id for k,a in targets.items()};calibration=[]
 # Hold only already-qualified targets outside the scene. Never qualify this state.
 for j,actor in enumerate(targets.values()):
  actor.set_transform(carla.Transform(carla.Location(10000+j*10,10000,500)));actor.set_target_velocity(carla.Vector3D(0,0,0));actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
 for name,m in rig['cameras'].items():
  mount=carla.Transform(carla.Location(*m['xyz_m']),carla.Rotation(yaw=m['yaw_deg'],pitch=m['pitch_deg']))
  calibration.append(dict(name=name,K=intrinsics_from_hfov(640,384,100).tolist(),T_cam_to_ego=camera_mount_to_hs_from_optical(mount.get_matrix()).tolist(),mount_matrix=mount.get_matrix()))
  for kind in ['rgb','depth','instance_segmentation','semantic_segmentation']:
   bp=lib.find('sensor.camera.'+kind)
   for key,value in dict(image_size_x=640,image_size_y=384,fov=100,sensor_tick=0,lens_k=0,lens_kcube=0).items():bp.set_attribute(key,str(value))
   if kind=='rgb':bp.set_attribute('motion_blur_intensity','0')
   sensor=session.spawn(bp,mount,parent=ego);sensor.listen(lambda image,n=name,k=kind:q.put((n,k,image)))
 def tick():
  frame=world.tick();snap=world.get_snapshot();found={};deadline=time.monotonic()+45
  while len(found)<24:
   if time.monotonic()>deadline:raise TimeoutError('Missing24SensorFrame')
   n,k,img=q.get(timeout=10)
   if img.frame==frame:found[n,k]=img
   elif img.frame>frame:raise RuntimeError('FutureSensorFrame')
  return frame,snap,found
 # Sensors stream transient frames for synchronization; only four keyframes stored.
 for _ in range(protocol['sensor_warmup_ticks']):tick()
 for phase in protocol['phase_order']:
  checks={}
  if phase in ['present','shifted']:
   for role,actor in targets.items():
    where=group['poses'][role]['phases'][phase];actor.set_transform(tf(where['spawn_transform']));actor.set_target_velocity(carla.Vector3D(0,0,0));actor.set_target_angular_velocity(carla.Vector3D(0,0,0));initial=pose(actor.get_transform());s=settle(session,actor,initial,protocol['settling'],directory/(role+'_'+phase+'_capture_trace.jsonl'),tick=tick);g=evaluate_final(session,actor,where['preflight'],initial,contract,env,protocol['settling']);checks[role]=dict(actor_id=actor.id,initial=initial,settling=s,geometry=g)
    if s['status']!='SettledPoseObserved' or g['status']!='GroundedByGeometryProxy':
     write(side/(phase+'_failed_geometry.json'),checks);raise TemplateStop(phase+' '+role+' failed actual settling/support; captured frames retained')
  elif phase=='removed':
   for actor in targets.values():
    if not actor.destroy():raise RuntimeError('TargetRemovalFailed')
  for _ in range(protocol['phase_settle_sensor_ticks']):tick()
  frame,snap,images=tick();ego_state=snap.find(ego.id)
  if ego_state is None:raise RuntimeError('MissingActualEgo')
  E=np.array(ego_state.get_transform().get_matrix());cal=[]
  for c in calibration:cal.append(dict(c,T_world_from_camera=(E@np.array(c['mount_matrix'])).tolist()))
  packet=dict(schema='m8b.six_rgb.v1',token=group['template']+'_'+phase,camera_order=[c['name'] for c in cal],frame_id=frame,timestamp_s=snap.timestamp.elapsed_seconds,image_size=[384,640],cameras=[]);sensors={};closure=[]
  for c in cal:
   name=c['name'];sr={}
   for kind,short in [('rgb','rgb'),('depth','depth'),('instance_segmentation','instance'),('semantic_segmentation','semantic')]:
    img=images[name,kind];assert img.frame==frame and abs(img.timestamp-snap.timestamp.elapsed_seconds)<1e-5;arr=np.frombuffer(img.raw_data,np.uint8).reshape(384,640,4);path=(rgbdir if short=='rgb' else side)/(phase+'_'+name+('.png' if short=='rgb' else '_'+short+'.npz'))
    if path.exists():raise FileExistsError(path)
    if short=='rgb':assert cv2.imwrite(str(path),arr[:,:,:3])
    else:
     with path.open('xb') as f:
      if short=='depth':np.savez_compressed(f,depth_m=decode_carla_depth_bgra(img.raw_data,384,640),raw_bgra=arr)
      elif short=='instance':np.savez_compressed(f,semantic=arr[:,:,2],instance=arr[:,:,1].astype(np.uint16)*256+arr[:,:,0].astype(np.uint16),raw_bgra=arr)
      else:np.savez_compressed(f,semantic=arr[:,:,2],raw_bgra=arr)
    sr[short]=dict(**asset(path),frame_id=img.frame,timestamp_s=img.timestamp,transform=img.transform.get_matrix());delta=float(np.max(np.abs(np.array(img.transform.get_matrix())-c['T_world_from_camera'])));closure.append(dict(camera=name,kind=short,matrix_max_abs_error=delta,passed=delta<1e-4))
   sensors[name]=sr;packet['cameras'].append(dict(name=name,path=sr['rgb']['path'],sha256=sr['rgb']['sha256'],frame_id=frame,timestamp_s=snap.timestamp.elapsed_seconds,K=c['K'],T_cam_to_ego=c['T_cam_to_ego']))
  objects=[]
  for actor in world.get_actors():
   if actor.type_id.startswith(('sensor.','spectator','controller.')):continue
   state=snap.find(actor.id)
   if state is None:raise RuntimeError('ActorMissingSnapshot')
   trans=state.get_transform();tags=list(actor.semantic_tags);pc,basis=taxonomy(tags,actor.type_id)
   objects.append(dict(source='Actor',entity_id=actor.id,name=actor.type_id,semantic_tags=tags,entity_transform=trans.get_matrix(),snapshot_present=True,frame_id=frame,timestamp_s=snap.timestamp.elapsed_seconds,is_ego=actor.id==ego.id,is_controlled_target=actor.id in ids.values(),petr_class=pc,taxonomy_basis=basis,**box_record(actor.bounding_box,trans)))
  write(rgbdir/(phase+'_packet.json'),packet)
  s=dict(schema='EvaluatorSidecar.m8h1.v1',source='CARLA_Privileged_EvaluatorOnly',world_episode=world.id,map=world.get_map().name,group=group['template'],split='development',phase=phase,frame_id=frame,timestamp_s=snap.timestamp.elapsed_seconds,T_carla_world_from_ego=E.tolist(),actual_ego_pose=pose(ego_state.get_transform()),calibration=cal,objects=objects,sensors=sensors,environment_inventory=asset(ROOT/'environment_inventory.json'),controlled_entity_ids=list(ids.values()),target_roles=ids,actual_geometry=checks,camera_pose_closure=closure,packet=asset(rgbdir/(phase+'_packet.json')))
  require_source_frame(s);write(side/(phase+'_sidecar.json'),s);records.append(asset(side/(phase+'_sidecar.json')));write(directory/(phase+'_frame_manifest.json'),dict(frame=records[-1],pose_closure_pass=all(r['passed'] for r in closure)));print(group['template'],phase,'saved',frame,flush=True)
 write(directory/'manifest.json',dict(**provenance('Actual4Keyframes_NoDriving'),frames=records,controlled_entity_ids=list(ids.values()),all24sensors_synchronized=True,FinalEligible=False))
 return dict(status='CapturedPendingIdentityAudit',frames=records)
