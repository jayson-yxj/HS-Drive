"""Capture one synchronized native CARLA RGB buffer packet for timing replay."""
import queue,json
from pathlib import Path
import numpy as np,yaml,carla
from tools.m5k_common import write,asset
from simulator.session import CarlaSession
from geometry.reference import intrinsics_from_hfov,camera_mount_to_hs_from_optical
from geometry.coordinates import carla_pose_to_hs

def main():
 out=Path('runs/m5k_native_buffer');out.mkdir(exist_ok=False);cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town01');rig=yaml.safe_load(Path('configs/cameras.yaml').read_text());q=queue.Queue();K=[];T=[]
 with CarlaSession(cfg,load_map=True)as session:
  pose=session.world.get_map().get_spawn_points()[0];world=session.world;R=np.array(pose.get_matrix())
  for name,mount in rig['cameras'].items():
   local=carla.Transform(carla.Location(*mount['xyz_m']),carla.Rotation(yaw=mount['yaw_deg'],pitch=mount['pitch_deg']));matrix=R@np.array(local.get_matrix());rot=matrix[:3,:3];tf=carla.Transform(carla.Location(*matrix[:3,3]),carla.Rotation(pitch=float(np.degrees(np.arcsin(rot[2,0]))),yaw=float(np.degrees(np.arctan2(rot[1,0],rot[0,0]))),roll=float(np.degrees(np.arctan2(-rot[2,1],rot[2,2])))))
   bp=world.get_blueprint_library().find('sensor.camera.rgb')
   for key,value in dict(image_size_x=640,image_size_y=384,fov=100,sensor_tick=0,lens_k=0,lens_kcube=0,motion_blur_intensity=0).items():bp.set_attribute(key,str(value))
   sensor=session.spawn(bp,tf);sensor.listen(lambda image,n=name:q.put((n,image)));K.append(intrinsics_from_hfov(640,384,100).tolist());T.append(camera_mount_to_hs_from_optical(local.get_matrix()).tolist())
  packet={}
  for _ in range(5):
   frame=world.tick();packet={}
   while len(packet)<6:
    name,image=q.get(timeout=30)
    if image.frame==frame:packet[name]=image
  ts=world.get_snapshot().timestamp.elapsed_seconds;buffers=[]
  for name in rig['cameras']:
   im=packet[name];assert abs(im.timestamp-ts)<1e-6;path=out/(name+'.bgra');path.write_bytes(bytes(im.raw_data));buffers.append(dict(camera=name,frame=im.frame,timestamp=im.timestamp,**asset(path)))
  values=dict(intrinsics=(K,'float32'),cam_to_ego=(T,'float32'),ego_pose=(carla_pose_to_hs(R).tolist(),'float64'),ego_state=([0]*7,'float32'),timestamps_s=(ts,'float64'),route_command=(0,'int64'),valid_cameras=([True]*6,'bool'))
  write(out/'manifest.json',dict(kind='actual_CARLA_six_RGB_native_buffer',buffers=buffers,metadata={k:dict(value=[v],dtype=dtype)for k,(v,dtype)in values.items()},limits='Static sensors, no ego controls, recorded for cost replay only'))
if __name__=='__main__':main()
