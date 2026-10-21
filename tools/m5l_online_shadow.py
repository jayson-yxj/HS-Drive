"""Actual independent 2Hz six-RGB shadow under a sole dedicated tick owner."""
import time,json,queue,threading,multiprocessing as mp
from pathlib import Path
import numpy as np,yaml
from tools.m5l_common import write,asset,summary
from simulator.m5l_shadow import worker,latest_put

def main():
 import sys
 sys.path.append(str(Path(".venv/lib/python3.10/site-packages").resolve()))
 import carla
 from simulator.session import CarlaSession
 from geometry.reference import intrinsics_from_hfov,camera_mount_to_hs_from_optical
 from geometry.coordinates import sys
 sys.path.append(str(Path(".venv/lib/python3.10/site-packages").resolve()))
 import carla_pose_to_hs
 ctx=mp.get_context('spawn');channel=ctx.Queue(1);results=ctx.Queue(128);stop=ctx.Event();ready=ctx.Event();proc=ctx.Process(target=worker,args=(channel,results,stop,ready));proc.start()
 if not ready.wait(90):stop.set();proc.terminate();raise TimeoutError('Workerinit')
 cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=2100,map='Town01');rig=yaml.safe_load(Path('configs/cameras.yaml').read_text());packets={};lock=threading.Lock();measure=[];received=[];sent=0;drops=0;aborted=False
 try:
  with CarlaSession(cfg,load_map=True)as session:
   world=session.world;pose=world.get_map().get_spawn_points()[0];R=np.array(pose.get_matrix());K=[];T=[]
   def callback(name,image):
    wall=time.perf_counter();raw=bytes(image.raw_data)
    with lock:
     packet=packets.setdefault(image.frame,dict(timestamp=image.timestamp,wall=wall,buffers={}));packet['buffers'][name]=raw;packet.setdefault('sensor_timestamps',{})[name]=image.timestamp
     for frame in list(packets):
      if frame<image.frame-3:del packets[frame]
   for name,mount in rig['cameras'].items():
    local=carla.Transform(carla.Location(*mount['xyz_m']),carla.Rotation(yaw=mount['yaw_deg'],pitch=mount['pitch_deg']));matrix=R@np.array(local.get_matrix());rot=matrix[:3,:3];tf=carla.Transform(carla.Location(*matrix[:3,3]),carla.Rotation(pitch=float(np.degrees(np.arcsin(rot[2,0]))),yaw=float(np.degrees(np.arctan2(rot[1,0],rot[0,0]))),roll=float(np.degrees(np.arctan2(-rot[2,1],rot[2,2])))));bp=world.get_blueprint_library().find('sensor.camera.rgb')
    for key,value in dict(image_size_x=640,image_size_y=384,fov=100,sensor_tick=0,lens_k=0,lens_kcube=0,motion_blur_intensity=0).items():bp.set_attribute(key,str(value))
    sensor=session.spawn(bp,tf);sensor.listen(lambda im,n=name:callback(n,im));K.append(intrinsics_from_hfov(640,384,100).tolist());T.append(camera_mount_to_hs_from_optical(local.get_matrix()).tolist())
   values=dict(intrinsics=(K,'float32'),cam_to_ego=(T,'float32'),ego_pose=(carla_pose_to_hs(R).tolist(),'float64'),ego_state=([0]*7,'float32'),timestamps_s=(1.,'float64'),route_command=(0,'int64'),valid_cameras=([True]*6,'bool'));metadata={k:dict(value=[v],dtype=t)for k,(v,t)in values.items()}
   last_sent=-1;last_send_wall=-1.
   for phase in ['baseline','shadow','recovery']:
    times=[];intervals=[];prev=None
    for i in range(260):
     begin=time.perf_counter();fid=world.tick();tick_end=time.perf_counter();packet=None
     if phase=='shadow'and begin-last_send_wall>=.5:
      with lock:
       complete=[f for f,p in packets.items()if len(p['buffers'])==6 and max(p['sensor_timestamps'].values())-min(p['sensor_timestamps'].values())<1e-6 and f>last_sent]
       if complete:
        f=max(complete);p=packets[f];packet=dict(frame=f,timestamp_s=p['timestamp'],captured_wall=p['wall'],buffers=[p['buffers'][n]for n in rig['cameras']],metadata=metadata);last_sent=f
      if packet:
       ok=latest_put(channel,packet);sent+=int(ok);drops+=int(not ok);last_send_wall=begin
     while True:
      try:received.append(results.get_nowait())
      except queue.Empty:break
     if i>=20:
      times.append((tick_end-begin)*1000)
      if prev is not None:intervals.append((begin-prev)*1000)
     prev=begin
     if phase=='shadow'and i>=60:
      base=measure[0]['tick_ms']['p95'];p95=float(np.quantile(times,.95))
      if p95>max(base*1.5,base+20)or max(times)>1000:aborted=True;stop.set();break
     time.sleep(max(0,.05-(time.perf_counter()-begin)))
    measure.append(dict(phase=phase,tick_ms=summary(times),interval_ms=summary(intervals),raw_tick_ms=times,raw_interval_ms=intervals))
 finally:
  stop.set();proc.join(10)
  if proc.is_alive():proc.terminate();proc.join(5)
  while True:
   try:received.append(results.get_nowait())
   except queue.Empty:break
 raw='runs/m5l_logs/shadow_raw.json';write(raw,dict(phases=measure,outputs=received));write('reports/m5l_online_shadow.json',dict(kind='actual_CARLA_online_sixcamera_independent_process_no_control',phases=[{k:v for k,v in r.items()if not k.startswith('raw')}for r in measure],sent=sent,owner_nonblocking_queue_drops=drops,outputs=len(received),status_counts={s:sum(r['status']==s for r in received)for s in {r['status']for r in received}},timing={key:summary([r[key]for r in received if key in r])for key in ['queue_age_ms','convert_h2d_ms','inference_ms','output_ms','end_to_end_ms']},stopped_for_tick_tail=aborted,worker_exitcode=proc.exitcode,raw=asset(raw),limits='End-to-end startsfirstsensorcallback, not renderingstart. StaticdetachedRGBsensors no drivingroute/controller resourcecontention. Sole harness TickOwner, worker noCARLAclient/no control. 2Hz wallclock throttle, originalsimtimestamp unchanged; agealso usesmonotonicwall. Rawbytes copiedcallback, MP queue serializedasync boundedlatestcomplete. 20warmupticks perphase, finite240ticks. Doesnotcertify drivingreal-time or purevisualsafety.'))
if __name__=='__main__':main()
