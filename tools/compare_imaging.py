"""Controlled same-pose photometry/resolution experiment on all twelve sensors."""
import argparse,json,subprocess,time,threading
from pathlib import Path
from queue import Empty
import numpy as np
import yaml
from simulator.session import CarlaSession
from simulator.camera_rig import spawn_rig
from simulator.imaging import load_camera_config
from simulator.synchronizer import FrameSynchronizer
from simulator.snapshot import extract_snapshot
from simulator.recorder import Recorder,atomic_json
from datasets.schema import SENSOR_IDS
from tools.provenance import source_hashes


class GpuSampler:
    """Whole-device samples, not CUDA allocator or per-process peak measurements."""
    def __init__(self):self.samples=[];self.stop=threading.Event();self.errors=[]
    def run(self):
        while not self.stop.is_set():
            try:
                output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()[0]
                memory,util=map(float,output.split(','))
                self.samples.append({'monotonic_s':time.monotonic(),'memory_mib':memory,'utilization_pct':util})
            except Exception as exc:self.errors.append(str(exc))
            self.stop.wait(.5)
    def __enter__(self):self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join(timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--config',default='configs/carla.yaml')
    p.add_argument('--profiles',nargs='+',default=[f'configs/imaging/{m}_{w}.yaml' for w in (640,960,1280) for m in ('geometry','training')])
    p.add_argument('--render-every-tick', action='store_true')
    p.add_argument('--weather',choices=['noon','sunset','night'],default='noon')
    p.add_argument('--frames',type=int,default=11);p.add_argument('--warmup-ticks',type=int,default=40)
    p.add_argument('--tick-delay',type=float,default=.3)
    a=p.parse_args()
    if a.frames<2 or a.warmup_ticks<0 or a.tick_delay<0:p.error('Invalid capture limits')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(Path(a.config).read_text())
    import carla
    results=[]
    with CarlaSession(cfg,load_map=True) as session:
        world=session.world;original_weather=world.get_weather()
        try:
            weather=carla.WeatherParameters.ClearNoon
            weather.sun_altitude_angle={'noon':50,'sunset':8,'night':-15}[a.weather]
            weather.sun_azimuth_angle=30
            world.set_weather(weather)
            bp=world.get_blueprint_library().find('vehicle.tesla.model3')
            bp.set_attribute('color','180,180,180')
            ego=session.spawn(bp,world.get_map().get_spawn_points()[0]);ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(20):world.tick()
            ego.set_simulate_physics(False)
            pose=ego.get_transform()
            target=session.spawn(bp,carla.Transform(pose.transform(carla.Location(x=25,z=.4)),pose.rotation))
            target.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(20):world.tick()
            target.set_simulate_physics(False)
            catalog={x.id:(x.type_id,x.bounding_box) for x in (ego,target)}
            for profile in a.profiles:
                cameras=load_camera_config('configs/cameras.yaml',profile)
                if a.render_every_tick:cameras['render_sensor_tick_s']=0.0
                scene=output/cameras['imaging_profile_id']
                ratio=cameras['sensor_tick_s']/cfg['fixed_delta_seconds']
                if ratio<1 or not np.isclose(ratio,round(ratio)):raise ValueError('Noninteger output cadence')
                sync=FrameSynchronizer(SENSOR_IDS,cadence_ticks=round(ratio))
                before=set(x.id for x in session.sensors)
                recorder=None;status='failed';tick_times=[];write_times=[];capture_start=None
                start=time.monotonic()
                with GpuSampler() as gpu:
                    try:
                        calibration=spawn_rig(session,ego,cameras,sync)
                        metadata={'carla_version':session.client.get_server_version(),'map':world.get_map().name,
                            'config':cfg,'camera_config':cameras,'weather':{'preset':a.weather,'sun_altitude_angle':weather.sun_altitude_angle,'sun_azimuth_angle':30},
                            'source_sha256':source_hashes(),'render_quality_declared':'Epic',
                            'capture_policy':'same_ego_target_pose_frozen_physics','ego_pose_carla':ego.get_transform().get_matrix(),
                            'target_pose_carla':target.get_transform().get_matrix(),'target_actor_id':target.id,
                            'profile_file':profile,'warmup_ticks':a.warmup_ticks,'tick_wall_delay_s':a.tick_delay}
                        recorder=Recorder(scene,calibration,metadata);sync.on_event=recorder.event
                        for _ in range(a.warmup_ticks):
                            world.tick();time.sleep(a.tick_delay)
                            while True:
                                try:sync.queue.get_nowait()
                                except Empty:break
                        sync.begin_interval(world.get_snapshot().frame + 1, explicit_cadence=a.render_every_tick)
                        capture_start=time.monotonic()
                        for _ in range(round(ratio)*a.frames+80):
                            now=time.monotonic();frame=world.tick();tick_times.append(time.monotonic()-now)
                            snap=world.get_snapshot()
                            if snap.frame!=frame:raise RuntimeError('Competing tick client')
                            state,objects=extract_snapshot(snap,ego,catalog)
                            sync.add_snapshot(frame,snap.timestamp.elapsed_seconds,state,objects)
                            time.sleep(a.tick_delay)
                            for bundle in sync.drain():
                                if recorder.committed<a.frames:
                                    now=time.monotonic();recorder.write(bundle);write_times.append(time.monotonic()-now)
                            if recorder.committed==a.frames:status='complete';break
                        if status!='complete':raise TimeoutError('Insufficient complete frames')
                        elapsed=time.monotonic()-capture_start
                    finally:
                        for sensor in list(session.sensors):
                            if sensor.id not in before:
                                sensor.stop();sensor.destroy();session.sensors.remove(sensor);session.actors.remove(sensor)
                        sync.finish()
                        if recorder:recorder.close(status,sync.stats)
                benchmark={'scene':str(scene),'status':status,'profile':profile,'capture_wall_s':elapsed,
                    'total_wall_s':time.monotonic()-start,'complete_frames':recorder.committed,
                    'complete_frames_per_wall_s':recorder.committed/elapsed,
                    'tick_rpc_p50_s':float(np.median(tick_times)),'tick_rpc_p95_s':float(np.quantile(tick_times,.95)),
                    'write_p50_s':float(np.median(write_times)),
                    'gpu_peak_sampled_mib':max((s['memory_mib'] for s in gpu.samples),default=None),
                    'gpu_samples':gpu.samples,'gpu_sampling_errors':gpu.errors,
                    'bytes_on_disk':sum(f.stat().st_size for f in scene.rglob('*') if f.is_file()),
                    'note':'Includes tick delay and PNG/NPZ write; whole-device GPU samples at ~0.5s, not exact peak.'}
                atomic_json(scene/'benchmark.json',benchmark);results.append(benchmark)
                atomic_json(output/'experiment.json',{'weather':a.weather,'results':results,'status':'running'})
                print(json.dumps({k:v for k,v in benchmark.items() if k not in ('gpu_samples','note')},indent=2),flush=True)
        finally:world.set_weather(original_weather)
    atomic_json(output/'experiment.json',{'weather':a.weather,'results':results,'status':'complete'})


if __name__=='__main__':main()
