"""Same-pose clear/fog six-camera RGB/depth/semantic diagnostic pairs."""
import argparse,json,time,queue
from pathlib import Path
import cv2,numpy as np,yaml
from simulator.session import CarlaSession
from simulator.recorder import atomic_json,file_hash
from geometry.reference import intrinsics_from_hfov,camera_mount_to_hs_from_optical,CARLA_EGO_TO_HS_EGO,transform_points,decode_carla_depth_bgra
from geometry.coordinates import carla_pose_to_hs


def semantic_groups(carla):
    groups={}
    for name in ['Roads','RoadLines','Sidewalks','Ground','Terrain','RailTrack']:
        if hasattr(carla.CityObjectLabel,name):groups[int(getattr(carla.CityObjectLabel,name))]=1
    for name in ['Buildings','Fences','Poles','Vegetation','Walls','TrafficSigns','Bridge','GuardRail','TrafficLight','Static']:
        if hasattr(carla.CityObjectLabel,name):groups[int(getattr(carla.CityObjectLabel,name))]=2
    for name in ['Pedestrians','Vehicles','Dynamic','Rider','Car','Truck','Bus','Train','Motorcycle','Bicycle']:
        if hasattr(carla.CityObjectLabel,name):groups[int(getattr(carla.CityObjectLabel,name))]=3
    return groups


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='/root/gpufree-data/hs-drive-data/raw/m45_semantic_pairs');p.add_argument('--port',type=int,default=2100);p.add_argument('--frames',type=int,default=8);a=p.parse_args()
    root=Path(a.output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(port=a.port,map='Town03')
    cameras=yaml.safe_load(Path('configs/cameras.yaml').read_text());q=queue.Queue(maxsize=500);cal={'schema_version':'hsd-m45-calibration-v1','cameras':{},'depth_convention':'optical_z'}
    with CarlaSession(cfg,load_map=True) as session:
        world=session.world;spawns=world.get_map().get_spawn_points();bp=world.get_blueprint_library().find('vehicle.tesla.model3');ego=session.spawn(bp,spawns[0]);ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
        for _ in range(30):world.tick()
        # Static front vehicle gives independently tagged obstacle surfaces and occlusion.
        wp=world.get_map().get_waypoint(ego.get_location());nexts=wp.next(14.)
        target=None
        if nexts:
            tf=nexts[0].transform;tf.location.z+=.5
            try:target=session.spawn(bp,tf);target.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            except RuntimeError:pass
        for _ in range(20):world.tick()
        base_pose=ego.get_transform();ego.set_simulate_physics(False);ego.set_transform(base_pose)
        for name,mount in cameras['cameras'].items():
            tf=carla.Transform(carla.Location(*mount['xyz_m']),carla.Rotation(yaw=mount['yaw_deg'],pitch=mount['pitch_deg']))
            cal['cameras'][name]={'K':intrinsics_from_hfov(640,384,100).tolist(),'T_hs_ego_from_opt':camera_mount_to_hs_from_optical(np.asarray(tf.get_matrix())).tolist(),'width':640,'height':384}
            for kind in ('rgb','depth','semantic_segmentation'):
                blue=world.get_blueprint_library().find('sensor.camera.'+kind)
                attrs={'image_size_x':640,'image_size_y':384,'fov':100,'sensor_tick':.05,'lens_k':0,'lens_kcube':0}
                if kind=='rgb':attrs.update(enable_postprocess_effects=True,motion_blur_intensity=0)
                for key,val in attrs.items():blue.set_attribute(key,str(val))
                sensor=session.spawn(blue,tf,parent=ego)
                sensor.listen(lambda image,n=name,k=kind:q.put((n,k,image),timeout=5))
        groups=semantic_groups(carla);summaries=[]
        for phase,fog in [('clear',2.),('fog',40.)]:
            weather=carla.WeatherParameters.ClearNoon;weather.sun_altitude_angle=50;weather.sun_azimuth_angle=180;weather.fog_density=fog;weather.fog_distance=10;world.set_weather(weather)
            for _ in range(20):
                world.tick();time.sleep(.08)
                while not q.empty():q.get_nowait()
            directory=root/phase;directory.mkdir();(directory/'frames').mkdir();atomic_json(directory/'calibration.json',cal)
            manifest={'schema_version':'hsd-m45-semantic-recording-v1','status':'recording','map':world.get_map().name,'carla':session.client.get_server_version(),'seed':7,'phase':phase,'frames':[],
                'semantic_groups':groups,'semantic_enum':{name:int(getattr(carla.CityObjectLabel,name)) for name in dir(carla.CityObjectLabel) if name[0].isupper() and hasattr(getattr(carla.CityObjectLabel,name),'__int__')},
                'protocol':'same map/pose/target/cameras/sun; only fog density changes; all frames diagnostic validation, no train/test splitting','weather':{k:getattr(weather,k) for k in ['cloudiness','precipitation','sun_altitude_angle','sun_azimuth_angle','fog_density','fog_distance']},'target_spawned':target is not None}
            for i in range(a.frames):
                frame=world.tick();snapshot=world.get_snapshot();found={};deadline=time.monotonic()+30
                while len(found)<18:
                    if time.monotonic()>deadline:raise TimeoutError('Incomplete semantic camera frame')
                    n,k,img=q.get(timeout=10)
                    if img.frame==frame:found[n,k]=img
                    elif img.frame>frame:raise ValueError('Future image before complete set')
                pose=np.asarray(snapshot.find(ego.id).get_transform().get_matrix());corners=np.array([[v.x,v.y,v.z] for v in ego.bounding_box.get_world_vertices(snapshot.find(ego.id).get_transform())]);corners=transform_points(corners,CARLA_EGO_TO_HS_EGO@np.linalg.inv(pose))
                f={'frame_id':frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,'T_world_from_ego':carla_pose_to_hs(pose).tolist(),'ego_bbox_corners_hs':corners.tolist(),'cameras':{}}
                for name in cameras['cameras']:
                    entry={}
                    for kind,short in [('rgb','rgb'),('depth','depth'),('semantic_segmentation','semantic')]:
                        img=found[name,kind]
                        if abs(img.timestamp-f['timestamp_s'])>1e-5:raise ValueError('Sensor time mismatch')
                        folder=directory/short/name;folder.mkdir(parents=True,exist_ok=True);rel=f'{short}/{name}/{frame:08d}'+('.png' if short=='rgb' else '.npz');path=directory/rel
                        array=np.frombuffer(img.raw_data,np.uint8).reshape(384,640,4)
                        if short=='rgb':cv2.imwrite(str(path),array[:,:,:3])
                        elif short=='depth':np.savez_compressed(path,depth_m=decode_carla_depth_bgra(img.raw_data,384,640))
                        else:np.savez_compressed(path,semantic=array[:,:,2])
                        entry[short]=rel;entry[short+'_sha256']=file_hash(path)
                    f['cameras'][name]=entry
                atomic_json(directory/'frames'/f'{frame:08d}.json',f);manifest['frames'].append(frame);print(phase,i+1,a.frames,flush=True)
            manifest['status']='complete';atomic_json(directory/'m45_manifest.json',manifest);summaries.append({'scene':str(directory),'frames':a.frames,'manifest_sha256':file_hash(directory/'m45_manifest.json')})
        atomic_json(root/'pair_manifest.json',{'status':'complete','pairs':summaries,'pose_fixed':True,'motion_pair_collected':False,'split':'diagnostic_val','unseen_blind_test':False})

if __name__=='__main__':main()
