"""Independent road-plane/collision-ray experiment for CARLA depth semantics.

A downward camera observes road geometry. Collision ray hits (not depth values)
define a plane; only coplanar Road hits are compared. Saves raw depth, mask,
plane and residuals at central/edge pixels. Does not certify all scene geometry.
"""
import argparse
import json
import time
from pathlib import Path
from queue import Queue, Empty
import cv2
import numpy as np
import yaml
from simulator.session import CarlaSession
from simulator.recorder import atomic_json
from geometry.reference import (intrinsics_from_hfov, OPTICAL_TO_CARLA_CAM,
                                transform_points, decode_carla_depth_bgra)
from geometry.oracle_bev import plane_depth_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/carla_smoke_town10.yaml')
    p.add_argument('--output',required=True)
    p.add_argument('--spawn-index',type=int,default=0)
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    import carla
    with CarlaSession(cfg) as session:
        world=session.world
        origin=world.get_map().get_spawn_points()[a.spawn_index].location
        # Independently survey nearby road collision geometry.
        survey=[]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                hits=world.cast_ray(carla.Location(origin.x+dx,origin.y+dy,origin.z+10),
                                    carla.Location(origin.x+dx,origin.y+dy,origin.z-10))
                if hits and hits[0].label==carla.CityObjectLabel.Roads:
                    q=hits[0].location
                    survey.append([q.x,q.y,q.z])
        if len(survey)<4:
            raise RuntimeError('Need >=4 independent road hits; choose a different spawn-index')
        survey=np.array(survey)
        center=survey.mean(axis=0)
        _,_,vh=np.linalg.svd(survey-center)
        normal=vh[-1]
        if normal[2]<0:normal=-normal
        offset=float(normal@center)
        residual=float(np.max(np.abs(survey@normal-offset)))
        if residual>.02:
            raise RuntimeError('Survey is not a plane within 2cm; choose another location')
        pose=carla.Transform(carla.Location(center[0],center[1],center[2]+10),carla.Rotation(pitch=-90))
        bp=world.get_blueprint_library().find('sensor.camera.depth')
        for key,value in {'image_size_x':'640','image_size_y':'384','fov':'100','sensor_tick':'0.1',
                           'lens_k':'0','lens_kcube':'0'}.items():bp.set_attribute(key,value)
        sensor=session.spawn(bp,pose)
        q=Queue(100)
        sensor.listen(q.put_nowait)
        image=None
        for _ in range(30):
            world.tick();time.sleep(.1)
            while True:
                try:image=q.get_nowait()
                except Empty:break
        if image is None:raise RuntimeError('No depth image')
        depth=decode_carla_depth_bgra(image.raw_data,image.height,image.width)
        K=intrinsics_from_hfov(640,384,100)
        T=np.asarray(image.transform.get_matrix())@OPTICAL_TO_CARLA_CAM
        n_opt=T[:3,:3].T@normal
        d_opt=offset-normal@T[:3,3]
        mask=np.zeros(depth.shape,dtype=bool)
        rows=[]
        for v in np.linspace(10,373,25).astype(int):
            for u in np.linspace(10,629,41).astype(int):
                ray=np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1.])
                direction=T[:3,:3]@ray
                direction/=np.linalg.norm(direction)
                start=T[:3,3]+direction*.1
                end=T[:3,3]+direction*60
                hits=world.cast_ray(carla.Location(*start),carla.Location(*end))
                if not hits:continue
                hit=hits[0]
                h=np.array([hit.location.x,hit.location.y,hit.location.z])
                if hit.label!=carla.CityObjectLabel.Roads or abs(normal@h-offset)>.02:continue
                mask[v,u]=True
                optical=transform_points([h],np.linalg.inv(T))[0]
                rows.append({'u':int(u),'v':int(v),'depth_m':float(depth[v,u]),
                    'collision_z_m':float(optical[2]),'collision_range_m':float(np.linalg.norm(optical))})
        v,u=np.indices(depth.shape)
        x=(u-319.5)/320;y=(v-191.5)/192
        regions={'all':mask,'center':mask & (abs(x)<.25)&(abs(y)<.25),
                 'edge':mask & ((abs(x)>.7)|(abs(y)>.7))}
        results={name:plane_depth_comparison(depth,K,n_opt,d_opt,m) for name,m in regions.items()}
        np.savez_compressed(out/'depth.npz',depth_m=depth)
        np.save(out/'plane_mask.npy',mask)
        view=cv2.applyColorMap(np.uint8(np.clip(depth,0,30)/30*255),cv2.COLORMAP_TURBO)
        view[mask]=(0,255,0)
        cv2.imwrite(str(out/'depth_plane.png'),view)
        evidence={'mode':'oracle_geometry','carla_version':session.client.get_server_version(),
            'map':world.get_map().name,'frame_id':image.frame,'timestamp_s':image.timestamp,
            'K':K.tolist(),'T_carla_world_from_opt':T.tolist(),
            'independent_survey_world_m':survey.tolist(),'survey_plane_max_residual_m':residual,
            'plane_normal_opt':n_opt.tolist(),'plane_offset_m':float(d_opt),
            'region_samples':{key:int(m.sum()) for key,m in regions.items()},'plane_depth_mae_m':results,
            'samples':rows,'interpretation':'Compare optical_z/ray_range; collision geometry can differ from render mesh'}
        atomic_json(out/'report.json',evidence)
        print(json.dumps({k:v for k,v in evidence.items() if k not in ('samples',)},indent=2))


if __name__=='__main__':
    main()
