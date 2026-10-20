"""Offline synthetic integration fixture, explicitly NOT a CARLA rendering test."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from datasets.schema import CAMERA_IDS, SensorSample, FrameBundle
from geometry.reference import intrinsics_from_hfov, camera_mount_to_hs_from_optical
from simulator.recorder import Recorder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    import carla  # Only Transform math, no server connection.
    cfg=yaml.safe_load(Path('configs/cameras.yaml').read_text())
    w,h=cfg['image_width'],cfg['image_height']
    K=intrinsics_from_hfov(w,h,cfg['horizontal_fov_deg'])
    cameras,sensors={},{}
    v,u=np.indices((h,w))
    for index,name in enumerate(CAMERA_IDS):
        c=cfg['cameras'][name]
        mount=carla.Transform(carla.Location(*c['xyz_m']),carla.Rotation(yaw=c['yaw_deg'],pitch=c['pitch_deg']))
        T=np.asarray(mount.get_matrix())
        hs=camera_mount_to_hs_from_optical(T)
        rays=np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)],axis=-1) @ hs[:3,:3].T
        # Independent analytic room: walls x/y=+-15 m, floor z=-1 m.
        depth=np.full((h,w),1000.)
        for axis,planes in [(0,[-15,15]),(1,[-15,15]),(2,[-1])]:
            for plane in planes:
                with np.errstate(divide='ignore',invalid='ignore'):
                    distance=(plane-hs[axis,3])/rays[...,axis]
                depth=np.minimum(depth,np.where(distance>0,distance,1000))
        enc=np.rint(depth/1000*16777215).astype(np.uint32)
        depth_raw=np.stack([(enc>>16)&255,(enc>>8)&255,enc&255,np.full_like(enc,255)],axis=-1).astype(np.uint8)
        rgb_raw=np.stack([u%256,v%256,np.full_like(u,40+index*30),np.full_like(u,255)],axis=-1).astype(np.uint8)
        cameras[name]={'K':K.tolist(),'width':w,'height':h,'T_carla_ego_from_camera':T.tolist(),
                       'T_hs_ego_from_opt':hs.tolist(),'sensors':{}}
        for kind,raw in [('rgb',rgb_raw),('depth',depth_raw)]:
            key=f'{name}/{kind}'
            sensors[key]=SensorSample(key,2,.1,w,h,raw.tobytes(),T)
    cal={'schema_version':'hsd-calibration-v1','cameras':cameras,'depth_convention':'synthetic_optical_z'}
    rec=Recorder(a.output,cal,{'carla_version':'synthetic-not-server','source':'synthetic analytic room',
        'camera_config':cfg,'depth_convention':'synthetic_optical_z'})
    rec.write(FrameBundle(2,.1,sensors,{'T_carla_world_from_ego':np.eye(4).tolist(),
                                     'T_world_from_ego':np.eye(4).tolist()},[]))
    rec.close('complete',{'complete':1})
    print(rec.root)


if __name__=='__main__':
    main()
