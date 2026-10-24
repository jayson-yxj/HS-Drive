"""RGB-only PETR adapter. No evaluator, dataset index or GT imports.

The historical framework's `lidar2img` is a projection from HS ego, not a
LiDAR input. Output free space and control authorization are deliberately absent.
"""
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

ORDER = ['front', 'front_left', 'front_right', 'rear_left', 'rear_right', 'rear']
OFFICIAL = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
ANGLES = [0, 60, -60, 120, -120, 180]
A = np.array([[2.2, 0, 0], [0, 2.2, -332], [0, 0, 1.]])

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def old_boxes_to_hs(boxes):
    """Old mmdet3d 0.17: bottom xyz,w,l,h,clockwise yaw,vx,vy -> HS."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
    out = b.copy()
    out[:, 2] += b[:, 5] / 2
    out[:, 3:6] = b[:, [4, 3, 5]]
    out[:, 6] = (-b[:, 6] - np.pi / 2 + np.pi) % (2*np.pi) - np.pi
    return out  # velocity already expressed in the reference x/y axes

def corners_hs(box):
    b = np.asarray(box)
    offsets = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]]) * b[3:6]
    c, s = np.cos(b[6]), np.sin(b[6])
    return offsets @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + b[:3]

class PETRCarlaOfflineAdapter:
    packet_keys = {'schema','token','camera_order','frame_id','timestamp_s','image_size','cameras'}
    camera_keys = {'name','path','sha256','frame_id','timestamp_s','K','T_cam_to_ego'}

    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve()

    def validate(self, p):
        if set(p) != self.packet_keys or p['schema'] != 'm8b.six_rgb.v1':
            raise ValueError('Packet whitelist violation')
        if not isinstance(p['frame_id'], int) or p['frame_id'] < 0 or not np.isfinite(p['timestamp_s']):
            raise ValueError('Invalid source time')
        if p['camera_order'] != ORDER or p['image_size'] != [384,640] or len(p['cameras']) != 6:
            raise ValueError('Camera order/shape violation')
        paths, hashes = set(), set()
        for name, angle, c in zip(ORDER, ANGLES, p['cameras']):
            if set(c) != self.camera_keys or c['name'] != name:
                raise ValueError('Camera whitelist/order violation')
            if c['frame_id'] != p['frame_id'] or c['timestamp_s'] != p['timestamp_s']:
                raise ValueError('Unsynchronized input')
            path = Path(c['path']).resolve()
            if self.bundle not in path.parents or path in paths or c['sha256'] in hashes:
                raise ValueError('Unsafe or duplicate image')
            paths.add(path); hashes.add(c['sha256'])
            K, T = np.array(c['K']), np.array(c['T_cam_to_ego'])
            if K.shape != (3,3) or T.shape != (4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
                raise ValueError('Invalid calibration')
            if not np.allclose(T[3], [0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3], np.eye(3), atol=2e-6) or abs(np.linalg.det(T[:3,:3])-1)>2e-6:
                raise ValueError('Nonrigid extrinsic')
            heading = np.rad2deg(np.arctan2(T[1,2],T[0,2]))
            if abs((heading-angle+180)%360-180) > 5:
                raise ValueError('Extrinsic/image named camera mismatch')
            if min(K[0,0], K[1,1]) <= 0 or not np.allclose(K[2], [0,0,1]):
                raise ValueError('Invalid intrinsic')
        return p

    def prepare(self, p):
        # Import only the frozen official implementation, after process isolation.
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage, NormalizeMultiviewImage, PadMultiViewImage
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        t=time.perf_counter(); self.validate(p); times={}
        images=[]
        for c in p['cameras']:
            blob=Path(c['path']).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=c['sha256']: raise ValueError('Image SHA mismatch')
            rgb=np.asarray(Image.open(io.BytesIO(blob)).convert('RGB'))
            if rgb.shape != (384,640,3): raise ValueError('Image shape mismatch')
            images.append(rgb)
        times['read_six_rgb_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter(); bgr=[im[...,::-1].copy() for im in images]
        times['rgb_to_bgr_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter()
        intr=[]; ext=[]
        for c in p['cameras']:
            k=np.eye(4); k[:3,:3]=c['K']; intr.append(k)
            ext.append(np.linalg.inv(np.asarray(c['T_cam_to_ego'])).T)
        tr=ResizeCropFlipImage(dict(H=384,W=640,final_dim=(512,1408),resize_lim=(.8,1.),bot_pct_lim=(0.,0.),rot_lim=(0.,0.),rand_flip=True),training=False)
        data=tr(dict(img=bgr,intrinsics=intr,extrinsics=ext))
        assert tr._sample_augmentation() == (2.2,(1408,844),(0,332,1408,844),False,0.0)
        for c,k in zip(p['cameras'],data['intrinsics']):
            assert np.allclose(k[:3,:3], A@c['K'], atol=1e-4)
        times['resize_crop_calibration_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter()
        data=NormalizeMultiviewImage(mean=[103.530,116.280,123.675],std=[1.,1.,1.],to_rgb=False)(data)
        data=PadMultiViewImage(size_divisor=32)(data)
        times['normalize_pad_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter(); cpu=np.stack(data['img']).transpose(0,3,1,2).copy()[None]
        assert cpu.shape==(1,6,3,512,1408) and cpu.dtype==np.float32
        meta=dict(img_shape=[(512,1408,3)]*6,pad_shape=[(512,1408,3)]*6,ori_shape=[(384,640,3)]*6,
            lidar2img=data['lidar2img'],box_type_3d=LiDARInstance3DBoxes,scale_factor=1.,flip=False)
        audit=dict(camera_mapping=dict(zip(ORDER,OFFICIAL)),A=A.tolist(),resize_dims=[1408,844],crop=[0,332,1408,844],
            intrinsic_aug=[k.tolist() for k in data['intrinsics']],lidar2img=[k.tolist() for k in data['lidar2img']],
            calibration_sha256=digest([{k:c[k] for k in ['name','K','T_cam_to_ego']} for c in p['cameras']]),
            augmentation_sha256=digest(dict(A=A.tolist(),resize=[1408,844],crop=[0,332,1408,844],mean=[103.530,116.280,123.675],std=[1,1,1],to_rgb=False)),metadata_fields=sorted(meta))
        times['assemble_ms']=1000*(time.perf_counter()-t)
        return cpu,meta,audit,times
