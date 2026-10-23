"""Strict six-RGB replay packet; GT metadata is not part of inference API."""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import hashlib,json,math
import numpy as np
from datasets.schema import CAMERA_IDS
from datasets.m7a_contracts import Stamp,EGO

KEYS={'schema','sample_id','camera_order','frame_id','timestamp_s','valid_until_s',
      'cameras','intrinsics','cam_to_ego','ego_pose','ego_pose_source','image_size','source'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_packet(p,now,*,verify_files=True):
    if set(p)!=KEYS or p['schema']!='m7a.six_rgb.v1' or p['source']!='RGB_OfflineReplay':
        raise ValueError('unknown_fields_or_GT_in_input')
    Stamp(p['frame_id'],p['timestamp_s'],p['valid_until_s'],EGO).validate(now)
    if tuple(p['camera_order'])!=CAMERA_IDS or len(p['cameras'])!=6:
        raise ValueError('six_camera_order')
    if p['image_size']!=[384,640]:raise ValueError('image_shape_contract')
    if np.shape(p['intrinsics'])!=(6,3,3) or np.shape(p['cam_to_ego'])!=(6,4,4):
        raise ValueError('calibration_shape')
    for name,cam,K,T in zip(CAMERA_IDS,p['cameras'],p['intrinsics'],p['cam_to_ego']):
        if set(cam)!={'name','frame_id','timestamp_s','path','sha256'} or cam['name']!=name:
            raise ValueError('camera_schema')
        if cam['frame_id']!=p['frame_id'] or not math.isfinite(cam['timestamp_s']) or abs(cam['timestamp_s']-p['timestamp_s'])>1e-6:
            raise ValueError('camera_not_synchronized')
        k=np.array(K);t=np.array(T)
        if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
            raise ValueError('intrinsics')
        if not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or abs(np.linalg.det(t[:3,:3])-1)>1e-5:
            raise ValueError('extrinsics')
        if not Path(cam['path']).name.endswith('_'+name+'_rgb.png'):
            raise ValueError('RGB_file_only')
        if verify_files and digest(cam['path'])!=cam['sha256']:raise ValueError('RGB_SHA')
    pose=np.array(p['ego_pose'])
    if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5):
        raise ValueError('ego_pose')
    if p['ego_pose_source']!='CARLA_metadata_for_alignment_only_not_model_feature':
        raise ValueError('pose_provenance')
    return True


class ReplayQueue:
    """Offline emulation of bounded delivery; does not claim live scheduling."""
    def __init__(self,capacity=2):
        self.queue=deque();self.capacity=capacity;self.dropped=[];self.last_frame=-1
    def push(self,p,now):
        try:validate_packet(p,now,verify_files=False)
        except ValueError as e:
            self.dropped.append(str(e));return False
        if p['frame_id']<=self.last_frame:
            self.dropped.append('out_of_order');return False
        self.last_frame=p['frame_id']
        if len(self.queue)>=self.capacity:self.queue.popleft();self.dropped.append('queue_overflow')
        # Serialization removes caller aliasing and any live object references.
        self.queue.append(json.loads(json.dumps(p)));return True
    def pop(self,now):
        while self.queue:
            p=self.queue.popleft()
            try:validate_packet(p,now,verify_files=False);return p
            except ValueError as e:self.dropped.append(str(e))
        return None
