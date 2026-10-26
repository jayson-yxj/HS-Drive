"""Standalone RGB whitelist and bounded cadence; never reads evaluator data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2

ORDER = ['front','front_left','front_right','rear_left','rear_right','rear']
MODES = ['rgb','depth','instance','semantic']
REQUIRED = tuple(f'{name}/{kind}' for name in ORDER for kind in MODES)


def schedule_frames(first_sensor_frame, last_init_frame, offsets):
    if len(offsets) != 12 or offsets != sorted(set(offsets)) or min(offsets) < 0:
        raise ValueError('Twelve frozen increasing offsets required')
    first = last_init_frame + 1
    first += (first_sensor_frame-first) % 2
    return [first + 2*k for k in offsets]


class RGBInferenceLoader:
    packet_keys = {'schema','source','calibration_sha256','camera_order','image_size',
                   'frame_id','timestamp_s','cameras','image_transform'}
    camera_keys = {'name','path','sha256','K','T_cam_to_ego','frame_id','timestamp_s'}

    def __init__(self, rgb_root):
        self.root = Path(rgb_root).resolve()

    def path(self, relative):
        p = self.root / relative
        # Reject symlink traversal even when it resolves back into the root.
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Path outside RGB root')
        for part in [p, *p.parents]:
            if part == self.root:
                break
            if part.is_symlink():
                raise ValueError('Symlink forbidden')
        if self.root not in p.resolve().parents:
            raise ValueError('Path outside RGB root')
        return p

    def load(self, packet_name):
        p = json.loads(self.path(packet_name).read_text())
        if set(p) != self.packet_keys or p['schema'] != 'RGBInferencePacket.m8i1.v1':
            raise ValueError('Packet whitelist')
        if p['camera_order'] != ORDER or p['image_size'] != [384,640] or len(p['cameras']) != 6:
            raise ValueError('Rig mismatch')
        if type(p['frame_id']) is not int or p['frame_id'] < 0 or not np.isfinite(p['timestamp_s']):
            raise ValueError('Invalid source time')
        if p['source'] != 'CARLA0.9.15_RGBOnly' or p['image_transform'] != 'Identity':
            raise ValueError('Unexpected source or augmentation')
        if len(p['calibration_sha256']) != 64:
            raise ValueError('Missing calibration hash')
        files, images = set(), []
        for name, c in zip(ORDER,p['cameras']):
            if set(c) != self.camera_keys or c['name'] != name:
                raise ValueError('Camera whitelist')
            if c['frame_id'] != p['frame_id'] or c['timestamp_s'] != p['timestamp_s']:
                raise ValueError('Mixed source time')
            K,T = np.asarray(c['K']),np.asarray(c['T_cam_to_ego'])
            if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
                raise ValueError('Invalid calibration')
            if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6):
                raise ValueError('Nonrigid calibration')
            f = self.path(c['path'])
            if f in files or f.suffix != '.png':
                raise ValueError('Duplicate/nonRGB payload')
            files.add(f)
            raw = f.read_bytes()
            if hashlib.sha256(raw).hexdigest() != c['sha256']:
                raise ValueError('Payload hash')
            img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
            if img is None or img.shape!=(384,640,3):
                raise ValueError('Image contract')
            images.append(img[:,:,::-1].copy())
        return np.stack(images)
