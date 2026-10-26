"""Independent single-frame RGB-only packet reader; never opens annotation files."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK')
HS_ORDER = ('front', 'front_left', 'front_right', 'rear_left', 'rear_right', 'rear')
VIEW = dict(zip(CAMERAS, ('front', 'front_left', 'front_right',
                         'back_left', 'back_right', 'back')))
TOP_FIELDS = {'schema', 'source', 'revision', 'clip', 'ordinal', 'source_frame',
              'timestamp_s', 'temporal_valid', 'camera_order', 'cameras'}
CAM_FIELDS = {'name', 'path', 'sha256', 'width', 'height', 'K', 'T_hs_ego_from_optical'}


def contained(root, relative):
    p = Path(relative)
    if p.is_absolute():
        raise ValueError('Absolute path forbidden')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Path outside RGB-only root')
    return resolved


def load_rgb_packet(root, packet_name):
    root = Path(root).resolve()
    p = contained(root, packet_name)
    if p.stat().st_size > 128 * 1024:
        raise ValueError('Packet too large')
    packet = json.loads(p.read_text())
    if set(packet) != TOP_FIELDS or packet['schema'] != 'm8j3-rgb-only-v1':
        raise ValueError('Packet field allowlist')
    if (packet['camera_order'] != list(CAMERAS) or len(packet['cameras']) != 6
            or packet['source_frame'] is not None or packet['timestamp_s'] is not None
            or packet['temporal_valid'] is not False):
        raise ValueError('Unknown raw time must remain unknown')
    images, ks, ts = [], [], []
    for name, c in zip(CAMERAS, packet['cameras']):
        if set(c) != CAM_FIELDS or c['name'] != name:
            raise ValueError('Camera field allowlist/order')
        image_path = contained(root, c['path'])
        raw = image_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != c['sha256']:
            raise ValueError('RGB checksum mismatch')
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        k = np.asarray(c['K'], np.float64)
        t = np.asarray(c['T_hs_ego_from_optical'], np.float64)
        if im is None or im.shape != (c['height'], c['width'], 3):
            raise ValueError('RGB dimensions mismatch')
        if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
            raise ValueError('Calibration shape/finite mismatch')
        if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
                or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-5):
            raise ValueError('Invalid pinhole/rigid calibration')
        images.append(im[..., ::-1].copy())
        ks.append(k); ts.append(t)
    return dict(images=np.stack(images), intrinsics=np.stack(ks),
                cam_to_ego=np.stack(ts), timestamps=None,
                temporal_valid=False, camera_order=HS_ORDER)
