"""Single-writer transactional frames: files first, per-frame JSON commit last.

A crash can leave orphan image files, but readers only consume committed JSONs.
An existing scene is never overwritten; start a new run after simulator restart.
"""
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
from datasets.schema import SCHEMA_VERSION, calibration_hash, validate_bundle
from geometry.carla_depth import decode_rgb_bgra, decode_carla_depth_bgra


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recorder:
    def __init__(self, root, calibration, metadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.calibration = calibration
        self.calibration_id = calibration_hash(calibration)
        self.names = tuple(calibration['cameras'])
        self.required = tuple(f'{c}/{k}' for c in self.names for k in ('rgb', 'depth'))
        self.metadata = {**metadata, 'schema_version': SCHEMA_VERSION, 'calibration_id': self.calibration_id,
                         'mode': 'oracle_geometry', 'status': 'recording', 'frames_committed': 0}
        atomic_json(self.root / 'calibration.json', calibration)
        atomic_json(self.root / 'manifest.json', self.metadata)
        self.committed = 0

    def event(self, event):
        with (self.root / 'diagnostics.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')

    def write(self, bundle):
        validate_bundle(bundle, self.required)
        stem = f'{bundle.frame_id:08d}'
        commit = self.root / 'frames' / f'{stem}.json'
        if commit.exists():
            raise FileExistsError(f'Frame already committed: {stem}')
        cameras, checksums = {}, {}
        for name in self.names:
            meta = {'frame_id': bundle.frame_id, 'sensors': {}}
            expected = (np.asarray(bundle.ego['T_carla_world_from_ego']) @
                        np.asarray(self.calibration['cameras'][name]['T_carla_ego_from_camera']))
            for kind in ('rgb', 'depth'):
                sample = bundle.sensors[f'{name}/{kind}']
                if not np.allclose(sample.T_carla_world_from_camera, expected, atol=2e-3, rtol=0):
                    raise ValueError(f'Extrinsic/image.transform mismatch: {name}/{kind}')
                entry = self.calibration['cameras'][name]
                if (sample.width, sample.height) != (entry['width'], entry['height']):
                    raise ValueError('Calibration/image dimensions mismatch')
                ext = 'png' if kind == 'rgb' else 'npz'
                rel = f'{kind}/{name}/{stem}.{ext}'
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if kind == 'rgb':
                    rgb = decode_rgb_bgra(sample.raw_bgra, sample.height, sample.width)
                    if not cv2.imwrite(str(path), rgb[..., ::-1]):
                        raise OSError(f'Failed to write {path}')
                else:
                    depth = decode_carla_depth_bgra(sample.raw_bgra, sample.height, sample.width)
                    np.savez_compressed(path, depth_m=depth)
                checksums[rel] = file_hash(path)
                meta[kind] = rel
                meta['sensors'][kind] = {'timestamp_s': sample.timestamp_s,
                    'frame_id': sample.frame_id, 'T_carla_world_from_camera': sample.T_carla_world_from_camera.tolist()}
            cameras[name] = meta
        objects_rel = f'labels/objects/{stem}.json'
        atomic_json(self.root / objects_rel, bundle.objects)
        checksums[objects_rel] = file_hash(self.root / objects_rel)
        labels={'objects':objects_rel}
        if bundle.traffic_rules is not None:
            from simulator.traffic_rules import validate_traffic
            validate_traffic(bundle.traffic_rules,bundle.frame_id,bundle.timestamp_s)
            rel=f'labels/traffic_rules/{stem}.json';atomic_json(self.root/rel,bundle.traffic_rules)
            checksums[rel]=file_hash(self.root/rel);labels['traffic_rules']=rel
        sample = {'schema_version': SCHEMA_VERSION, 'scene_id': self.root.name,
                  'frame_id': bundle.frame_id, 'timestamp_s': bundle.timestamp_s,
                  'carla_version': self.metadata['carla_version'], 'calibration_id': self.calibration_id,
                  'mode': 'training-supervision-available', 'cameras': cameras, 'ego': bundle.ego,
                  'labels': labels, 'sha256': checksums}
        atomic_json(commit, sample)
        self.committed += 1
        return sample

    def close(self, status, stats):
        self.metadata.update(status=status, frames_committed=self.committed, synchronization=dict(stats))
        atomic_json(self.root / 'manifest.json', self.metadata)
