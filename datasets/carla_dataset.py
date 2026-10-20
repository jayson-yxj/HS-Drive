"""Streaming replay of committed M1 frames, optionally verifying every checksum."""
import json
from pathlib import Path
import cv2
import numpy as np
from .schema import SCHEMA_VERSION, calibration_hash
from simulator.recorder import file_hash


class CarlaDataset:
    def __init__(self, scene):
        self.root = Path(scene)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.calibration = json.loads((self.root / 'calibration.json').read_text())
        if calibration_hash(self.calibration) != self.manifest['calibration_id']:
            raise ValueError('Calibration hash mismatch')
        self.frames = sorted((self.root / 'frames').glob('*.json'))

    def __len__(self):
        return len(self.frames)

    def read(self, index, *, verify=True):
        sample = json.loads(self.frames[index].read_text())
        if sample['schema_version'] != SCHEMA_VERSION or sample['calibration_id'] != self.manifest['calibration_id']:
            raise ValueError('Schema/calibration mismatch')
        if set(sample['cameras']) != set(self.calibration['cameras']):
            raise ValueError('Camera set mismatch')
        if verify:
            for path, expected in sample['sha256'].items():
                if file_hash(self.root / path) != expected:
                    raise ValueError(f'Checksum mismatch: {path}')
        images, depths = {}, {}
        for name, entry in sample['cameras'].items():
            if entry['frame_id'] != sample['frame_id']:
                raise ValueError('Mixed frame IDs')
            for sensor in entry['sensors'].values():
                if sensor['frame_id'] != sample['frame_id'] or abs(sensor['timestamp_s']-sample['timestamp_s']) > 1e-5:
                    raise ValueError('Mixed sensor metadata')
            image = cv2.imread(str(self.root / entry['rgb']))
            if image is None:
                raise ValueError('Unreadable RGB image')
            images[name] = image[..., ::-1].copy()
            with np.load(self.root / entry['depth'], allow_pickle=False) as data:
                depths[name] = data['depth_m']
        objects = json.loads((self.root / sample['labels']['objects']).read_text())
        return sample, images, depths, objects
