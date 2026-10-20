"""Serializable contracts; no CARLA Actor objects cross this boundary."""
from dataclasses import dataclass
from typing import Any
import hashlib
import json
import numpy as np

CAMERA_IDS = ('front', 'front_left', 'front_right', 'rear_left', 'rear_right', 'rear')
SENSOR_IDS = tuple(f'{c}/{kind}' for c in CAMERA_IDS for kind in ('rgb', 'depth'))
SCHEMA_VERSION = 'hsd-frames-v1'


@dataclass(frozen=True)
class SensorSample:
    sensor_id: str
    frame_id: int
    timestamp_s: float
    width: int
    height: int
    raw_bgra: bytes
    T_carla_world_from_camera: np.ndarray


@dataclass(frozen=True)
class FrameBundle:
    frame_id: int
    timestamp_s: float
    sensors: dict[str, SensorSample]
    ego: dict[str, Any]
    objects: list[dict[str, Any]]
    traffic_rules: dict | None = None


def calibration_hash(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_bundle(bundle, required=SENSOR_IDS):
    if not np.isfinite(bundle.timestamp_s):
        raise ValueError('Nonfinite frame timestamp')
    if set(bundle.sensors) != set(required):
        raise ValueError('Incomplete or unexpected sensor set')
    for name, sample in bundle.sensors.items():
        if sample.sensor_id != name or sample.frame_id != bundle.frame_id:
            raise ValueError('Mixed frame or sensor ID')
        if not np.isfinite(sample.timestamp_s) or abs(sample.timestamp_s - bundle.timestamp_s) > 1e-5:
            raise ValueError('Mixed simulation timestamps')
        if sample.width < 1 or sample.height < 1 or len(sample.raw_bgra) != sample.width * sample.height * 4:
            raise ValueError('Invalid image size')


def validate_learned_inputs(inputs):
    """Fail closed; caller must construct model inputs explicitly from RGB/state."""
    allowed = {'images', 'intrinsics', 'cam_to_ego', 'ego_pose', 'ego_state',
               'route_command', 'timestamps_s', 'valid_cameras'}
    extra = set(inputs) - allowed
    if extra:
        raise ValueError(f'Forbidden learned input fields: {sorted(extra)}')

    # Reject containers under an approved name: a nested ego_state must never
    # smuggle actor/depth/map labels past a top-level allowlist.
    for key, value in inputs.items():
        is_tensor = type(value).__module__.split('.')[0] == 'torch' and hasattr(value, 'dtype')
        if not (is_tensor or isinstance(value, (np.ndarray, int, float, bool, np.number))):
            raise TypeError(f'Learned inputs must be numeric arrays/tensors: {key}')
        if isinstance(value, np.ndarray) and value.dtype.kind not in 'bifu':
            raise TypeError(f'Non-numeric learned input: {key}')
