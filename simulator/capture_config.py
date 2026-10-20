"""Versioned defaults for the existing collector; explicit CLI options override."""
from pathlib import Path
import math
import yaml

_FIELDS = {'render_every_tick': bool, 'config': str, 'cameras': str, 'camera_profile': str,
           'frames': int, 'warmup_ticks': int, 'tick_wall_delay': (int, float),
           'render_quality': str, 'motion': str}


def load_capture_defaults(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 'hsd-capture-defaults-v1':
        raise ValueError('Invalid capture defaults schema')
    if set(data) - {'schema_version', 'collector', 'network_input', 'validation_scope'}:
        raise ValueError('Unknown capture defaults fields')
    defaults = dict(data['collector'])
    if set(defaults) - _FIELDS.keys():
        raise ValueError('Unknown collector option in capture defaults')
    for key, value in defaults.items():
        if (isinstance(value, bool) and _FIELDS[key] is not bool) or not isinstance(value, _FIELDS[key]):
            raise ValueError(f'Invalid capture default type: {key}')
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f'Invalid capture default number: {key}')
    for key in ('config', 'cameras', 'camera_profile'):
        if key in defaults:
            resolved = (path.parent / defaults[key]).resolve()
            if not resolved.is_file():
                raise ValueError(f'Missing capture defaults dependency: {resolved}')
            defaults[key] = str(resolved)
    if defaults.get('frames', 1) < 1:
        raise ValueError('Capture frame count must be positive')
    if defaults.get('render_quality', 'Epic') not in ('Epic', 'Low'):
        raise ValueError('Unknown render quality')
    if defaults.get('motion', 'stationary') not in ('stationary', 'straight', 'autopilot'):
        raise ValueError('Unknown capture motion')
    return defaults
