"""Photometric/resolution profiles without changing six-camera mounts or FOV."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

RGB_ATTRIBUTES = frozenset(('enable_postprocess_effects','gamma','exposure_mode',
    'exposure_compensation','exposure_min_bright','exposure_max_bright',
    'exposure_speed_up','exposure_speed_down','shutter_speed','iso','fstop',
    'focal_distance','bloom_intensity','lens_flare_intensity','motion_blur_intensity',
    'motion_blur_max_distortion','motion_blur_min_object_screen_size','blur_amount',
    'blur_radius','chromatic_aberration_intensity','chromatic_aberration_offset'))


def rgb_attributes(config):
    attributes = {'enable_postprocess_effects': False, **config.get('rgb_attributes', {})}
    unknown = set(attributes) - RGB_ATTRIBUTES
    if unknown:
        raise ValueError(f'Unsupported RGB attributes (geometry overrides forbidden): {sorted(unknown)}')
    if not isinstance(attributes['enable_postprocess_effects'], bool):
        raise ValueError('enable_postprocess_effects must be a YAML boolean')
    for key, value in attributes.items():
        if key == 'enable_postprocess_effects':
            continue
        if key == 'exposure_mode':
            if value not in ('manual','histogram'):
                raise ValueError('Invalid exposure mode')
        elif not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError(f'Expected finite numeric attribute: {key}')
    return {key: str(value).lower() if isinstance(value,bool) else str(value)
            for key,value in attributes.items()}


def apply_profile(base, profile):
    allowed = {'schema_version','profile_id','image_width','image_height','rgb_attributes','notes'}
    if set(profile)-allowed or profile.get('schema_version') != 'hsd-imaging-v1':
        raise ValueError('Invalid imaging profile schema/fields')
    result = deepcopy(base)
    for key in ('image_width','image_height'):
        value=profile[key]
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Image dimensions must be positive integers')
        result[key]=value
    result['imaging_profile_id']=profile['profile_id']
    result['rgb_attributes']=deepcopy(profile['rgb_attributes'])
    rgb_attributes(result)
    return result


def load_camera_config(camera_path, profile_path=None):
    base=yaml.safe_load(Path(camera_path).read_text())
    return apply_profile(base,yaml.safe_load(Path(profile_path).read_text())) if profile_path else base
