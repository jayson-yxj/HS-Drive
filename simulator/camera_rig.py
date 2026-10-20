"""Six paired RGB/depth cameras; mounting values come from baseline YAML."""
from datasets.schema import CAMERA_IDS
from geometry.reference import intrinsics_from_hfov, camera_mount_to_hs_from_optical
import numpy as np
from .imaging import rgb_attributes


def spawn_rig(session, ego, config, synchronizer, camera_ids=CAMERA_IDS):
    import carla
    calibration = {'schema_version': 'hsd-calibration-v1',
                   'coordinate_frame': 'hs_ego', 'units': 'm',
                   'pixel_centers': 'integer; principal point ((W-1)/2,(H-1)/2)',
                   'depth_convention': 'unverified', 'cameras': {}}
    # Validate before spawning actors, including unsupported geometry overrides.
    rgb_settings = rgb_attributes(config)
    calibration['imaging_profile_id'] = config.get('imaging_profile_id', 'geometry-baseline-v1')
    K = intrinsics_from_hfov(config['image_width'], config['image_height'], config['horizontal_fov_deg'])
    for name in camera_ids:
        cfg = config['cameras'][name]
        mount = carla.Transform(carla.Location(*cfg['xyz_m']),
                                carla.Rotation(pitch=cfg['pitch_deg'], yaw=cfg['yaw_deg']))
        entry = {'K': K.tolist(), 'width': config['image_width'], 'height': config['image_height'],
                 'T_carla_ego_from_camera': mount.get_matrix(),
                 'T_hs_ego_from_opt': camera_mount_to_hs_from_optical(np.asarray(mount.get_matrix())).tolist(),
                 'sensors': {}}
        for kind in ('rgb', 'depth'):
            bp = session.world.get_blueprint_library().find(f'sensor.camera.{kind}')
            attrs = {'image_size_x': config['image_width'], 'image_size_y': config['image_height'],
                     'fov': config['horizontal_fov_deg'], 'sensor_tick': config.get('render_sensor_tick_s', config['sensor_tick_s']),
                     **config['distortion']}
            if kind == 'rgb':
                attrs.update(rgb_settings)
            for key, value in attrs.items():
                if not bp.has_attribute(key):
                    raise ValueError(f'CARLA blueprint lacks requested attribute: {kind}/{key}')
                bp.set_attribute(key, str(value))
            sensor = session.spawn(bp, mount, parent=ego)
            entry['sensors'][kind] = dict(sensor.attributes)
            sensor.listen(synchronizer.callback(f'{name}/{kind}'))
        calibration['cameras'][name] = entry
    return calibration
