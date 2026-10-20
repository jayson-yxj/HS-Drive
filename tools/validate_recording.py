"""Checksum, calibration, geometry and cadence checks on every committed frame."""
import argparse
import json
import numpy as np
from datasets.carla_dataset import CarlaDataset
from datasets.schema import CAMERA_IDS
from simulator.recorder import atomic_json


def validate(scene, *, require_six=True):
    ds = CarlaDataset(scene)
    if not len(ds):
        raise ValueError('No committed frames')
    if require_six and set(ds.calibration['cameras']) != set(CAMERA_IDS):
        raise ValueError('M1 requires six cameras')
    ids, times, rgb_nonempty = [], [], True
    for i in range(len(ds)):
        sample, images, depths, _ = ds.read(i)
        ids.append(sample['frame_id'])
        times.append(sample['timestamp_s'])
        for name, image in images.items():
            camera = ds.calibration['cameras'][name]
            expected_shape = (camera['height'], camera['width'])
            if image.shape != (*expected_shape,3) or depths[name].shape != expected_shape:
                raise ValueError('Image/calibration dimensions mismatch')
            if not np.isfinite(depths[name]).all():
                raise ValueError('Nonfinite stored depth')
            rgb_nonempty &= bool(image.max() > 0 and image.std() > 0)
            expected_pose = (np.asarray(sample['ego']['T_carla_world_from_ego']) @
                             np.asarray(camera['T_carla_ego_from_camera']))
            for sensor in sample['cameras'][name]['sensors'].values():
                if not np.allclose(sensor['T_carla_world_from_camera'], expected_pose, atol=2e-3,rtol=0):
                    raise ValueError('Sensor pose mismatch')
    if len(ids)>1 and (np.any(np.diff(ids)<=0) or np.any(np.diff(times)<=0)):
        raise ValueError('Nonmonotonic frame/time sequence')
    period = ds.manifest['camera_config']['sensor_tick_s']
    duration = times[-1]-times[0]
    expected = round(duration/period)+1
    cadence_errors = int(np.count_nonzero(~np.isclose(np.diff(times),period,atol=1e-5)))
    sync = ds.manifest.get('synchronization',{})
    bad = sum(sync.get(key,0) for key in ('queue_overflow','invalid_frame','dropped_frame','unknown_sensor'))
    clean = rgb_nonempty and cadence_errors==0 and bad==0 and ds.manifest['status']=='complete'
    return {'scene':str(ds.root),'frames_verified':len(ds),'capture_duration_s':duration,
            'first_frame':ids[0],'last_frame':ids[-1],
            'interval_completeness':len(ids)/expected,'cadence_gaps':cadence_errors,
            'rgb_nonempty_nonconstant':rgb_nonempty,'synchronization':sync,
            'recording_integrity_checks_passed':bool(clean),
            'm1_automated_checks_passed':bool(clean and require_six and
                ds.manifest.get('carla_version') == '0.9.15' and duration >= 29.9),
            'm2_scene_alignment_verified':False,
            'remaining_manual_gates':['camera body occlusion','box/depth alignment','known-plane Z/range experiment']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--front-only',action='store_true')
    a=p.parse_args()
    result=validate(a.scene,require_six=not a.front_only)
    atomic_json(a.output,result)
    print(json.dumps(result,indent=2))
    return 0 if result['recording_integrity_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
