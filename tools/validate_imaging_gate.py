"""Imaging engineering gate for a specific profile and lighting scope."""
import argparse
import json
import math
from pathlib import Path
import yaml
from datasets.schema import CAMERA_IDS
from simulator.recorder import atomic_json, file_hash
from tools.validate_recording import validate


def evaluate_photometry(per_camera, thresholds):
    failures = []
    if set(per_camera) != set(CAMERA_IDS):
        failures.append('camera_set')
    for name, metrics in per_camera.items():
        surface = metrics['depth_valid_surface']
        for metric, threshold in [
            ('near_black_fraction', 'max_surface_near_black_fraction'),
            ('near_white_fraction', 'max_surface_near_white_fraction'),
        ]:
            value = surface[metric]
            if not math.isfinite(value) or not 0 <= value <= thresholds[threshold]:
                failures.append(name + '/' + metric)
        value = metrics['static_temporal_rgb_mae_8bit']
        if not math.isfinite(value) or not 0 <= value <= thresholds['max_static_rgb_mae_8bit']:
            failures.append(name + '/static_temporal_rgb_mae_8bit')
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-id', required=True)
    p.add_argument('--reports', nargs='+', required=True)
    p.add_argument('--stability-scene', required=True)
    p.add_argument('--geometry-report', required=True)
    p.add_argument('--motion-report', required=True)
    p.add_argument('--depth-report', required=True)
    p.add_argument('--thresholds', default='configs/imaging/acceptance.yaml')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    limits = yaml.safe_load(Path(a.thresholds).read_text())
    checks, failures, weather_seen, evidence = {}, {}, set(), set()

    def scene_manifest(scene):
        root = Path(scene)
        evidence.update(str(root / filename) for filename in ('manifest.json', 'calibration.json'))
        return json.loads((root / 'manifest.json').read_text())

    camera_configs = []
    for path in a.reports:
        report = json.loads(Path(path).read_text())
        weather = report['weather']
        if weather in weather_seen or weather not in ('noon', 'sunset'):
            raise ValueError('Supply unique noon and sunset reports')
        weather_seen.add(weather)
        selected = [entry for entry in report['results'] if entry['profile_id'] == a.profile_id]
        if len(selected) != 1:
            raise ValueError('Expected exactly one matching profile per report')
        entry = selected[0]
        failure = evaluate_photometry(entry['per_camera'], limits['daytime'])
        checks[weather + '_photometry'] = not failure
        checks[weather + '_integrity'] = entry['integrity']['recording_integrity_checks_passed']
        manifest = scene_manifest(entry['integrity']['scene'])
        camera_configs.append(manifest['camera_config'])
        checks[weather + '_profile'] = manifest['camera_config'].get('imaging_profile_id') == a.profile_id
        checks[weather + '_calibration'] = manifest['calibration_id'] == entry['calibration_id']
        failures[weather] = failure
    checks['noon_and_sunset_present'] = {'noon', 'sunset'} <= weather_seen
    integrity = validate(a.stability_scene)
    manifest = scene_manifest(a.stability_scene)
    geometry = json.loads(Path(a.geometry_report).read_text())
    motion = json.loads(Path(a.motion_report).read_text())
    motion_manifest = scene_manifest(motion['scene'])
    depth = json.loads(Path(a.depth_report).read_text())
    depth_manifest = scene_manifest(depth['second'])
    checks['identical_camera_configuration'] = all(config == manifest['camera_config'] for config in
        camera_configs + [motion_manifest['camera_config'], depth_manifest['camera_config']])
    checks['same_stability_profile'] = manifest['camera_config'].get('imaging_profile_id') == a.profile_id
    checks['stability'] = (integrity['m1_automated_checks_passed'] and
                           integrity['frames_verified'] >= limits['sampling']['minimum_stability_frames'])
    checks['geometry'] = (geometry['cross_view_passed'] and
                          Path(geometry['scene']).resolve() == Path(a.stability_scene).resolve())
    checks['motion'] = (motion['motion_passed'] and motion['cross_view_passed'] and
                        motion_manifest['camera_config'].get('imaging_profile_id') == a.profile_id)
    checks['depth_photometry'] = depth['passed'] and depth_manifest['camera_config'].get('imaging_profile_id') == a.profile_id
    evidence.update(a.reports + [a.geometry_report, a.motion_report, a.depth_report, a.thresholds])
    result = {
        'schema_version': 'hsd-imaging-gate-v1', 'profile_id': a.profile_id,
        'scope': 'Town03 fixed rig at sun altitude 50deg/8deg plus low-speed geometry; daytime capture engineering screening only',
        'checks': checks, 'photometry_failures': failures, 'passed': all(checks.values()),
        'stability': integrity, 'motion_scene': motion['scene'],
        'capture_pacing_s': manifest['tick_wall_delay_s'],
        'evidence': {path: file_hash(path) for path in sorted(evidence)},
        'limitations': ['night excluded', 'no trained-model quality claim', 'no rain/fog validation',
                        'no real-time throughput claim', 'finite-scene validation does not guarantee zero drops in future collections'],
    }
    atomic_json(a.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
