"""Explain radius-filter membership differences from existing private metadata.

No inference, decoding, evaluation, or prediction changes. Public report contains
only aggregate distances/counts; annotation and prediction identities stay private.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = read('configs/m8j9_retry1_protocol.json')
    private = Path(cfg['private_output'])
    data = Path(cfg['dataset_root']) / cfg['version']
    poses = {x['token']: x for x in read(private / 'export_provenance.json')['poses']}
    annotations = {x['token']: x for x in read(data / 'sample_annotation.json')}
    instances = {x['token']: x for x in read(data / 'instance.json')}
    categories = {x['token']: x['name'] for x in read(data / 'category.json')}
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.utils import category_to_detection_name
    from pyquaternion import Quaternion
    import numpy as np
    ranges = config_factory(cfg['official_detection_config']).class_range
    counts = Counter()
    beyond = []
    for x in read(private / 'GT_membership_comparison.json'):
        if not (x['project_retained'] and not x['official_retained']):
            continue
        ann = annotations[x['annotation_token']]
        pose = poses[x['sample_token']]
        name = category_to_detection_name(categories[instances[ann['instance_token']]['category_token']])
        distance = math.hypot(ann['translation'][0] - pose['ego2global_translation'][0],
                              ann['translation'][1] - pose['ego2global_translation'][1])
        assert distance >= ranges[name]
        assert ann['num_lidar_pts'] + ann['num_radar_pts'] > 0
        counts[name] += 1
        beyond.append(distance - ranges[name])
    removed = []
    for token, items in read(private / 'export/results_nusc.json')['results'].items():
        pose = poses[token]
        rotation = Quaternion(pose['ego2global_rotation']).rotation_matrix
        for box in items:
            delta = np.asarray(box['translation']) - pose['ego2global_translation']
            global_xy = float(np.linalg.norm(delta[:2]))
            ego_xy = float(np.linalg.norm((rotation.T @ delta)[:2]))
            radius = ranges[box['detection_name']]
            if global_xy >= radius:
                assert ego_xy <= radius
                removed.append((global_xy - radius, radius - ego_xy))
    assert sum(counts.values()) == 166 and len(removed) == 1
    report = dict(
        project_only_GT_count=sum(counts.values()),
        all_project_only_fail_official_class_radius=True,
        all_have_nonzero_point_support=True,
        project_only_by_class=dict(counts),
        minimum_GT_distance_beyond_radius_m=min(beyond),
        prediction_radius_exclusions=len(removed),
        maximum_prediction_global_xy_over_radius_m=max(x[0] for x in removed),
        minimum_prediction_ego_xy_inside_radius_m=min(x[1] for x in removed),
        explanation='Exporter ego XY versus evaluator global XY relative to ego differ with roll/pitch; both official rules retained',
        no_prediction_modification=True,
        command='python -m tools.m8j9_radius_audit --check-only')
    target = Path('reports/m8j9_radius_difference.json')
    if args.check_only:
        assert read(target) == report
    else:
        with target.open('x') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
