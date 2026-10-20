"""Deterministic group splits. Repeated routes must share one group_id."""
import hashlib
import numpy as np

SPLITS = ('train', 'val', 'test')


def assign_splits(scenes, *, seed, val_fraction, test_maps):
    if not 0 < val_fraction < 1 or not test_maps:
        raise ValueError('Require a validation fraction in (0,1) and held-out test maps')
    assignments, groups = {}, {}
    for scene in sorted(scenes, key=lambda s: s['scene_id']):
        group = scene['group_id']
        if not group:
            raise ValueError('An explicit route/episode group_id is required')
        held_out = scene['map'] in test_maps
        fraction = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest(), 16) / 2**256
        split = scene.get('split', 'test' if held_out else ('val' if fraction < val_fraction else 'train'))
        if split not in SPLITS or (split == 'test') != held_out:
            raise ValueError('Held-out maps must be test-only, other maps train/val only')
        if group in groups and groups[group] != split:
            raise ValueError(f'Group crosses splits: {group}')
        groups[group] = split
        assignments[scene['scene_id']] = split
    if set(assignments.values()) != set(SPLITS):
        raise ValueError('Need nonempty train, val, and unseen-map test scenes')
    return assignments


def verify_spatial_separation(scenes, minimum_m):
    """Conservative XY guard on all ego samples, not a guarantee of visual novelty."""
    if not np.isfinite(minimum_m) or minimum_m <= 0:
        raise ValueError('Spatial separation must be positive')
    measurements = []
    for i, first in enumerate(scenes):
        for second in scenes[i+1:]:
            if first['map'] != second['map'] or first['split'] == second['split']:
                continue
            a, b = np.asarray(first['positions_xy']), np.asarray(second['positions_xy'])
            distance = min(float(np.linalg.norm(a[j:j+256,None]-b[None], axis=-1).min())
                           for j in range(0, len(a), 256))
            if distance < minimum_m:
                raise ValueError(f"Nearby routes cross splits: {first['scene_id']}/{second['scene_id']} ({distance:.2f}m)")
            measurements.append({'first': first['scene_id'], 'second': second['scene_id'], 'min_distance_m': distance})
    return measurements
