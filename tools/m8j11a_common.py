"""Small append-only helpers for the bounded temporal experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = Path('configs/m8j11a_protocol.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def compare_arrays(a, b, atol=1e-5, rtol=1e-5):
    assert set(a) == set(b), 'CacheFieldsDiffer'
    error = {}
    for k in a:
        assert a[k].shape == b[k].shape, 'CacheShapeDiffer:' + k
        assert np.isfinite(a[k]).all() and np.isfinite(b[k]).all()
        if a[k].dtype.kind in 'iu':
            np.testing.assert_array_equal(a[k], b[k])
        else:
            np.testing.assert_allclose(a[k], b[k], atol=atol, rtol=rtol)
        error[k] = float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) if a[k].size else 0.
    return error


def schedules(frames):
    """Input-only selection: no GT or prediction inspection."""
    assert len({f['sample_token'] for f in frames}) == len(frames)
    groups = {}
    for f in frames:
        groups.setdefault(f['scene_name'], []).append(f)
    result = {k: [] for k in ('T1', 'T2', 'T3')}
    scenes = []
    for scene, group in sorted(groups.items()):
        group = sorted(group, key=lambda f: (f['timestamp_us'], f['sample_token']))
        assert len({f['scene_token'] for f in group}) == 1
        assert all(b['timestamp_us'] > a['timestamp_us'] for a, b in zip(group, group[1:]))
        n = len(group); reset = n // 2
        scenes.append(dict(scene_name=scene, samples=n, reset_scene_index=reset,
                           sparse_scene_indices=list(range(0, n, 2))))
        for mode in result:
            previous = None
            for i, f in enumerate(group):
                if mode == 'T3' and i % 2:
                    continue
                reason = ('scene_start' if i == 0 else 'every_frame_cold' if mode == 'T1'
                          else 'mid_scene_reset' if mode == 'T2' and i == reset else None)
                row = {k: f[k] for k in ('index', 'sample_token', 'scene_name', 'scene_token', 'timestamp_us')}
                row.update(scene_index=i, scene_samples=n, reset_scene_index=reset,
                           reset_reason=reason, input_dt_s=None if previous is None else
                           (f['timestamp_us'] - previous['timestamp_us'])/1e6)
                result[mode].append(row); previous = f
    return result, scenes
