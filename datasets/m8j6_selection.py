"""Metadata-only, predeclared eight-keyframe selection. No model or downloads.

The returned manifest contains private dataset identifiers and must stay outside
the repository. Loading source JSON tables is not expanding the training set.
"""
import ast
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def official_splits(path):
    """Read installed frozen split lists without importing a newer devkit."""
    lists = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lists[target.id] = ast.literal_eval(node.value)
    required = ('train_detect', 'train_track', 'val', 'mini_train', 'mini_val')
    if not all(k in lists for k in required):
        raise ValueError('FrozenOfficialSplitSchemaChanged')
    lists['train'] = sorted(set(lists['train_detect']) | set(lists['train_track']))
    return lists


def select_samples(scenes, samples, splits, protocol):
    eligible = sorted(set(splits['mini_train']) & set(splits['train']))
    names = eligible[:2]
    if names != protocol['selection']['scenes']:
        raise ValueError('PredeclaredSceneSelectionMismatch')
    forbidden = set(splits['val']) | set(splits['mini_val'])
    if set(names) & forbidden:
        raise ValueError('ValidationSceneForbidden')
    by_name = {s['name']: s for s in scenes}
    if len({s['token'] for s in samples}) != len(samples):
        raise ValueError('DuplicateSampleToken')
    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError('MissingPredeclaredScene:' + name)
        seq = sorted((s for s in samples if s['scene_token'] == by_name[name]['token']),
                     key=lambda s: (s['timestamp'], s['token']))
        count = protocol['selection']['frames_per_scene']
        if len(seq) < count:
            raise ValueError('InsufficientPredeclaredSamples:' + name)
        if any(not isinstance(s['timestamp'], int) or s['timestamp'] <= 0 for s in seq[:count]):
            raise ValueError('InvalidSourceTimestamp')
        selected.extend(dict(s, scene_name=name) for s in seq[:count])
    if len(selected) != 8:
        raise ValueError('EightSamplesRequired')
    return selected


def private_manifest(dataroot, splits, protocol):
    """Resolve eight samples' real camera/calibration/pose/annotation links.

    The devkit normally decorates sample.data and sample.anns; join the raw
    tables here explicitly so this preflight needs no CUDA or framework import.
    No image or LiDAR payload is read. A later adapter must verify image hashes.
    """
    root = Path(dataroot).resolve()
    table_root = root / protocol['dataset_version']
    names = ('scene', 'sample', 'sample_data', 'sensor', 'calibrated_sensor',
             'ego_pose', 'sample_annotation')
    tables = {}
    hashes = {}
    for name in names:
        path = table_root / (name + '.json')
        tables[name] = json.loads(path.read_text())
        hashes[name] = sha256(path)
    selected = select_samples(tables['scene'], tables['sample'], splits, protocol)
    indices = {name: {r['token']: r for r in rows} for name, rows in tables.items()}
    if any(len(indices[name]) != len(rows) for name, rows in tables.items()):
        raise ValueError('DuplicateTableToken')
    links = {}
    for sd in tables['sample_data']:
        if not sd['is_key_frame']:
            continue
        cs = indices['calibrated_sensor'][sd['calibrated_sensor_token']]
        sensor = indices['sensor'][cs['sensor_token']]
        key = (sd['sample_token'], sensor['channel'])
        if key in links:
            raise ValueError('AmbiguousKeyframeSensorLink')
        links[key] = sd
    records = []
    for sample in selected:
        record = dict(sample_token=sample['token'], scene_token=sample['scene_token'],
                      scene_name=sample['scene_name'], timestamp_us=sample['timestamp'], cameras=[])
        for channel in protocol['cameras'] + ['LIDAR_TOP']:
            sd = links[(sample['token'], channel)]
            path = (root / sd['filename']).resolve()
            if root not in path.parents:
                raise ValueError('SensorPathEscapesDataRoot')
            if not path.is_file():
                raise FileNotFoundError(str(path))
            cs = indices['calibrated_sensor'][sd['calibrated_sensor_token']]
            pose = indices['ego_pose'][sd['ego_pose_token']]
            if not isinstance(sd['timestamp'], int) or sd['timestamp'] <= 0:
                raise ValueError('InvalidSensorTimestamp')
            entry = dict(channel=channel, sample_data_token=sd['token'],
                         path=str(path), timestamp_us=sd['timestamp'],
                         calibrated_sensor=cs, ego_pose=pose)
            if channel == 'LIDAR_TOP':
                record['reference_lidar_metadata'] = entry
            else:
                if indices['sensor'][cs['sensor_token']]['modality'] != 'camera':
                    raise ValueError('NotCamera')
                record['cameras'].append(entry)
        record['annotation_tokens'] = [a['token'] for a in tables['sample_annotation']
                                       if a['sample_token'] == sample['token']]
        records.append(record)
    return dict(version=protocol['dataset_version'], records=records,
                source_table_sha256=hashes, image_content_verified=False,
                GT_adaptation='NotRun', source_authorization='MustBeRecordedSeparately',
                local_hash_semantics='ReproducibilityOnly_NotOfficialCertification')
