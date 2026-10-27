"""Synthetic metadata only; never read private images or annotation files."""
import copy
import json
from pathlib import Path

import pytest

from datasets.m8j6_selection import private_manifest, select_samples

PROTOCOL = json.loads(Path('configs/m8j6_protocol.json').read_text())


def fixture():
    names = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757']
    scenes = [dict(name=n, token=n + '-token') for n in names]
    samples = [dict(token=n + '-sample-' + str(i), scene_token=n + '-token', timestamp=1000 + i)
               for n in names for i in range(6)]
    splits = dict(mini_train=names, train=['scene-0061', 'scene-0655', 'scene-0757'],
                  val=['scene-0553'], mini_val=['scene-0103'])
    return scenes, samples, splits


def test_selection_uses_intersection_and_source_order_not_predictions():
    scenes, samples, splits = fixture()
    selected = select_samples(scenes, list(reversed(samples)), splits, PROTOCOL)
    assert [s['scene_name'] for s in selected] == ['scene-0061'] * 4 + ['scene-0655'] * 4
    assert [s['timestamp'] for s in selected] == [1000, 1001, 1002, 1003] * 2


def test_same_timestamp_has_stable_token_tiebreak():
    scenes, samples, splits = fixture()
    for s in samples:
        s['timestamp'] = 1
    assert [s['token'] for s in select_samples(scenes, samples[::-1], splits, PROTOCOL)] == [
        name + '-sample-' + str(i) for name in PROTOCOL['selection']['scenes'] for i in range(4)]


@pytest.mark.parametrize('failure', ['missing_scene', 'too_few', 'duplicate', 'val', 'changed_split', 'timestamp'])
def test_failure_never_silently_substitutes_samples(failure):
    scenes, samples, splits = fixture()
    if failure == 'missing_scene':
        scenes = scenes[1:]
    elif failure == 'too_few':
        samples = [s for s in samples if s['scene_token'] != 'scene-0061-token' or s['timestamp'] < 1003]
    elif failure == 'duplicate':
        samples.append(copy.deepcopy(samples[0]))
    elif failure == 'val':
        splits['val'].append('scene-0061')
    elif failure == 'changed_split':
        splits['train'].remove('scene-0061')
    else:
        samples[0]['timestamp'] = 0
    with pytest.raises(ValueError):
        select_samples(scenes, samples, splits, PROTOCOL)


def write_tables(root):
    scenes, samples, splits = fixture()
    tables = dict(scene=scenes, sample=samples, sensor=[], calibrated_sensor=[], ego_pose=[],
                  sample_data=[], sample_annotation=[])
    for j, channel in enumerate(PROTOCOL['cameras'] + ['LIDAR_TOP']):
        tables['sensor'].append(dict(token=channel, channel=channel,
                                     modality='lidar' if channel == 'LIDAR_TOP' else 'camera'))
        tables['calibrated_sensor'].append(dict(token='cs-' + channel, sensor_token=channel,
            rotation=[1, 0, 0, 0], translation=[0, 0, 0], camera_intrinsic=[[10, 0, 5], [0, 10, 5], [0, 0, 1]]))
        for s in samples:
            token = s['token'] + channel
            filename = 'samples/' + token + ('.bin' if channel == 'LIDAR_TOP' else '.jpg')
            path = root / filename
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'synthetic placeholder, not an image')
            tables['ego_pose'].append(dict(token='pose-' + token, translation=[j, 0, 0], rotation=[1, 0, 0, 0]))
            tables['sample_data'].append(dict(token=token, sample_token=s['token'], is_key_frame=True,
                calibrated_sensor_token='cs-' + channel, ego_pose_token='pose-' + token,
                timestamp=s['timestamp'] + j, filename=filename))
    table_root = root / 'v1.0-mini'
    table_root.mkdir()
    for name, rows in tables.items():
        (table_root / (name + '.json')).write_text(json.dumps(rows))
    return splits, tables


def test_manifest_keeps_asynchronous_calibrations_and_metadata_only(tmp_path):
    splits, _ = write_tables(tmp_path)
    result = private_manifest(tmp_path, splits, PROTOCOL)
    assert len(result['records']) == 8
    r = result['records'][0]
    assert [c['channel'] for c in r['cameras']] == PROTOCOL['cameras']
    assert [c['timestamp_us'] for c in r['cameras']] == list(range(1000, 1006))
    assert r['reference_lidar_metadata']['timestamp_us'] == 1006
    assert result['GT_adaptation'] == 'NotRun' and not result['image_content_verified']


@pytest.mark.parametrize('failure', ['missing_file', 'escape_path', 'duplicate_sensor', 'missing_camera'])
def test_manifest_rejects_broken_links(tmp_path, failure):
    splits, tables = write_tables(tmp_path)
    records = tables['sample_data']
    if failure == 'missing_file':
        (tmp_path / records[0]['filename']).unlink()
    elif failure == 'escape_path':
        records[0]['filename'] = '../escape.jpg'
    elif failure == 'duplicate_sensor':
        records.append(dict(records[0], token='different-token-same-camera'))
    else:
        records.pop(0)
    (tmp_path / 'v1.0-mini/sample_data.json').write_text(json.dumps(records))
    with pytest.raises((FileNotFoundError, ValueError, KeyError)):
        private_manifest(tmp_path, splits, PROTOCOL)
