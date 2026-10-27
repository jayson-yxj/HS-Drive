"""Synthetic CPU checks in the EXISTING PETR environment, not real-data approval.

Load official implementations, not a rewritten converter. This test is skipped
in the main HS environment; it never initializes a model or GPU context.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

RT = os.environ.get('M8J6_RUNTIME')
if not RT:
    pytest.skip('Run in frozen PETR environment with M8J6_RUNTIME', allow_module_level=True)

import torch
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import ObjectNameFilter, ObjectRangeFilter
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipImage

from datasets.m8j6_selection import official_splits

PROTOCOL = json.loads(Path('configs/m8j6_protocol.json').read_text())
spec = importlib.util.spec_from_file_location('m8j6_frozen_converter',
    Path(RT) / 'sources/PETR/tools/data_converter/nuscenes_converter.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def dataset(boxes=None, names=None):
    d = object.__new__(CustomNuScenesDataset)
    d.CLASSES = PROTOCOL['classes']
    d.with_velocity = True
    d.use_valid_flag = True
    d.box_mode_3d = 0
    d.test_mode = False
    d.modality = dict(use_camera=True)
    if boxes is None:
        boxes = [[10, 1, 2, 2, 4, 2, -.7]]
    names = names if names is not None else ['car'] * len(boxes)
    d.data_infos = [dict(token='synthetic', timestamp=1000000, lidar_path='NeverRead.bin', sweeps=[],
        gt_boxes=np.array(boxes, np.float32), gt_names=np.array(names),
        gt_velocity=np.array([[1., 2.]] * len(boxes)), valid_flag=np.ones(len(boxes), bool),
        cams={name:dict(timestamp=1000000 + i * 1000, data_path='synthetic-' + name + '.jpg',
                       sensor2lidar_rotation=np.eye(3), sensor2lidar_translation=np.array([i, 0, 0]),
                       cam_intrinsic=np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1]]))
              for i, name in enumerate(PROTOCOL['cameras'])})]
    return d


def test_official_intersection_excludes_full_val():
    splits = official_splits(Path(RT) / 'env/lib/python3.8/site-packages/nuscenes/utils/splits.py')
    assert sorted(set(splits['mini_train']) & set(splits['train']))[:2] == PROTOCOL['selection']['scenes']
    assert sorted(set(splits['mini_train']) & set(splits['val'])) == ['scene-0553', 'scene-0796']


def test_official_configuration_unchanged_ten_classes():
    import mmcv
    cfg = mmcv.Config.fromfile(str(Path(RT) / 'sources/PETR' / PROTOCOL['official']['config']))
    assert list(cfg.class_names) == PROTOCOL['classes']
    assert cfg.model.pts_bbox_head.num_query == 900
    assert cfg.model.pts_bbox_head.transformer.decoder.num_layers == 6
    assert cfg.data.train.use_valid_flag and cfg.input_modality.use_camera
    assert not cfg.input_modality.use_lidar and not cfg.input_modality.use_radar
    assert not any('Points' in x['type'] or 'Sweeps' in x['type'] for x in cfg.train_pipeline)


def test_official_center_roundtrip_and_velocity():
    d = dataset()
    ann = d.get_ann_info(0)
    b = ann['gt_bboxes_3d']
    np.testing.assert_allclose(b.gravity_center.numpy(), [[10, 1, 2]])
    np.testing.assert_allclose(b.tensor[:, 2], [1])  # storage bottom center
    np.testing.assert_allclose(b.tensor[:, 3:6], [[2, 4, 2]])
    np.testing.assert_allclose(b.tensor[:, 7:], [[1, 2]])
    assert ann['gt_labels_3d'].tolist() == [0]


def test_official_task_filters_are_not_custom_visibility_or_z_filters():
    d = dataset([[10, 0, 100, 2, 4, 2, 0], [51.2, 0, 0, 2, 4, 2, 0],
                 [10, 0, 0, 2, 4, 2, 0], [10, 0, 0, 2, 4, 2, 0]],
                ['car', 'pedestrian', 'animal', 'car'])
    d.data_infos[0]['valid_flag'][-1] = False
    ann = d.get_ann_info(0)
    assert len(ann['gt_labels_3d']) == 3
    filtered = ObjectRangeFilter(PROTOCOL['geometry']['point_cloud_range'])(ann)
    filtered = ObjectNameFilter(PROTOCOL['classes'])(filtered)
    assert filtered['gt_labels_3d'].tolist() == [0]
    assert filtered['gt_bboxes_3d'].gravity_center[0, 2] == 100


def test_official_nan_velocity_fallback_is_tracked_not_interpolated():
    d = dataset()
    d.data_infos[0]['gt_velocity'][0] = [np.nan, np.nan]
    np.testing.assert_array_equal(d.get_ann_info(0)['gt_bboxes_3d'].tensor[:, 7:], [[0, 0]])


def test_individual_camera_timestamps_and_extrinsic_direction():
    info = dataset().get_data_info(0)
    np.testing.assert_allclose(info['img_timestamp'], np.arange(6) * .001 + 1)
    for i, (k, ext, proj) in enumerate(zip(info['intrinsics'], info['extrinsics'], info['lidar2img'])):
        np.testing.assert_allclose(k @ ext.T, proj)
        np.testing.assert_allclose(ext.T[:3, 3], [-i, 0, 0])


def test_official_fixed_crop_updates_intrinsics_and_projection():
    info = dataset().get_data_info(0)
    old_k = copy.deepcopy(info['intrinsics'])
    # The loader's BGR channels must remain BGR across PIL resize/crop.
    image = np.zeros((900, 1600, 3), np.float32)
    image[:] = [10, 20, 30]
    info['img'] = [image.copy() for _ in range(6)]
    cfg = dict(resize_lim=(.8, 1.), final_dim=(512, 1408), bot_pct_lim=(0., 0.),
               rot_lim=(0., 0.), H=900, W=1600, rand_flip=True)
    out = ResizeCropFlipImage(data_aug_conf=cfg, training=False)(info)
    assert all(x.shape == (512, 1408, 3) for x in out['img'])
    np.testing.assert_array_equal(out['img'][0][0, 0], [10, 20, 30])
    ida = np.array([[.88, 0, 0], [0, .88, -280], [0, 0, 1]])
    for k0, k, ext, proj in zip(old_k, out['intrinsics'], out['extrinsics'], out['lidar2img']):
        np.testing.assert_allclose(k[:3, :3], ida @ k0[:3, :3], atol=1e-4)
        np.testing.assert_allclose(k @ ext.T, proj)


def test_sensor2top_uses_each_sensor_pose_at_its_real_timestamp():
    class SyntheticNuScenes:
        def __init__(self):
            self.rows = dict(sample_data=dict(token='camera', calibrated_sensor_token='cs', ego_pose_token='pose', timestamp=123456),
                             calibrated_sensor=dict(rotation=Quaternion(axis=[0, 0, 1], angle=.2).elements,
                                                    translation=[1, 2, 3]),
                             ego_pose=dict(rotation=Quaternion(axis=[0, 0, 1], angle=.3).elements, translation=[10, 0, 0]))
        def get(self, table, token):
            return self.rows[table]
        def get_sample_data_path(self, token):
            return '/synthetic/camera.jpg'
    n = SyntheticNuScenes()
    rl = Quaternion(axis=[0, 0, 1], angle=-.1).rotation_matrix
    re = Quaternion(axis=[0, 0, 1], angle=-.2).rotation_matrix
    tl, te = np.array([.5, 1., 0.]), np.array([5., 2., 0.])
    result = converter.obtain_sensor2top(n, 'camera', tl, rl, te, re, 'CAM_FRONT')
    point = np.array([1., 1., 10.])
    cs, ep = n.rows['calibrated_sensor'], n.rows['ego_pose']
    global_p = Quaternion(ep['rotation']).rotation_matrix @ (Quaternion(cs['rotation']).rotation_matrix @ point + cs['translation']) + ep['translation']
    expected = rl.T @ (re.T @ (global_p - te) - tl)
    actual = result['sensor2lidar_rotation'] @ point + result['sensor2lidar_translation']
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert result['timestamp'] == 123456


@pytest.mark.parametrize('yaw', [0., .3, -1.2, np.pi / 2])
def test_converter_old_second_yaw_and_extent_match_nuscenes_corners(yaw):
    from nuscenes.utils.data_classes import Box
    # nuScenes dimensions are full w/l/h, never CARLA half extents.
    reference = Box([10, 2, 3], [2, 4, 1.5], Quaternion(axis=[0, 0, 1], angle=yaw))
    converted = LiDARInstance3DBoxes(
        torch.tensor([[10, 2, 3, 2, 4, 1.5, -yaw - np.pi / 2]], dtype=torch.float32),
        box_dim=7, origin=(.5, .5, .5))
    points = converted.corners[0].numpy()
    source = reference.corners().T
    distances = np.linalg.norm(points[:, None] - source[None], axis=-1)
    assert distances.min(axis=0).max() < 2e-6
    assert distances.min(axis=1).max() < 2e-6
