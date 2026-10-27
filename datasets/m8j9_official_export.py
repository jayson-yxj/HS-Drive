"""Thin adapter into the frozen MMDetection3D nuScenes exporter; no model."""
import json
from pathlib import Path
import numpy as np
import torch
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.detection.config import config_factory
from training.m8j7_metrics import prediction_mask


ALLOWED_TABLES = ('sample','sample_data','calibrated_sensor','ego_pose','scene','sensor')


def read_metadata_only(root, version):
    """Export deliberately never constructs NuScenes, which loads GT tables."""
    tables={}
    for name in ALLOWED_TABLES:
        rows=json.loads((Path(root)/version/(name+'.json')).read_text())
        tables[name]={r['token']:r for r in rows}
        assert len(tables[name])==len(rows)
    return tables


def check_manifest(manifest, tables, mini_val_scenes):
    samples=tables['sample'];scenes=tables['scene']
    expected={s['token'] for s in samples.values() if scenes[s['scene_token']]['name'] in mini_val_scenes}
    tokens=[f['sample_token'] for f in manifest['frames']]
    assert len(tokens)==len(set(tokens)), 'Duplicate sample token'
    assert set(tokens)==expected, 'Missing or extra mini_val sample'
    for f in manifest['frames']:
        sample=samples[f['sample_token']]
        assert sample['scene_token']==f['scene_token']
        assert scenes[sample['scene_token']]['name']==f['scene_name']
        assert sample['timestamp']==f['timestamp_us']
    return tokens


def pose_info(sample_token, tables):
    sample=tables['sample'][sample_token]
    sd=tables['sample_data'][sample['data']['LIDAR_TOP']]
    assert sd['sample_token']==sample_token and sd['is_key_frame']
    cs=tables['calibrated_sensor'][sd['calibrated_sensor_token']]
    sensor=tables['sensor'][cs['sensor_token']]
    assert sensor['channel']=='LIDAR_TOP' and sensor['modality']=='lidar'
    pose=tables['ego_pose'][sd['ego_pose_token']]
    return dict(token=sample_token,lidar2ego_translation=cs['translation'],lidar2ego_rotation=cs['rotation'],
                ego2global_translation=pose['translation'],ego2global_rotation=pose['rotation'],
                reference_sample_data_token=sd['token'],reference_timestamp_us=sd['timestamp'],
                ego_pose_token=pose['token'],calibrated_sensor_token=cs['token'])


def prediction_to_export_input(cache, classes):
    """Access only prediction fields; GT members of the same npz stay unopened."""
    boxes=np.asarray(cache['boxes']);scores=np.asarray(cache['scores']);labels=np.asarray(cache['labels'])
    assert boxes.ndim==2 and boxes.shape[1]==9, 'Missing box or predicted velocity'
    assert boxes.shape[0]==len(scores)==len(labels)
    assert len(scores)<=300 and np.isfinite(boxes).all() and np.isfinite(scores).all()
    assert (boxes[:,3:6]>0).all() and ((scores>=0)&(scores<=1)).all()
    assert np.issubdtype(labels.dtype,np.integer) and ((labels>=0)&(labels<len(classes))).all()
    keep=prediction_mask(boxes,scores,.2)
    return dict(boxes_3d=LiDARInstance3DBoxes(torch.from_numpy(boxes[keep].copy()),box_dim=9),
                scores_3d=torch.from_numpy(scores[keep].copy()),labels_3d=torch.from_numpy(labels[keep].copy())),np.flatnonzero(keep)


def exporter(infos, classes, modality):
    # PETR CustomNuScenesDataset inherits this exact method; no override.
    dataset=object.__new__(NuScenesDataset)
    dataset.data_infos=infos;dataset.CLASSES=tuple(classes);dataset.modality=modality
    dataset.eval_version='detection_cvpr_2019'
    dataset.eval_detection_configs=config_factory(dataset.eval_version)
    return dataset


def independent_transform(boxes, info):
    """Independent column-vector check of source box convention, no GT access."""
    ql=Quaternion(info['lidar2ego_rotation']);qe=Quaternion(info['ego2global_rotation'])
    Rl,Re=ql.rotation_matrix,qe.rotation_matrix
    center=boxes[:,:3].astype(np.float64).copy();center[:,2]+=boxes[:,5]/2
    ego=center@Rl.T+np.asarray(info['lidar2ego_translation'])
    world=ego@Re.T+np.asarray(info['ego2global_translation'])
    local_v=np.c_[boxes[:,7:9],np.zeros(len(boxes))]
    world_v=local_v@Rl.T@Re.T
    orientations=[(qe*ql*Quaternion(axis=[0,0,1],radians=-float(yaw)-np.pi/2)).elements for yaw in boxes[:,6]]
    return ego,world,world_v,orientations


def compare_export(detection,info,rows,classes,config):
    boxes=detection['boxes_3d'].tensor.numpy()
    scores=detection['scores_3d'].numpy();labels=detection['labels_3d'].numpy()
    ego,world,velocity,quaternions=independent_transform(boxes,info)
    keep=[i for i,lab in enumerate(labels) if np.linalg.norm(ego[i,:2])<=config.class_range[classes[int(lab)]]]
    assert len(rows)==len(keep)
    maxima=dict(center_m=0.,size_m=0.,rotation_matrix=0.,velocity_mps=0.,score=0.)
    for row,i in zip(rows,keep):
        assert row['sample_token']==info['token'] and row['detection_name']==classes[int(labels[i])]
        expected=dict(center_m=np.max(np.abs(np.asarray(row['translation'])-world[i])),
            size_m=np.max(np.abs(np.asarray(row['size'])-boxes[i,3:6])),
            rotation_matrix=np.max(np.abs(Quaternion(row['rotation']).rotation_matrix-Quaternion(quaternions[i]).rotation_matrix)),
            velocity_mps=np.max(np.abs(np.asarray(row['velocity'])-velocity[i,:2])),score=abs(row['detection_score']-float(scores[i])))
        for k,v in expected.items():maxima[k]=max(maxima[k],float(v))
        # Float32 yaw preprocessing/box gravity center can round relative to double math.
        assert all(v<=1e-5 for v in expected.values()),expected
        assert abs(np.linalg.norm(row['rotation'])-1.)<1e-6
    return maxima,keep
