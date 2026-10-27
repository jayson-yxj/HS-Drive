import json
import numpy as np
import pytest
from pyquaternion import Quaternion
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from datasets.m8j9_official_export import (prediction_to_export_input,exporter,compare_export,
    check_manifest,pose_info)

CLASSES=('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
META=dict(use_camera=True,use_lidar=False,use_radar=False,use_map=False,use_external=False)


def info():
    return dict(token='sample',lidar2ego_translation=[1.,2.,3.],
        lidar2ego_rotation=Quaternion(axis=[0,0,1],angle=np.pi/2).elements.tolist(),
        ego2global_translation=[100.,200.,2.],ego2global_rotation=Quaternion(axis=[0,0,1],angle=.3).elements.tolist())


def record():
    return dict(boxes=np.array([[2.,1.,-.4,1.7,4.3,1.9,-.8,2.,.5]],dtype=np.float32),
                scores=np.array([.8],dtype=np.float32),labels=np.array([0]))


def test_frozen_petr_inherits_exact_exporter():
    from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
    from mmdet3d.datasets import NuScenesDataset
    assert CustomNuScenesDataset._format_bbox is NuScenesDataset._format_bbox


def test_asymmetric_size_bottom_center_yaw_velocity_global_roundtrip(tmp_path):
    r=record();det,_=prediction_to_export_input(r,CLASSES);ds=exporter([info()],CLASSES,META)
    path=ds._format_bbox([det],str(tmp_path/'export'))
    payload=json.loads(open(path).read());rows=payload['results']['sample']
    errors,keep=compare_export(det,info(),rows,CLASSES,ds.eval_detection_configs)
    assert max(errors.values())<1e-5 and keep==[0]
    np.testing.assert_allclose(rows[0]['size'],[1.7,4.3,1.9],atol=1e-6)
    box=DetectionBox.deserialize(rows[0]);assert box.serialize()==DetectionBox.deserialize(box.serialize()).serialize()
    loaded,meta=load_prediction(path,500,DetectionBox)
    assert loaded.sample_tokens==['sample'] and meta==META
    np.testing.assert_allclose(loaded['sample'][0].translation,rows[0]['translation'])


def test_export_does_not_access_GT_members(tmp_path):
    class Guard(dict):
        def __getitem__(self,key):
            assert key in ('boxes','scores','labels'),key
            return super().__getitem__(key)
    r=record();r['gt']=object();r['gt_labels']=object()
    det,_=prediction_to_export_input(Guard(r),CLASSES)
    path=exporter([info()],CLASSES,META)._format_bbox([det],str(tmp_path/'no_gt'))
    assert len(json.loads(open(path).read())['results']['sample'])==1


def test_score_range_and_zero_predictions_are_real_not_missing(tmp_path):
    r=record();r['boxes']=np.repeat(r['boxes'],3,axis=0);r['boxes'][2,0]=51.2
    r['scores']=np.array([.2,.199,.9]);r['labels']=np.zeros(3,dtype=int)
    det,ids=prediction_to_export_input(r,CLASSES);assert ids.tolist()==[0]
    r['scores'][:]=.1;empty,_=prediction_to_export_input(r,CLASSES)
    path=exporter([info()],CLASSES,META)._format_bbox([empty],str(tmp_path/'empty'))
    assert json.loads(open(path).read())['results']=={'sample':[]}


def test_missing_velocity_stops_instead_of_GT_fill():
    r=record();r['boxes']=r['boxes'][:,:7]
    with pytest.raises(AssertionError,match='velocity'):prediction_to_export_input(r,CLASSES)


def test_official_class_range_applied_without_GT(tmp_path):
    r=record();r['boxes'][0,0]=45.;r['labels'][0]=8
    det,_=prediction_to_export_input(r,CLASSES);ds=exporter([info()],CLASSES,META)
    path=ds._format_bbox([det],str(tmp_path/'range'))
    assert json.loads(open(path).read())['results']['sample']==[]


@pytest.mark.parametrize('name,speed,attribute',[
    ('car',1.,'vehicle.moving'),('car',0.,'vehicle.parked'),
    ('bus',0.,'vehicle.stopped'),('pedestrian',0.,'pedestrian.standing'),
    ('pedestrian',1.,'pedestrian.moving'),('bicycle',0.,'cycle.without_rider'),
    ('motorcycle',1.,'cycle.with_rider'),('traffic_cone',0.,''),('barrier',0.,'')])
def test_original_deterministic_attribute_rules(tmp_path,name,speed,attribute):
    r=record();r['labels'][0]=CLASSES.index(name);r['boxes'][0,7:9]=[speed,0.]
    det,_=prediction_to_export_input(r,CLASSES)
    path=exporter([info()],CLASSES,META)._format_bbox([det],str(tmp_path/name))
    assert json.loads(open(path).read())['results']['sample'][0]['attribute_name']==attribute


def test_sample_manifest_catches_missing_duplicate_and_wrong_timestamp():
    tables=dict(sample={'s':dict(token='s',scene_token='scene',timestamp=100)},scene={'scene':dict(name='val')})
    row=dict(sample_token='s',scene_token='scene',scene_name='val',timestamp_us=100)
    assert check_manifest(dict(frames=[row]),tables,['val'])==['s']
    for rows in ([],[row,row],[dict(row,timestamp_us=101)]):
        with pytest.raises(AssertionError):check_manifest(dict(frames=rows),tables,['val'])


def test_raw_json_sensor_join_uses_only_keyframes_and_no_annotations(tmp_path):
    from datasets.m8j9_official_export import read_metadata_only
    root=tmp_path/'v1.0-mini';root.mkdir()
    pose=dict(token='pose',translation=[1.,2.,3.],rotation=[1.,0.,0.,0.])
    cs=dict(token='cs',sensor_token='sensor',translation=[.1,.2,.3],rotation=[1.,0.,0.,0.])
    key=dict(token='key',sample_token='sample',calibrated_sensor_token='cs',ego_pose_token='pose',timestamp=99,is_key_frame=True)
    data=dict(sample=[dict(token='sample',scene_token='scene',timestamp=100)],
        sample_data=[key,dict(key,token='sweep',is_key_frame=False)],calibrated_sensor=[cs],
        ego_pose=[pose],scene=[dict(token='scene',name='mini_val')],
        sensor=[dict(token='sensor',channel='LIDAR_TOP',modality='lidar')])
    for name,rows in data.items():(root/(name+'.json')).write_text(json.dumps(rows))
    tables=read_metadata_only(tmp_path,'v1.0-mini')
    assert tables['sample']['sample']['data']==dict(LIDAR_TOP='key')
    result=pose_info('sample',tables)
    assert result['reference_timestamp_us']==99
    assert result['ego2global_translation']==[1.,2.,3.]


def test_diagnostic_canonical_handles_official_numpy_velocity_and_nan():
    from tools.m8j9_stage import canonical
    x=dict(v=np.array([1.,float('nan')]),n=np.int64(2),t=(3.,4.))
    y=dict(t=[3.,4.],n=2,v=[1.,float('nan')])
    assert canonical(x)==canonical(y)
    assert canonical(x)!=canonical(dict(y,n=3))
