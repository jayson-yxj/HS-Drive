import json
from dataclasses import replace
import numpy as np
import pytest
from datasets.schema import SensorSample, FrameBundle, CAMERA_IDS
from datasets.carla_dataset import CarlaDataset
from simulator.recorder import Recorder
from geometry.reference import camera_mount_to_hs_from_optical, intrinsics_from_hfov


def fixture_bundle():
    K = intrinsics_from_hfov(4,2,100).tolist()
    T = np.eye(4)
    calibration = {'cameras': {c: {'width':4,'height':2,'K':K,
        'T_carla_ego_from_camera':T.tolist(),
        'T_hs_ego_from_opt':camera_mount_to_hs_from_optical(T).tolist()} for c in CAMERA_IDS}}
    sensors = {f'{c}/{k}': SensorSample(f'{c}/{k}',2,.1,4,2,bytes([1,2,3,255])*8,T)
               for c in CAMERA_IDS for k in ('rgb','depth')}
    bundle = FrameBundle(2,.1,sensors,{'T_carla_world_from_ego':T.tolist()},[])
    return calibration,bundle


def test_record_replay_checksums_and_partial_commit(tmp_path):
    cal,bundle = fixture_bundle()
    recorder = Recorder(tmp_path/'scene',cal,{'carla_version':'0.9.15'})
    recorder.write(bundle)
    recorder.close('complete',{'complete':1})
    ds = CarlaDataset(tmp_path/'scene')
    sample,rgb,depth,objects = ds.read(0)
    assert len(ds)==1 and len(rgb)==6
    np.testing.assert_array_equal(rgb['front'][0,0],[3,2,1])
    assert depth['front'].dtype == np.float32
    assert objects == []
    with pytest.raises(FileExistsError):
        recorder.write(bundle)
    (ds.root/sample['cameras']['front']['rgb']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Checksum'):
        ds.read(0)


def test_mismatch_and_failed_write_never_commit(tmp_path,monkeypatch):
    cal,bundle = fixture_bundle()
    recorder = Recorder(tmp_path/'scene',cal,{'carla_version':'0.9.15'})
    sensors = dict(bundle.sensors)
    sensors['front/rgb'] = replace(sensors['front/rgb'],frame_id=3)
    with pytest.raises(ValueError,match='Mixed frame'):
        recorder.write(replace(bundle,sensors=sensors))
    monkeypatch.setattr('simulator.recorder.cv2.imwrite',lambda *args:False)
    with pytest.raises(OSError):
        recorder.write(bundle)
    assert len(CarlaDataset(tmp_path/'scene'))==0


def test_calibration_tampering_rejected(tmp_path):
    cal,bundle = fixture_bundle()
    recorder = Recorder(tmp_path/'scene',cal,{'carla_version':'0.9.15'})
    path=recorder.root/'calibration.json'
    cal['tampered']=True
    path.write_text(json.dumps(cal))
    with pytest.raises(ValueError,match='Calibration'):
        CarlaDataset(recorder.root)


def test_saved_sensor_transform_must_match_same_frame_ego(tmp_path):
    cal,bundle=fixture_bundle()
    recorder=Recorder(tmp_path/'scene',cal,{'carla_version':'0.9.15'})
    sensors=dict(bundle.sensors)
    moved=np.eye(4)
    moved[0,3]=.1
    sensors['front/rgb']=replace(sensors['front/rgb'],T_carla_world_from_camera=moved)
    with pytest.raises(ValueError,match='Extrinsic'):
        recorder.write(replace(bundle,sensors=sensors))
    assert len(CarlaDataset(recorder.root))==0
