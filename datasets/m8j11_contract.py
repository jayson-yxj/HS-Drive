"""Small, prediction-only bridges for the frozen StreamPETR comparison."""
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion

CAMERAS=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')
INPUT_FIELDS={'images','lidar2img','intrinsics','extrinsics','timestamp','img_timestamp','ego_pose','ego_pose_inv'}
TABLES=('sample','sample_data','calibrated_sensor','ego_pose','scene','sensor')


def read_metadata(root):
    t={k:{x['token']:x for x in json.loads((Path(root)/'v1.0-mini'/f'{k}.json').read_text())} for k in TABLES}
    for s in t['sample'].values():s['data']={}
    for sd in t['sample_data'].values():
        if sd['is_key_frame']:
            cs=t['calibrated_sensor'][sd['calibrated_sensor_token']]
            ch=t['sensor'][cs['sensor_token']]['channel']
            assert ch not in t['sample'][sd['sample_token']]['data']
            t['sample'][sd['sample_token']]['data'][ch]=sd['token']
    return t


def matrix(rotation,translation):
    m=np.eye(4);m[:3,:3]=Quaternion(rotation).rotation_matrix;m[:3,3]=translation;return m


def sensor_global(sd,t):
    c=t['calibrated_sensor'][sd['calibrated_sensor_token']];e=t['ego_pose'][sd['ego_pose_token']]
    return matrix(e['rotation'],e['translation'])@matrix(c['rotation'],c['translation'])


def info_from_metadata(frame,t,root):
    s=t['sample'][frame['sample_token']];ld=t['sample_data'][s['data']['LIDAR_TOP']]
    cs=t['calibrated_sensor'][ld['calibrated_sensor_token']];ep=t['ego_pose'][ld['ego_pose_token']]
    ref=sensor_global(ld,t);cams={}
    for name in CAMERAS:
        sd=t['sample_data'][s['data'][name]];c=t['calibrated_sensor'][sd['calibrated_sensor_token']]
        camera_to_lidar=np.linalg.inv(ref)@sensor_global(sd,t)
        cams[name]=dict(sensor2lidar_rotation=camera_to_lidar[:3,:3],sensor2lidar_translation=camera_to_lidar[:3,3],
            timestamp=sd['timestamp'],data_path=str(Path(root)/sd['filename']),cam_intrinsic=np.asarray(c['camera_intrinsic']))
    return dict(token=s['token'],scene_token=s['scene_token'],timestamp=s['timestamp'],prev=s['prev'],next=s['next'],
        frame_idx=frame['index'],lidar_path='NOT_READ_METADATA_REFERENCE_ONLY',sweeps=[],cams=cams,
        lidar2ego_rotation=cs['rotation'],lidar2ego_translation=cs['translation'],
        ego2global_rotation=ep['rotation'],ego2global_translation=ep['translation'])


def check_sequence(frames,t):
    seen=set();last=None
    for f in frames:
        s=t['sample'][f['sample_token']]
        assert f['sample_token'] not in seen;seen.add(f['sample_token'])
        assert s['timestamp']==f['timestamp_us'] and s['scene_token']==f['scene_token']
        if last and last['scene_token']==f['scene_token']:
            assert f['timestamp_us']>last['timestamp_us']
            assert s['prev']==last['sample_token']
        else:assert not s['prev'],'Must start each scene at its first keyframe'
        last=f
    assert not t['sample'][frames[-1]['sample_token']]['next']


def to_legacy_boxes(boxes):
    """rc6 bottom XYZ, full LWH, CCW yaw -> J9 bottom XYZ, WLH, SECOND yaw."""
    b=np.asarray(boxes);assert b.ndim==2 and b.shape[1]==9 and np.isfinite(b).all()
    out=b.copy();out[:,3:5]=b[:,[4,3]];out[:,6]=-b[:,6]-np.pi/2
    return out


def from_legacy_boxes(boxes):
    return to_legacy_boxes(boxes)  # Both dimension swap and yaw map are involutions.
