"""Evaluator-side metadata audit and explicit RGB-only packet export."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from datasets.schema import CAMERA_IDS
from datasets.m7a_shadow import validate_packet,ReplayQueue
from tools.m5o_common import rows
from tools.m7a_common import read,write,asset,provenance


def export(row):
    inp=read(row['input']);frame=read(row['raw_frame'])
    assert asset(row['input'])['sha256']==row['input_sha256']
    assert set(inp)=={'sample_id','images','intrinsics','cam_to_ego','ego_pose','timestamp_s'}
    cameras=[]
    for name,image in zip(CAMERA_IDS,inp['images']):
        raw=frame['cameras'][name]
        assert raw['rgb_sha256']==image['sha256']
        assert np.array(Image.open(image['path'])).shape==(384,640,3)
        cameras.append(dict(name=name,path=image['path'],sha256=image['sha256'],
           frame_id=raw['rgb_frame_id'],timestamp_s=raw['rgb_timestamp_s']))
    assert np.allclose(inp['ego_pose'],frame['T_world_from_ego'],atol=1e-8)
    assert inp['timestamp_s']==frame['timestamp_s']
    p=dict(schema='m7a.six_rgb.v1',sample_id=inp['sample_id'],camera_order=list(CAMERA_IDS),
      frame_id=frame['frame_id'],timestamp_s=inp['timestamp_s'],valid_until_s=inp['timestamp_s']+.35,
      cameras=cameras,intrinsics=inp['intrinsics'],cam_to_ego=inp['cam_to_ego'],ego_pose=inp['ego_pose'],
      ego_pose_source='CARLA_metadata_for_alignment_only_not_model_feature',image_size=[384,640],source='RGB_OfflineReplay')
    validate_packet(p,p['timestamp_s'])
    # FP64 optical -> ego -> optical reprojection, all six cameras.
    errors=[]
    for K,T in zip(p['intrinsics'],p['cam_to_ego']):
        K=np.array(K);T=np.array(T);pixels=np.array([[0.,0.,1.],[319.5,191.5,1.],[639,383,1.]])
        opt=np.linalg.solve(K,pixels.T).T*np.array([5,20,50])[:,None]
        ego=opt@T[:3,:3].T+T[:3,3]
        back=(np.c_[ego,np.ones(3)]@np.linalg.inv(T).T)[:,:3]
        uv=back@K.T;uv=uv[:,:2]/uv[:,2:]
        errors.append(float(abs(uv-pixels[:,:2]).max()))
    return p,errors


def main():
    root=Path('reports/m7a_shadow_packets');root.mkdir(exist_ok=False);records=[];selected=[]
    for split in ['train','val','calibration','sealed']:
        seen=set()
        for row in rows(split):
            if row['group'] in seen:continue
            seen.add(row['group']);selected.append((split,row))
    train=rows('train');groups=list(dict.fromkeys(r['group'] for r in train))[:2]
    gpu=[next(r for r in train if r['group']==g and r['configuration']==1) for g in groups]
    for split,row in selected+[('GPU_replay',r) for r in gpu]:
        p,errors=export(row);path=root/(row['sample_id'].replace('/','_')+'.json')
        if not path.exists():write(path,p)
        records.append(dict(sample_id=row['sample_id'],group=row['group'],historical_split=split,packet=asset(path),
          input=asset(row['input']),raw_metadata_evaluator_only=asset(row['raw_frame']),frame=p['frame_id'],timestamp=p['timestamp_s'],
          roundtrip_pixel_max=max(errors),camera_count=6,calibration_source='recorded_K_T',status='Passed'))
    cfg=read('configs/m5j_models.json');mobile=next(r for r in cfg['pixel'] if r['name']=='mobile_gt' and r['seed']==101)
    direct=next(r for r in read('reports/m5o_training.json')['records'] if r['seed']==101 and r['fraction']==1 and r['arm']=='direct')['selected']
    manifest=dict(packets=[r['packet'] for r in records if r['historical_split']=='GPU_replay'],
      mobile=dict(checkpoint=mobile['checkpoint'],sha256=mobile['sha256']),
      direct=dict(checkpoint=direct['path'],sha256=direct['sha256']),qualification='CandidateNoGo')
    write('reports/m7a_shadow_inference_manifest.json',manifest)
    write('reports/m7a_six_camera_shadow_contract.json',dict(kind='OfflineReplayMetadataAndRGBBytes',provenance=provenance(),
      status='InterfaceVerified_NotPerceptionQuality',records=records,packets=len(records),unique_groups=len(set(r['group'] for r in records)),
      camera_order=list(CAMERA_IDS),image_size=[384,640],rgb_only_model_inputs=['RGB','K','T_cam_to_ego'],
      ego_pose='Simulator metadata alignment only; not model feature; not proof of visual odometry',
      frame_timestamp='per_camera_original_CARLA_frame_and_timestamp_verified',
      actor_reference='M5O ActorBoxCenter evaluation sidecar only; none in packets',
      coordinate='HS ego x-forward/y-left/z-up; optical x-right/y-down/z-forward',
      limits=['Historical sealed opened previously; regression only','Static historical samples do not certify live moving six-camera robustness','No new sensor collection','No depth/instance/actor fields in model input','No model capability upgrade']))
    print('Audited',len(records),'six-camera packets;',len(manifest['packets']),'selected frozen GPU forwards')

if __name__=='__main__':main()
