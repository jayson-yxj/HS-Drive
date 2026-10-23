import copy
import numpy as np
import pytest
from datasets.m7a_shadow import validate_packet,ReplayQueue
from datasets.schema import CAMERA_IDS


def packet(frame=10,t=1.):
    return dict(schema='m7a.six_rgb.v1',sample_id='s',camera_order=list(CAMERA_IDS),
      frame_id=frame,timestamp_s=t,valid_until_s=t+.35,
      cameras=[dict(name=n,path='/tmp/f_'+n+'_rgb.png',sha256='sha',frame_id=frame,timestamp_s=t) for n in CAMERA_IDS],
      intrinsics=[[[100,0,319.5],[0,100,191.5],[0,0,1]]]*6,
      cam_to_ego=[np.eye(4).tolist()]*6,ego_pose=np.eye(4).tolist(),
      ego_pose_source='CARLA_metadata_for_alignment_only_not_model_feature',image_size=[384,640],source='RGB_OfflineReplay')


def test_strict_schema_gt_rejected():
    p=packet();p['actors']=[]
    with pytest.raises(ValueError,match='GT'):validate_packet(p,1,verify_files=False)


@pytest.mark.parametrize('mutation',['missing','order','frame','timestamp','reflection','depth','nan'])
def test_corrupt_camera_contract(mutation):
    p=packet()
    if mutation=='missing':p['cameras'].pop()
    if mutation=='order':p['camera_order'].reverse()
    if mutation=='frame':p['cameras'][2]['frame_id']+=1
    if mutation=='timestamp':p['cameras'][2]['timestamp_s']+=.05
    if mutation=='reflection':p['cam_to_ego']=copy.deepcopy(p['cam_to_ego']);p['cam_to_ego'][0][0][0]=-1
    if mutation=='depth':p['cameras'][0]['path']='/tmp/f_front_depth.png'
    if mutation=='nan':p['cameras'][0]['timestamp_s']=float('nan')
    with pytest.raises(ValueError):validate_packet(p,1,verify_files=False)


def test_queue_overflow_stale_out_of_order_and_no_alias():
    q=ReplayQueue(2);p=packet();assert q.push(p,1.)
    p['cameras'][0]['path']='changed'
    assert q.queue[0]['cameras'][0]['path']!='changed'
    assert q.push(packet(11,1.05),1.05)
    assert q.push(packet(12,1.10),1.10)
    assert not q.push(packet(11,1.05),1.10)
    assert q.pop(2.) is None
    assert q.dropped==['queue_overflow','out_of_order','stale_or_future','stale_or_future']
