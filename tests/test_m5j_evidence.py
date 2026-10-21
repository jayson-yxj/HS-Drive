import pytest
import torch
from models.m5j_evidence import reduce_evidence
from models.m5j_candidate import validate_packet,MobileBEVCandidate,CAMERAS


def test_duplicate_bins_not_independent_rays_or_cameras():
    rank=torch.tensor([[[[[0]],[[0]],[[1]]],[[[0]],[[1]],[[1]]]]])
    p=torch.tensor([[[[[.2]],[[.3]],[[.5]]],[[[.4]],[[.1]],[[.5]]]]])
    r=reduce_evidence(rank,torch.ones_like(rank,dtype=torch.bool),p,torch.tensor([[[[.1]],[[.8]]]]),(1,1,2))
    assert r['ray_bin_count'].flatten().tolist()==[3,3]
    assert r['ray_count'].flatten().tolist()==[2,2]
    assert r['camera_count'].flatten().tolist()==[2,2]
    assert torch.allclose(r['foreground_mass'].flatten(),torch.tensor([.37,.53]))
    assert torch.allclose(r['max_ray_mass'].flatten(),torch.tensor([.32,.48]))


def test_regularized_never_amplifies_tiny_total_and_gate_loses_weak_positive():
    r=reduce_evidence(torch.zeros(1,1,1,1,1,dtype=torch.long),torch.ones(1,1,1,1,1,dtype=torch.bool),torch.full((1,1,1,1,1),1e-9),torch.full((1,1,1,1),.8),(1,1,1))
    assert 0<=r['scores']['regularized'].item()<=r['scores']['raw'].item()<1e-8
    assert r['scores']['gate'].item()==0


def packet():
    return dict(images=torch.zeros(1,6,3,384,640),intrinsics=torch.eye(3).expand(1,6,3,3).clone(),cam_to_ego=torch.eye(4).expand(1,6,4,4).clone(),ego_pose=torch.eye(4)[None],ego_state=torch.zeros(1,7),route_command=torch.zeros(1,dtype=torch.int64),timestamps_s=torch.tensor([2.],dtype=torch.float64),valid_cameras=torch.ones(1,6,dtype=torch.bool))


def test_inference_contract_rejects_supervision_and_invalid_time():
    x=packet();validate_packet(x,CAMERAS,2.1)
    for key in ['depth','actor','instance','pair_witness','surface_valid']:
        with pytest.raises(ValueError):validate_packet(dict(x,**{key:torch.zeros(1)}),CAMERAS)
    for now in [1.,3.,float('nan')]:
        with pytest.raises(ValueError):validate_packet(x,CAMERAS,now)
    with pytest.raises(ValueError):validate_packet(x,tuple(reversed(CAMERAS)))
    with pytest.raises(ValueError):MobileBEVCandidate()
    x['intrinsics'][0,0,0,0]=0
    with pytest.raises(ValueError):validate_packet(x,CAMERAS)

def test_distinct_support_matches_explicit_unique_and_conserves_mass():
    torch.manual_seed(4)
    rank=torch.sort(torch.randint(0,6,(1,2,12,3,2)),dim=2).values
    keep=torch.ones_like(rank,dtype=torch.bool);keep[:,:,0]=False
    pr=torch.softmax(torch.randn_like(rank,dtype=torch.float32),2)
    q=torch.rand(1,2,3,2);result=reduce_evidence(rank,keep,pr,q,(1,2,3))
    for cell in range(6):
        hit=keep&(rank==cell);unique=hit.any(2).sum()
        assert result['ray_count'].flatten()[cell]==unique
        assert result['camera_count'].flatten()[cell]==hit.any(2).flatten(2).any(2).sum()
    assert torch.allclose(result['foreground_mass'].sum(),(pr*q[:,:,None])[keep].sum(),atol=1e-6)


def test_spatial_distance_metric_matches_existing_reference():
    import numpy as np
    from tools.m5j_metrics import evaluate
    from diagnostics.m5c_localization import matching
    truth=np.zeros((4,8,8),bool);truth[2,4,4]=True
    score=np.zeros_like(truth,dtype=float);score[2,5,4]=1;score[1,4,4]=1
    result=evaluate(score,truth,np.ones_like(truth),np.array([truth]),.2)
    old=matching(score>=.2,truth)
    assert result['matched1']==old['tolerances']['1.0']['matched_gt']
    assert result['matchedpred1']==old['tolerances']['1.0']['matched_pred']
