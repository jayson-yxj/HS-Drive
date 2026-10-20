import pytest,numpy as np
torch=pytest.importorskip('torch')
from geometry.temporal_warp import warp_bev,BEVMemory


def pose(x=0,y=0,yaw=0):
    p=torch.eye(4)[None];c,s=np.cos(yaw),np.sin(yaw);p[0,:2,:2]=torch.tensor([[c,-s],[s,c]]);p[0,:2,3]=torch.tensor([x,y]);return p

KW=dict(x_range=(-2.,2.),y_range=(-2.,2.),resolution=.5)


@pytest.mark.parametrize('dx,dy,expected',[(0,0,(5,4)),(1,0,(3,4)),(0,1,(5,2))])
def test_identity_and_exact_two_cell_translation(dx,dy,expected):
    x=torch.zeros(1,1,8,8);x[0,0,5,4]=1
    y,m,t=warp_bev(x,pose(),pose(dx,dy),**KW)
    assert y[0,0,expected[0],expected[1]]==1 and torch.count_nonzero(y)==1
    if dx:assert not m[0,0,-1].any()


def test_rotation_and_combination_against_forward_point():
    x=torch.zeros(1,1,8,8);x[0,0,5,4]=1
    for pp in [pose(yaw=np.pi/2),pose(.5,0,np.pi/2)]:
        y,m,t=warp_bev(x,pose(),pp,**KW);point=t[0]@torch.tensor([.75,.25,0,1.]);ix=int(torch.floor((point[0]+2)/.5));iy=int(torch.floor((point[1]+2)/.5))
        assert y[0,0,ix,iy]==pytest.approx(1,abs=1e-6)


def test_fractional_batch_outside_and_mask():
    x=torch.ones(2,1,8,8);prev=pose().expand(2,4,4);cur=torch.cat([pose(.25),pose(10.)]);y,m,_=warp_bev(x,prev,cur,**KW)
    assert torch.all(y[m]==1) and not m[1].any() and not y[1].any()
    vm=torch.ones_like(x,dtype=torch.bool);vm[:,:,3,3]=False
    y,m,_=warp_bev(x,prev,prev,previous_valid=vm,**KW);assert not m[:,:,3,3].any() and not y[:,:,3,3].any()


def test_memory_resets_missing_scene_pose_time():
    m=BEVMemory(torch.ones(1,2,8,8),pose(),1.,'a',torch.ones(1,1,8,8,dtype=torch.bool))
    assert m.align(pose(),1.1,'a',**KW)[1]=='aligned'
    assert m.align(pose(),1.50000000745,'a',**KW)[1]=='aligned'
    assert m.align(pose(),1.501,'a',**KW)[0] is None
    assert m.align(pose(),1.1,'b',**KW)[1]=='scene_change'
    assert m.align(pose(),1.,'a',**KW)[1]=='timestamp_gap_or_reversal'
    assert m.align(pose(),2.,'a',**KW)[0] is None
    assert m.align(pose(9),1.1,'a',**KW)[1]=='localization_jump'
    p=pose();p[0,0,0]=2;assert m.align(p,1.1,'a',**KW)[0] is None
