import numpy as np,torch
from diagnostics.m5l_objects import plane_intersection,locate,components,match
from models.m5l_p4 import sample_map,P4LocalDepth

def test_feature_centers_not_nominal_depth_centers():
    f=torch.arange(16.).reshape(1,1,4,4);uv=torch.tensor([[[4.,4.],[6.,6.]]]);v=sample_map(f,uv,stride=4,center=0)
    assert torch.allclose(v[...,0],torch.tensor([[5.,7.5]]))

def test_horizon_abstains_and_camera_ray_plane():
    T=np.eye(4);T[:3,:3]=np.array([[0,0,1],[-1,0,0],[0,-1,0]]);T[2,3]=1.5;K=np.array([[100,0,50],[0,100,50],[0,0,1.]])
    p,_=plane_intersection([50,60],K,T,0);assert np.allclose(p,[15,0,0])
    assert plane_intersection([50,50],K,T)[0]is None

def test_empty_and_component_no_truth_dependency():
    assert components(np.zeros((32,32)))==[]
    q=np.zeros((32,32));q[10:20,10:20]=1;c=components(q);pairs,fp,fn=match(c,[q.astype(bool)])
    assert len(pairs)==1 and not fp and not fn

def test_localdepth_zero_init():
    head=P4LocalDepth();p=torch.rand(2,10,48).softmax(-1);out=head(torch.rand(2,10,88),p)
    assert torch.allclose(out.softmax(-1),p,atol=1e-7)
