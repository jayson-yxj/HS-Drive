import pytest
import numpy as np
torch=pytest.importorskip('torch')
from diagnostics.splat_confidence import ConfidenceSplat,aggregate_sum
from models.view_transform import HeightSplat


def cfg():return dict(resolution_m=.5,x_range_m=[-2,4],y_range_m=[-2,2],z_edges_m=[-2,-.5,1,2.5,4],network=dict(depth_bins=4,depth_range_m=[.5,3.]))


def test_single_ray_amplitude_monotonic_empty_and_tau_limit():
    m=torch.tensor([[[[0.,1e-4,.01,1.]]]]);s=m[:,None]*2
    b=aggregate_sum(s,m);g=aggregate_sum(s,m,'gated',.01);su=aggregate_sum(s,m,'sum')
    assert torch.isfinite(g).all() and g.flatten()[0]==0
    assert torch.all(g.flatten().diff()>0)
    torch.testing.assert_close(su,m[:,None]*2)
    torch.testing.assert_close(aggregate_sum(s,m,'gated',1e-12),b,atol=1e-6,rtol=1e-6)
    assert b.flatten()[1]==2 and su.flatten()[1]<.001


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_chunks_gradients_default_precision_boundaries(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('No CUDA')
    if device=='cuda':
        from training.runtime import prepare_cuda
        prepare_cuda()
    torch.manual_seed(3);ctx=torch.randn(1,2,3,2,2,device=device,requires_grad=True)
    p=torch.softmax(torch.randn(1,2,4,2,2,device=device),2).requires_grad_()
    k=torch.eye(3,device=device).expand(1,2,3,3);t=torch.eye(4,device=device).expand(1,2,4,4)
    original=HeightSplat(cfg()).to(device);base=ConfidenceSplat(cfg()).to(device)
    a,m=original(ctx,p,k,t,(2,2));b,mb=base(ctx,p,k,t,(2,2));torch.testing.assert_close(a,b,rtol=0,atol=0);assert torch.equal(m,mb)
    for mode,tau in [('sum',None),('gated',.01)]:
        x=ConfidenceSplat(cfg(),1,mode,tau).to(device);y=ConfidenceSplat(cfg(),1000,mode,tau).to(device)
        h,mm=x(ctx,p,k,t,(2,2));hh,_=y(ctx,p,k,t,(2,2));torch.testing.assert_close(h,hh)
        torch.testing.assert_close(h,aggregate_sum(a*m[:,None].clamp_min(1e-6),m,mode,tau),atol=1e-5,rtol=1e-5)
        with torch.autocast(device,dtype=torch.bfloat16):ha,ma=x(ctx,p,k,t,(2,2))
        torch.testing.assert_close(ha,h);h.sum().backward(retain_graph=True)
    assert torch.isfinite(ctx.grad).all() and torch.isfinite(p.grad).all()
    pts=torch.tensor([[-2,-2,-2],[4,0,0],[0,2,0],[0,0,4],[0,0,-.5],[float('nan'),0,0]],device=device)
    _,valid=base.ranks(pts);assert valid.tolist()==[True,False,False,False,True,False]
    assert set(original.state_dict())==set(base.state_dict())


def test_multiple_rays_manual_sum_and_invalid_index_no_alias():
    # Both cameras observe exactly one valid bin at the same voxel. The second
    # experiment puts camera 2 at y_max, where flat ranks can alias next x.
    c=cfg();c['network']=dict(depth_bins=1,depth_range_m=[.5,1.])
    context=torch.tensor([2.,6.]).reshape(1,2,1,1,1)
    probability=torch.tensor([.2,.8]).reshape(1,2,1,1,1)
    k=torch.eye(3).expand(1,2,3,3);t=torch.eye(4).repeat(1,2,1,1)
    expected={'baseline':5.2,'sum':5.2,'gated':2.6}
    for mode in expected:
        splat=ConfidenceSplat(c,1,mode,1. if mode=='gated' else None)
        h,m=splat(context,probability,k,t,(1,1))
        assert m.sum().item()==pytest.approx(1.)
        assert h.sum().item()==pytest.approx(expected[mode])
    t[:,1,1,3]=2.
    h,m=ConfidenceSplat(c,mode='sum')(context,probability,k,t,(1,1))
    assert m.sum().item()==pytest.approx(.2)
    assert h.sum().item()==pytest.approx(.4)
