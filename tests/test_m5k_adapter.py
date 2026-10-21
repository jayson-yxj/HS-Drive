import torch
import pytest
from models.m5k_depth_adapter import DepthResidual
from training.m5k_losses import losses

def test_zero_residual_same_capacity_and_no_gt_interface():
    a=DepthResidual();b=DepthResidual(mobile_condition=True)
    f=torch.randn(2,64,3,4);log=torch.randn(2,48,3,4);q=torch.rand(2,1,3,4)
    assert torch.equal(a(f,log,q),log) and torch.equal(b(f,log,q),log)
    assert sum(p.numel()for p in a.parameters())==sum(p.numel()for p in b.parameters())
    with pytest.raises(TypeError):a(f,log,q,gt_depth=torch.ones(2,3,4))

def test_unknown_zero_gradient_and_no_fake_spatial_candidate():
    log=torch.randn(1,48,2,2,requires_grad=True);depth=torch.ones(1,2,2)*4;valid=torch.zeros_like(depth,dtype=torch.bool);sp=torch.zeros_like(log,dtype=torch.bool);edges=torch.linspace(1.5,80,49)
    terms=losses(log,log.detach(),depth,valid,valid,sp,edges);sum(terms).backward();assert not log.grad.any()
    valid[:]=True;terms=losses(log,log.detach(),depth,valid,valid,sp,edges);assert terms[0]>0 and terms[1]==0
