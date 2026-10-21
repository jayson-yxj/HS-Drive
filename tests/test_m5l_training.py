import torch
from models.m5k_depth_adapter import DepthResidual
from training.m5k_losses import losses
from tools.m5l_adaptive import region_uniform

def test_matched_initialization_budget_and_unknown_gradient():
    torch.manual_seed(101);b=DepthResidual();torch.manual_seed(101);c=DepthResidual(mobile_condition=True)
    assert all(torch.equal(b.state_dict()[k],v)for k,v in c.state_dict().items())
    logits=torch.randn(1,48,3,4,requires_grad=True);valid=torch.zeros(1,3,4,dtype=torch.bool);depth=torch.ones(1,3,4)*20;space=torch.zeros_like(logits,dtype=torch.bool)
    loss=sum(losses(logits,logits.detach(),depth,valid,valid,space,torch.linspace(1.5,80,49)));loss.backward();assert not logits.grad.any()

def test_region_budget_empty_and_nonrectangular():
    roi=torch.zeros(1,6,8,8,dtype=torch.bool);roi[0,0,2:6,2]=True;roi[0,0,5,2:6]=True
    out=region_uniform(roi);assert out.sum()==roi.sum();assert not out[0,1:].any();assert not out[0,0,:2].any()
    assert not region_uniform(torch.zeros_like(roi)).any()
