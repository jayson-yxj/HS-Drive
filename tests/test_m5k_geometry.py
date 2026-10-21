import torch
import pytest
from models.m5k_geometry import resample_probability,sampling_masks

def test_center_mapping_and_probability_normalization():
    p=torch.zeros(1,1,2,2,2);p[:,:,0]=torch.tensor([[.2,.4],[.6,.8]]);p[:,:,1]=1-p[:,:,0]
    out=resample_probability(p,(4,4))
    assert torch.allclose(out.sum(2),torch.ones(1,1,4,4))
    assert out[0,0,0,1,1].item()==pytest.approx(.35)
    assert out[0,0,0,0,0].item()==pytest.approx(.2)

def test_budget_exact_including_empty_full_and_multiple_batches():
    q=torch.zeros(3,6,5,7);q[1,1,2,:]=1;q[2]=1
    adaptive,uniform=sampling_masks(q)
    assert torch.equal(adaptive.flatten(1).sum(1),uniform.flatten(1).sum(1))
    assert torch.equal(adaptive,q>=.9)
    assert uniform[1].sum()==7

def test_sampling_rejects_gt_keyword():
    with pytest.raises(TypeError):sampling_masks(torch.zeros(1,6,2,2),gt_mask=torch.ones(1,6,2,2))
