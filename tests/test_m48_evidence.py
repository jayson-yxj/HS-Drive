import numpy as np
import pytest
from datasets.m48_evidence import build_evidence,EvidenceConfig,validate_evidence
from geometry.ray_tracing import VoxelGrid

def test_same_coarse_cell_ground_and_free_can_coexist():
    g=VoxelGrid((0,2),(0,1),(0,1.5),.5)
    d,_=build_evidence([[.1,.2,1.2]],[[1.8,.2,1.2]],[[.7,.2,.1]],[1],grid=g)
    r,_=g.locate([.7,.2,.1]);assert d['surface_evidence'].flat[r] and d['free_ray_evidence'].flat[r]
    assert d['mixed_cell_mask'].flat[r] and d['surface_semantic'][0].flat[r]
    assert not d['surface_semantic'][1].flat[r]

def test_known_fine_surface_truncates_crossing_without_free_behind():
    g=VoxelGrid((0,4),(0,1),(0,1.5),.5)
    d,_=build_evidence([[.1,.2,.2]],[[3.8,.2,.2]],[[1.2,.2,.2]],[2],grid=g)
    r,_=g.locate([1.2,.2,.2]);behind,_=g.locate([2.2,.2,.2])
    assert d['evidence_conflict_mask'].flat[r]
    assert not d['free_evidence_valid'].flat[r] and not d['observability'].flat[behind]

def test_invalid_ego_and_half_open_boundaries():
    g=VoxelGrid((0,2),(0,1),(0,1.5),.5)
    d,_=build_evidence([[np.nan,0,0]],[[1,.2,.2]],[[2,.2,.2],[.2,.2,.2]],[2,1],grid=g,ego_box=(np.zeros(3),np.array([.5,.5,.5])))
    assert not d['observability'].any()
    with pytest.raises(ValueError):build_evidence([],[],[],[],grid=g,config=EvidenceConfig(fine_height_m=.7))

def test_unknown_is_not_a_negative_and_mixed_has_independent_gradients():
    torch=pytest.importorskip('torch')
    from training.m48_evidence_loss import evidence_loss
    shape=(1,4,2,2);mask=torch.zeros(shape,dtype=torch.bool);mask[0,0,0,0]=True
    labels=dict(surface_evidence=mask,free_ray_evidence=mask,surface_valid=mask,free_evidence_valid=mask,
        surface_semantic=torch.stack([mask,torch.zeros_like(mask),torch.zeros_like(mask)],1),surface_semantic_valid=mask,obstacle_valid=mask)
    outs={k:torch.zeros((1,3,4,2,2) if k=='semantic' else shape,requires_grad=True) for k in ['surface','free','semantic']}
    loss,_=evidence_loss(outs,labels);loss.backward()
    for k in ['surface','free']:
        assert outs[k].grad[0,0,0,0]<0 and not outs[k].grad[~mask].any()
    assert outs['semantic'].grad[0,1,0,0,0]>0  # ground is negative obstacle


def test_semantic_coexistence_across_fine_heights_and_ambiguity_within_one():
    g=VoxelGrid((0,2),(0,1),(0,1.5),.5)
    d,_=build_evidence([],[],[[.7,.2,.1],[.7,.2,1.1]],[1,2],grid=g)
    r,_=g.locate([.7,.2,.1])
    assert d['surface_semantic'][0].flat[r] and d['surface_semantic'][1].flat[r]
    assert d['obstacle_valid'].flat[r]  # observed both, no forced ground/obstacle exclusive class
    d,_=build_evidence([],[],[[.7,.2,.1],[.7,.2,.12]],[1,2],grid=g)
    assert d['semantic_conflict_mask'].flat[r] and not d['obstacle_valid'].flat[r]


def test_unreliable_ray_from_outside_grid_does_not_carve():
    g=VoxelGrid((0,2),(0,1),(0,1.5),.5)
    d,_=build_evidence([[-2,.2,.2]],[[1.8,.2,.2]],[],[],grid=g,reliable=[False])
    assert not d['free_ray_evidence'].any()
    d,_=build_evidence([[-2,.2,.2]],[[1.8,.2,.2]],[[1.8,.2,.2]],[2],grid=g)
    assert d['free_ray_evidence'][0,0,0] and d['surface_evidence'][0,3,0]
