import numpy as np
import torch
import pytest
from datasets.m5e_visibility import planar_patch
from training.m5e_losses import spatial_loss,presence_only


@pytest.mark.parametrize('sx,sy',[(0,0),(.001,0),(0,.001),(.001,.0005)])
def test_perspective_planar_and_slanted_surfaces(sx,sy):
    y,x=np.mgrid[-2:3,-2:3];d=1/(1/35+sx*x+sy*y);tags=np.ones((5,5),int)
    r=planar_patch(d,tags,tags);assert r['candidate'] and r['residual_m']<1e-8


@pytest.mark.parametrize('kind',['step','occlusion','adjacent','pole','pedestrian','noise','nan'])
def test_boundary_identity_and_noise_rejected(kind):
    d=np.full((5,5),35.);s=np.ones((5,5),int);i=s.copy()
    if kind in ['step','occlusion']:d[:,3:]=20
    if kind=='adjacent':i[:,3:]=2
    if kind in ['pole','pedestrian']:d[:,2]=15
    if kind=='noise':d[2,2]+=.3
    if kind=='nan':d[2,2]=np.nan
    assert not planar_patch(d,s,i)['candidate']


def test_spatial_loss_moves_mass_to_tolerant_correct_layer_and_unknown_zero():
    l=torch.zeros(1,1,3,1,2,requires_grad=True)
    err=torch.tensor([.6,1.,10.]).view(1,1,3,1,1).expand_as(l)
    same=torch.ones_like(l,dtype=torch.bool);valid=torch.tensor([[[[True,False]]]])
    loss=spatial_loss(l,err,same,valid,torch.full((1,1,1,2),25.));loss.backward()
    assert l.grad[0,0,0,0,0]<0 and l.grad[0,0,2,0,0]>0
    assert l.grad[...,1].count_nonzero()==0
    assert torch.isfinite(l.grad).all()


def test_candidate_is_not_an_absence_target_and_positive_unknown_gradient_zero():
    l=torch.zeros(1,4,160,120,requires_grad=True);m=torch.zeros_like(l,dtype=torch.bool);m[0,2,90,60]=True
    presence_only(l,m).backward();assert l.grad.count_nonzero()==1 and l.grad[0,2,90,60]<0


def test_frozen_spatial_objective_cannot_change_context_gradient():
    logits=torch.zeros(1,1,3,1,1)
    xy=torch.tensor([.5,1.,10.]).view(1,1,3,1,1)
    sp=spatial_loss(logits,xy,torch.ones_like(logits,dtype=torch.bool),torch.ones(1,1,1,1,dtype=torch.bool),torch.full((1,1,1,1),20.))
    p=torch.tensor(2.,requires_grad=True)
    g0=torch.autograd.grad((p-1).square(),p)[0]
    g1=torch.autograd.grad((p-1).square()+.1*sp,p)[0]
    assert not sp.requires_grad and torch.equal(g0,g1)


def test_no_correct_height_bin_leaves_ray_unsupervised():
    l=torch.zeros(1,1,3,1,1,requires_grad=True)
    loss=spatial_loss(l,torch.ones_like(l),torch.zeros_like(l,dtype=torch.bool),torch.ones(1,1,1,1,dtype=torch.bool),torch.full((1,1,1,1),20.))
    loss.backward()
    assert loss==0 and l.grad.count_nonzero()==0
