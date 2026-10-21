import torch
from training.m5h_losses import teacher_mask,distillation_loss,near_kl

def test_teacher_conflict_unknown_and_cone_never_distilled():
    y=torch.tensor([True,True,True,False,True,True]);v=torch.tensor([True,True,True,True,False,True]);classes=torch.tensor([1,2,3,0,1,1]);q=torch.tensor([[.9,.1,.9,.9,.9,.5],[.1,.9,.9,.9,.9,.9]])
    score,m=teacher_mask(y,v,classes,q);assert m.tolist()==[True,True,False,False,False,False]
    log=torch.zeros(6,requires_grad=True);distillation_loss(log,score,m).backward();assert torch.equal(log.grad[~m],torch.zeros(4))

def test_near_kl_zero_at_original_and_no_far_gradient():
    old=torch.randn(1,48,2,2);new=old.clone().requires_grad_();m=torch.tensor([[[True,False],[False,False]]]);loss=near_kl(new,old,m);assert abs(float(loss.detach()))<1e-7
    changed=(old+torch.randn_like(old)).requires_grad_();near_kl(changed,old,m).backward();assert changed.grad[:,:,1].abs().sum()==0

def test_sampler_complete_deterministic_and_minimum_budget():
    from tools.m5h_sampling import schedule
    import pytest
    views=[dict(index=i,camera_index=0,tags=['a'])for i in range(7)]
    a,e=schedule(views,101,20);b,_=schedule(views,101,20);assert a==b;assert len(set(a[:7]))==7;assert all(x['updates']>=1 for x in e)
    with pytest.raises(ValueError):schedule(views,101,6)

def test_lowfpr_denominator_unknown_and_components():
    import numpy as np
    from diagnostics.m5h_metrics import enhanced
    shape=(1,4,4);valid=np.ones(shape,bool);valid[:,0]=False;target=np.zeros(shape,bool);target[0,1,0:3]=True;classes=target.astype(np.uint8)
    d=dict(target=target,valid=valid,classes=classes,actor=target.astype(np.int16),radial=np.ones(shape)*5)
    row=dict(group='g',configuration=0,camera_names=['front'],weather='Clear',sample_id='g/1',actors=[dict(actors=[])])
    p=np.zeros(shape);p[:,0]=1;p[0,1,0:3]=.9;p[0,3,0:3]=.9
    r=enhanced([row],[d],[p],.5)
    assert abs(r['confirmed_nontarget_fpr']-3/9)<1e-9
    assert r['false_foreground_per_million_valid_pixels']==250000
    assert r['false_detection_components'][0]['false_components']==1
    low=r['low_fpr_validation'];rr=enhanced([row],[d],[p],.5,locked_low=low);assert 'low_fpr_validation'not in rr

def test_depth_ce_ignores_unknown_even_with_outofrange_labels():
    from training.m5h_losses import depth_ce
    from datasets.labels import depth_bin_edges
    edges=torch.tensor(depth_bin_edges());x=torch.zeros(1,48,1,3,requires_grad=True);depth=torch.tensor([[[5.,1000.,0.]]]);valid=torch.tensor([[[True,False,False]]]);loss=depth_ce(x,depth,valid,edges);loss.backward();assert torch.isfinite(loss);assert x.grad[:,:,:,1:].abs().sum()==0

def test_teacher_empty_mask_has_finite_zero_gradient():
    x=torch.zeros(2,requires_grad=True);loss=distillation_loss(x,torch.ones(2),torch.zeros(2,dtype=torch.bool));loss.backward();assert loss==0;assert torch.equal(x.grad,torch.zeros(2))
