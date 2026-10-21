import torch
from training.m5i_transfer import masks,output_kd,feature_kd,feature_regions
from models.m5i_visual import align_teacher,MobilePixel

def test_kd_never_relabels_missed_gt_cone_unknown_or_conflict():
    y=torch.tensor([True,True,True,False,False,False]);v=torch.tensor([True,True,True,True,True,False]);c=torch.tensor([1,2,3,0,0,0]);s=torch.tensor([[.9,0.,0.,0.,.9,0.],[0.,0.,0.,0.,0.,0.]]);q,p,n=masks(y,v,c,s,s);assert p.tolist()==[True,False,False,False,False,False];assert n.tolist()==[False,False,False,True,False,False];x=torch.zeros(6,requires_grad=True);output_kd(x,q,p,n).backward();assert x.grad[1:3].abs().sum()==0;assert x.grad[4:].abs().sum()==0

def test_teacher_alignment_ramp_uses_actual_resize_not_padding():
    f=torch.arange(168).float()[None,None,None,:].expand(1,1,100,168);a=align_teacher(f,(384,640),(799,1333));expected=((torch.arange(80)*8+.5)*1333/640-.5)/8;assert torch.allclose(a[0,0,5],expected,atol=3e-5)

def test_feature_mask_and_empty_gradients():
    t=torch.zeros(1,16,16,dtype=torch.bool);v=torch.zeros_like(t);c=torch.zeros_like(t,dtype=torch.int64);r=feature_regions(t,v,c);assert not any(m.any()for m in r);x=torch.randn(1,4,2,2,requires_grad=True);loss=feature_kd(x,x.detach(),r);loss.backward();assert x.grad.abs().sum()==0

def test_mobile_true_p8_and_rgb_contract():
    import pytest
    m=MobilePixel().eval()
    with torch.no_grad():l=m.levels(torch.zeros(1,6,3,64,64));assert [tuple(x.shape[1:])for x in l]==[(40,8,8),(112,4,4),(960,2,2)];assert m.head(m.fuse(l),(64,64)).shape==(6,64,64)
    with pytest.raises(ValueError):m(images=torch.zeros(1,6,3,64,64),depth=torch.zeros(1))

def test_feature_transfer_has_finite_nonzero_student_gradient():
    torch.manual_seed(19)
    x=torch.randn(1,4,3,3,requires_grad=True);teacher=torch.randn_like(x)
    foreground=torch.zeros(1,3,3,dtype=torch.bool);foreground[:,1,1]=True
    background=~foreground;edge=torch.zeros_like(foreground)
    loss=feature_kd(x,teacher,[foreground,background,edge]);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum()>0


def test_reliable_background_kd_pushes_foreground_score_down():
    logit=torch.tensor([0.,0.],requires_grad=True)
    loss=output_kd(logit,torch.tensor([.9,.02]),torch.tensor([True,False]),torch.tensor([False,True]))
    loss.backward()
    assert logit.grad[0]<0 and logit.grad[1]>0
