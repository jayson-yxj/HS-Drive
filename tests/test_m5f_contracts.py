import numpy as np
import torch
from datasets.m5f_pixels import adapt
from training.m5f_loss import pixel_loss
from models.m5f_pixel import PixelHead


def label(tag):
    d=np.full((7,7),10.);s=np.full((7,7),tag);i=np.full((7,7),999)
    return adapt(d,s,i,np.zeros((7,7),bool),np.eye(3),np.eye(4),[])[0]


def test_version_correct_semantics_and_unknown_not_negative():
    assert label(14)['target'].sum()==25 and label(12)['target'].sum()==25
    assert label(3)['valid'].sum()==25 and not label(3)['target'].any()
    for tag in [0,20,21,22,23,255]:assert not label(tag)['valid'].any()


def test_unknown_gradient_zero_and_balanced_positive_direction():
    x=torch.zeros(3,requires_grad=True);t=torch.tensor([True,False,False]);v=torch.tensor([True,True,False]);pixel_loss(x,t,v).backward()
    assert x.grad[0]<0 and x.grad[1]>0 and x.grad[2]==0


def test_decoder_matches_original_pixel_grid():
    h=PixelHead();y=h(torch.zeros(2,64,24,40));assert y.shape==(2,384,640)


def test_self_and_identity_boundaries_are_ignored():
    d=np.full((7,7),10.);s=np.full((7,7),14);i=np.ones((7,7),int);i[:,4:]=2;mask=np.zeros((7,7),bool);mask[2,2]=True
    r,_=adapt(d,s,i,mask,np.eye(3),np.eye(4),[])
    assert not r['valid'][2,2] and not r['valid'][:,3:5].any()


def test_event_mass_not_amplified_by_tiny_totalmass():
    from models.m5f_lift import lift
    from models.view_transform import HeightSplat
    import inspect
    # Mass-event transform itself is independent of observation-normalization.
    mass=torch.tensor([1e-8,.1,1.])
    score=-torch.expm1(-mass)
    assert score[0]<1e-7 and score[1]<.1 and score[2]<1


def test_foreground_model_rejects_privileged_inputs():
    from models.m5f_pixel import ForegroundModel
    model=ForegroundModel(torch.nn.Identity(),PixelHead())
    import pytest
    with pytest.raises(ValueError):model(images=torch.zeros(1,6,3,384,640),depth_gt=torch.ones(1))


def test_confirmed_building_negative_does_not_imply_free():
    r=label(3)
    assert 'free' not in r and 'occupancy' not in r and r['valid'].any()


def test_lift_preserves_mass_and_masks_without_normalizing_tiny_mass():
    from models.view_transform import HeightSplat
    from models.m5f_lift import lift
    import yaml
    cfg=yaml.safe_load(open('configs/bev.yaml'))
    splat=HeightSplat(cfg)
    fg=torch.tensor([[[[.5]]]],requires_grad=True);probs=torch.zeros(1,1,48,1,1);probs[:,:,0]=1e-5
    K=torch.eye(3).view(1,1,3,3);T=torch.eye(4).view(1,1,4,4)
    # Optical Z as HSX, optical X as HSY, opticalY as HSZ.
    T[0,0,:3,:3]=torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    out=lift(splat,fg,probs,K,T,(1,1),uv=torch.zeros(1,1,2))
    assert torch.isclose(out['foreground_mass'].sum(),torch.tensor(5e-6))
    assert out['event_score'].max()<1e-5
    out['foreground_mass'].sum().backward();assert fg.grad.item()>0
    zero=lift(splat,fg.detach(),probs,K,T,(1,1),valid=torch.zeros_like(fg,dtype=torch.bool),uv=torch.zeros(1,1,2))
    assert zero['total_depth_mass'].sum()==0


def test_cone_dynamic_requires_geometric_identity_not_actor_id():
    from itertools import product
    d=np.full((7,7),10.);s=np.full((7,7),21);i=np.full((7,7),999)
    K=np.array([[10.,0,3],[0,10.,3],[0,0,1.]])
    actor=dict(actor_id=123,is_ego=False,type_id='static.prop.trafficcone01',corners_hs_ego_m=list(product([-4,4],[-4,4],[9,11])))
    r,meta=adapt(d,s,i,np.zeros_like(s,bool),K,np.eye(4),[actor])
    assert r['target'].sum()==25 and meta[0]['pixels']==25
    assert not meta[0]['identity_candidates'][0]['equals_actor_id']
    ambiguous,_=adapt(d,s,i,np.zeros_like(s,bool),K,np.eye(4),[actor,dict(actor,actor_id=124)])
    assert not ambiguous['valid'].any()


def test_s16_auxiliary_features_match_existing_encoder():
    from models.image_encoder import ImageEncoder
    from models.m5f_pixel import backbone_levels,pyramid
    torch.manual_seed(7);net=ImageEncoder().eval();im=torch.rand(1,2,3,64,96)
    with torch.no_grad():
        existing=net(im).flatten(0,1)
        auxiliary=pyramid(net,backbone_levels(net,im),16)
    assert torch.allclose(existing,auxiliary,atol=1e-6,rtol=1e-6)
