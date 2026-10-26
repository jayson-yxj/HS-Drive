"""Actual frozen official PETR losses, synthetic CUDA gradients, zero updates."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from training.m8j5b_coverage_loss import CoverageSupervision,CoverageAwarePETRLoss,integral_mask,projected_ignore

RECORDS=[]


@pytest.fixture(scope='module')
def head():
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_head
    root=Path(os.environ['M8J5_RUNTIME'])
    cfg=mmcv.Config.fromfile(str(root/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    h=build_head(dict(cfg.model.pts_bbox_head,train_cfg=cfg.model.train_cfg.pts)).cuda().eval()
    yield h
    p=os.environ.get('M8J5B_TEST_REPORT')
    if p:
        with Path(p).open('x') as f:json.dump(dict(evidence='SyntheticCUDA_ActualOfficialPETRHead',
            GPU=torch.cuda.get_device_name(),torch=torch.__version__,records=RECORDS,optimizer_steps=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()),f,indent=2)


def fixture(complete=False,zero=False):
    from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
    gt=torch.tensor([[0.,0,20,2,4,2,0,0,0]],device='cuda')
    box=normalize_bbox(gt,None).repeat(4,1);box[:,0]=torch.tensor([.25,10,20,30],device='cuda');box.requires_grad_()
    cls=torch.zeros((4,10),device='cuda',requires_grad=True)
    mask=torch.zeros((1,128,128),device='cuda',dtype=torch.bool)
    if not complete:mask[:,56:73,47:58]=True
    matrix=torch.tensor([[[40.,0,32,0],[0,40,64,0],[0,0,1,0]]],device='cuda')
    cert=dict(status='Verified',class_ids=list(range(10)),reviewer='SYNTHETIC_TEST_ONLY',
        evidence_ref='SyntheticExactSceneGeometry',scope='CompleteAnnotation' if complete else 'BoundedRegion')
    spec=CoverageSupervision(gt[:0] if zero else gt,torch.zeros(0 if zero else 1,device='cuda',dtype=torch.long),
        torch.ones(0 if zero else 1,device='cuda',dtype=torch.bool),torch.empty((0,6),device='cuda'),
        torch.empty((0,6),device='cuda') if complete else torch.tensor([[16.,-4,16,24,4,24]],device='cuda'),
        cert,complete,matrix,integral_mask(mask),True,'SYNTHETIC_COVERAGE')
    return cls,box,spec


def run(head,name,c,b,s):
    lc,lb,a=CoverageAwarePETRLoss(head).layer_loss(c[None],b[None],[s]);(lc+lb).backward()
    RECORDS.append(dict(name=name,loss_cls=float(lc),loss_bbox=float(lb),audit=a,
        class0_gradient=c.grad[:,0].tolist(),classification_gradient_abs=c.grad.abs().sum(1).tolist(),
        regression_gradient_abs=b.grad.abs().sum(1).tolist()))
    return lc,lb,a


def test_complete_loss_and_all_gradients_equal_official(head):
    c,b,s=fixture(True);c0=c.detach().clone().requires_grad_();b0=b.detach().clone().requires_grad_()
    original=head.loss_single(c0[None],b0[None],[s.boxes],[s.labels]);sum(original).backward()
    lc,lb,a=run(head,'complete_parity',c,b,s)
    assert torch.allclose(lc,original[0],atol=1e-6) and torch.allclose(lb,original[1],atol=1e-6)
    assert torch.allclose(c.grad,c0.grad,atol=1e-6) and torch.allclose(b.grad,b0.grad,atol=1e-6)
    assert a['samples'][0]['state']==[1,0,0,0]


def test_image_unknown_negative_and_positive_gradient(head):
    c,b,s=fixture();s.safe_aabbs=torch.tensor([[-5.,-5,15,35,5,25]],device='cuda')
    lc,lb,a=run(head,'image_unknown_veto_safe_negative_positive',c,b,s)
    assert a['samples'][0]['state']==[1,-1,0,0]
    assert c.grad[1].abs().sum()==0 and b.grad[1:].abs().sum()==0
    assert c.grad[0,0]<0 and c.grad[0,1]>0 and b.grad[0,0]>0
    assert (c.grad[2]>0).all() and (c.grad[3]>0).all()


def test_positive_overlapping_ignore_is_preserved(head):
    c,b,s=fixture();mask=torch.ones((1,128,128),device='cuda',dtype=torch.bool);s.ignore_integral=integral_mask(mask)
    with pytest.warns(RuntimeWarning):lc,lb,a=run(head,'positive_precedence',c,b,s)
    assert a['samples'][0]['positive_in_image_veto']==1 and a['positive']==1
    assert c.grad[0,0]<0 and b.grad[0].abs().sum()>0 and c.grad[1:].abs().sum()==0


def test_all_ignore_zero_finite_gradients_warning(head):
    c,b,s=fixture(zero=True);s.frame_negative_enabled=False
    with pytest.warns(RuntimeWarning,match='HighIgnoreFraction.*NoNegative'):
        lc,lb,a=run(head,'all_ignore',c,b,s)
    assert lc==0 and lb==0 and torch.isfinite(c.grad).all() and torch.isfinite(b.grad).all()
    assert c.grad.abs().sum()==0 and b.grad.abs().sum()==0
    assert a['cls_avg_factor']==a['reg_avg_factor']==1


def test_zero_positive_keeps_certified_negative_gradients(head):
    c,b,s=fixture(zero=True);lc,lb,a=run(head,'zero_positive_safe_negative',c,b,s)
    assert a['samples'][0]['state']==[-1,-1,0,-1]
    assert (c.grad[2]>0).all() and b.grad.abs().sum()==0 and lc>0 and lb==0


def test_six_layers_ignore_and_official_full_parity(head):
    for complete in (True,False):
        c,b,s=fixture(complete);cs=c.detach()[None,None].repeat(6,1,1,1).requires_grad_();bs=b.detach()[None,None].repeat(6,1,1,1).requires_grad_()
        loss,a=CoverageAwarePETRLoss(head)(dict(all_cls_scores=cs,all_bbox_preds=bs),[s]);sum(loss.values()).backward()
        assert len(loss)==12 and len(a)==6
        if not complete:
            assert all(r['samples'][0]['state']==[1,-1,0,-1] for r in a)
            assert cs.grad[:,:,1].abs().sum()==0 and cs.grad[:,:,3].abs().sum()==0
            assert (cs.grad[:,:,2]>0).all() and (cs.grad[:,:,0,0]<0).all()
        else:
            from mmdet3d.core.bbox import LiDARInstance3DBoxes
            bottom=s.boxes.clone();bottom[:,2]-=bottom[:,5]/2
            cs0=cs.detach().clone().requires_grad_();bs0=bs.detach().clone().requires_grad_()
            orig=head.loss([LiDARInstance3DBoxes(bottom,box_dim=9)],[s.labels],
                dict(all_cls_scores=cs0,all_bbox_preds=bs0,enc_cls_scores=None,enc_bbox_preds=None))
            sum(orig.values()).backward()
            assert all(torch.allclose(loss[k],orig[k],atol=1e-6) for k in loss)
            assert torch.allclose(cs.grad,cs0.grad,atol=1e-6) and torch.allclose(bs.grad,bs0.grad,atol=1e-6)
        RECORDS.append(dict(name='six_layers_complete' if complete else 'six_layers_partial',audit=a,
            masked_gradient_abs=0. if complete else float(cs.grad[:,:,1].abs().sum())))


def test_frame_fallback_no_certificate_and_high_ignore_warning(head):
    c,b,s=fixture();s.safe_aabbs=s.safe_aabbs[:0];s.negative_certificate={};s.frame_negative_enabled=False
    with pytest.warns(RuntimeWarning,match='NoNegative'):
        _,_,a=run(head,'actual_policy_frame_fallback',c,b,s)
    assert a['samples'][0]['state']==[1,-1,-1,-1]


def test_no_projection_near_plane_and_bad_geometry_fail_closed(head):
    c,b,s=fixture(zero=True);p=b.detach().clone()
    p[0,4]=-20;p[1,4]=0;p[2,2]=1000
    veto,reason=projected_ignore(p,s)
    assert veto[:3].all() and reason['near_plane_ambiguous']>=1 and reason['invalid_envelope']>=1


def test_ignore_no_confidence_or_geometry_gradient_shortcut(head):
    c,b,s=fixture(zero=True);v,_=projected_ignore(b,s)
    with torch.no_grad():c[1]=100
    lc,lb,a=run(head,'high_score_unknown_still_zero_gradient',c,b,s)
    assert a['samples'][0]['state'][1]==-1 and c.grad[1].abs().sum()==0 and b.grad.abs().sum()==0
    assert not v.requires_grad


def test_multiview_any_unknown_veto(head):
    c,b,s=fixture(zero=True);s.image_projection=s.image_projection.repeat(2,1,1)
    masks=torch.zeros((2,128,128),device='cuda',dtype=torch.bool);masks[1,56:73,66:80]=True
    s.ignore_integral=integral_mask(masks)
    with pytest.warns(RuntimeWarning):_,_,a=run(head,'second_camera_veto',c,b,s)
    assert a['negative']==0 and c.grad.abs().sum()==0


def test_invalid_gt_and_false_complete_and_malformed_mask_rejected(head):
    c,b,s=fixture();s.boxes[0,3]=0
    with pytest.raises(ValueError):CoverageAwarePETRLoss(head).layer_loss(c[None],b[None],[s])
    c,b,s=fixture();s.complete_annotation=True;s.negative_certificate['scope']='CompleteAnnotation'
    with pytest.raises(ValueError):CoverageAwarePETRLoss(head).layer_loss(c[None],b[None],[s])
    c,b,s=fixture();s.ignore_integral[0,0,0]=1
    with pytest.raises(ValueError):CoverageAwarePETRLoss(head).layer_loss(c[None],b[None],[s])


def test_unknown_velocity_grad_zero_and_normalization_ignores_unknown(head):
    c,b,s=fixture();s.velocity_valid[:]=False
    with torch.no_grad():b[0,8:]=4
    original=head.bg_cls_weight;head.bg_cls_weight=.5
    try:
        _,_,a=run(head,'normalization_and_velocity',c,b,s)
        assert a['cls_avg_factor']==1.5 and a['reg_avg_factor']==1
        assert b.grad[:,8:].abs().sum()==0
    finally:head.bg_cls_weight=original


def test_actual_sidecar_not_golden_and_no_fake_box(head):
    from datasets.m8j5b_loss_adapter import load_coverage_supervision
    root=Path(os.environ['M8J5B_PRIVATE'])
    s,r=load_coverage_supervision(root,'00124')
    assert not (s.labels==8).any() and not s.frame_negative_enabled and not len(s.safe_aabbs)
    assert s.ignore_integral[:,-1,-1].sum()>0 and r['human_signoff']=='Pending_Suspended'
    assert all(o['positive']['golden_gt'] is False for o in r['objects'])


def test_actual_coverage_uses_exact_frozen_rgb_projection(head):
    import numpy as np
    from datasets.m8j4_petr_input import prepare
    from datasets.m8j5b_loss_adapter import load_coverage_supervision
    root=Path(os.environ['M8J5B_PRIVATE']);maxerr=0.
    for ordinal in ('00107','00124','00196','00213'):
        s,_=load_coverage_supervision(root,ordinal)
        with np.load(root/'inference_input'/(ordinal+'.npz'),allow_pickle=False) as z:
            _,meta,_=prepare(z['images'],z['intrinsics'],z['cam_to_ego'])
        maxerr=max(maxerr,float(abs(np.asarray(meta['lidar2img'])[:,:3,:]-s.image_projection.cpu().numpy()).max()))
        assert s.ignore_integral.shape==(6,513,1409)
    assert maxerr<.001
    RECORDS.append(dict(name='actual_four_rgb_calibration_parity',max_abs_matrix_error=maxerr))


def test_empty_query_and_mixed_batch_finite(head):
    c,b,s=fixture(zero=True)
    # A zero-query input is malformed for this 900-query network: explicit failure
    # is preferable to reshape/min reductions giving an uninformative crash.
    with pytest.raises(ValueError):CoverageAwarePETRLoss(head).layer_loss(c[:0][None],b[:0][None],[s])
    c0,b0,s0=fixture();s.frame_negative_enabled=False
    with pytest.warns(RuntimeWarning):
        lc,lb,a=CoverageAwarePETRLoss(head).layer_loss(torch.stack([c0,c]),torch.stack([b0,b]),[s0,s])
    assert torch.isfinite(lc+lb) and (a['positive'],a['negative'],a['ignore'])==(1,1,6)
