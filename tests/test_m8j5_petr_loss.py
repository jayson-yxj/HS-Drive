"""Run only in existing isolated PETR environment; all labels here synthetic."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from training.m8j5_ignore_loss import IgnoreAwarePETRLoss,Supervision,query_envelopes


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
    record=dict(evidence_type='SyntheticCUDA_OfficialPETRHeadLoss',device=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,records=RECORDS,model_optimizer_steps=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),negative_regions='SyntheticOnly_NotRealDataCertification')
    if os.environ.get('M8J5_TEST_REPORT'):
        with Path(os.environ['M8J5_TEST_REPORT']).open('x') as f:json.dump(record,f,indent=2)


def fixture(complete=False,zero_positive=False,velocity=True):
    from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
    dev='cuda'
    gt=torch.tensor([[0.,0,0,2,4,2,0,0,0]],device=dev)
    boxes=normalize_bbox(gt,None).repeat(4,1)
    boxes[:,0]=torch.tensor([.25,10,20,30],device=dev)
    boxes.requires_grad_()
    cls=torch.zeros((4,10),device=dev,requires_grad=True)
    cert=dict(status='Verified',class_ids=list(range(10)),reviewer='SYNTHETIC_TEST',evidence_ref='SYNTHETIC_UNIT_GEOMETRY',
              scope='CompleteAnnotation' if complete else 'BoundedRegion')
    spec=Supervision(gt[:0] if zero_positive else gt,torch.zeros(0 if zero_positive else 1,device=dev,dtype=torch.long),
        torch.full((0 if zero_positive else 1,),velocity,device=dev,dtype=torch.bool),
        torch.tensor([[7.,-4,-3,13,4,3]],device=dev) if not complete else torch.empty((0,6),device=dev),
        torch.tensor([[16.,-4,-3,24,4,3]],device=dev) if not complete else torch.empty((0,6),device=dev),cert,complete)
    return cls,boxes,spec


def run(head,name,cls,box,spec):
    lc,lb,audit=IgnoreAwarePETRLoss(head).layer_loss(cls[None],box[None],[spec])
    (lc+lb).backward()
    RECORDS.append(dict(name=name,loss_cls=float(lc),loss_bbox=float(lb),audit=audit,
        class0_gradient=cls.grad[:,0].tolist(),class_gradient_abs_by_query=cls.grad.abs().sum(1).tolist(),
        regression_gradient_abs_by_query=box.grad.abs().sum(1).tolist()))
    return lc,lb,audit


def test_complete_annotations_match_original_loss_and_gradients(head):
    c,b,s=fixture(complete=True)
    c2=c.detach().clone().requires_grad_();b2=b.detach().clone().requires_grad_()
    original=head.loss_single(c2[None],b2[None],[s.boxes],[s.labels],None)
    sum(original).backward()
    lc,lb,a=run(head,'complete_annotation_parity',c,b,s)
    assert torch.allclose(lc,original[0],atol=1e-6) and torch.allclose(lb,original[1],atol=1e-6)
    assert torch.allclose(c.grad,c2.grad,atol=1e-6) and torch.allclose(b.grad,b2.grad,atol=1e-6)
    assert a['positive']==1 and a['negative']==3


def test_unknown_zero_gradient_safe_negative_gradient_and_positive_direction(head):
    c,b,s=fixture();lc,lb,a=run(head,'partial_labels',c,b,s)
    assert a['samples'][0]['state']==[1,-1,0,-1]
    assert c.grad[0,0]<0 and c.grad[0,1]>0 and b.grad[0,0]>0
    assert c.grad[1].abs().sum()==0 and c.grad[3].abs().sum()==0
    assert (c.grad[2]>0).all() and b.grad[1:].abs().sum()==0


def test_missing_target_inside_safe_region_is_vetoed(head):
    c,b,s=fixture();s.safe_aabbs=torch.tensor([[-5.,-5,-5,40,5,5]],device='cuda')
    _,_,a=run(head,'unknown_overrides_safe_background',c,b,s)
    assert a['samples'][0]['state']==[1,-1,0,0]
    assert c.grad[1].abs().sum()==0 and (c.grad[2]>0).all()


def test_valid_positive_inside_unknown_is_not_suppressed(head):
    c,b,s=fixture();s.unknown_aabbs=torch.tensor([[-5.,-5,-5,5,5,5]],device='cuda')
    _,_,a=run(head,'positive_precedence',c,b,s)
    assert a['samples'][0]['state'][0]==1 and c.grad[0,0]<0 and b.grad[0].abs().sum()>0


def test_all_ignore_zero_finite_gradients(head):
    c,b,s=fixture(zero_positive=True);s.safe_aabbs=s.safe_aabbs[:0]
    lc,lb,a=run(head,'all_ignore',c,b,s)
    assert lc==0 and lb==0 and torch.isfinite(c.grad).all() and torch.isfinite(b.grad).all()
    assert c.grad.abs().sum()==0 and b.grad.abs().sum()==0 and a['ignore']==4
    assert a['cls_avg_factor']==a['reg_avg_factor']==1


def test_zero_positive_certified_background_still_learns(head):
    c,b,s=fixture(zero_positive=True);lc,lb,a=run(head,'zero_positive_safe_background',c,b,s)
    assert lc>0 and lb==0 and a['negative']==1 and (c.grad[2]>0).all()
    assert b.grad.abs().sum()==0


def test_unknown_velocity_has_no_regression_gradient(head):
    c,b,s=fixture(velocity=False)
    with torch.no_grad(): b[0,8:10]=3
    run(head,'velocity_unknown',c,b,s)
    assert b.grad[:,8:10].abs().sum()==0 and b.grad[0,:8].abs().sum()>0


def test_all_decoder_layers_and_official_auxiliary_parity(head):
    c,b,s=fixture(complete=True)
    cs=c.detach()[None,None].repeat(6,1,1,1).requires_grad_();bs=b.detach()[None,None].repeat(6,1,1,1).requires_grad_()
    cs2=cs.detach().clone().requires_grad_();bs2=bs.detach().clone().requires_grad_()
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    bottom=s.boxes.clone();bottom[:,2]-=bottom[:,5]/2
    original=head.loss([LiDARInstance3DBoxes(bottom,box_dim=9)],[s.labels],
        dict(all_cls_scores=cs2,all_bbox_preds=bs2,enc_cls_scores=None,enc_bbox_preds=None))
    losses,audit=IgnoreAwarePETRLoss(head)(dict(all_cls_scores=cs,all_bbox_preds=bs),[s])
    assert set(losses)==set(original) and len(losses)==12
    sum(losses.values()).backward();sum(original.values()).backward()
    assert all(torch.allclose(losses[k],original[k],atol=1e-6) for k in losses)
    assert torch.allclose(cs.grad,cs2.grad,atol=1e-6) and torch.allclose(bs.grad,bs2.grad,atol=1e-6)
    # Same wrapper, partial labels, on each auxiliary layer.
    c,b,s=fixture();cs=c.detach()[None,None].repeat(6,1,1,1).requires_grad_();bs=b.detach()[None,None].repeat(6,1,1,1).requires_grad_()
    losses,audit=IgnoreAwarePETRLoss(head)(dict(all_cls_scores=cs,all_bbox_preds=bs),[s])
    sum(losses.values()).backward()
    assert all(a['samples'][0]['state']==[1,-1,0,-1] for a in audit)
    assert cs.grad[:,:,1].abs().sum()==0 and cs.grad[:,:,3].abs().sum()==0
    assert (cs.grad[:,:,2]>0).all() and (cs.grad[:,:,0,0]<0).all()
    assert bs.grad[:,:,1:].abs().sum()==0
    RECORDS.append(dict(name='six_layers_partial_and_full_parity',layers=6,loss_terms=12,audit=audit,
        unknown_gradient_abs=float(cs.grad[:,:,1].abs().sum()),negative_gradient_abs=float(cs.grad[:,:,2].abs().sum())))


def test_whole_envelope_required_confidence_not_used(head):
    # No Positive: isolate negative policy from the intentional score-dependent
    # Hungarian cost. A high-score query can legitimately win Positive matching.
    c,b,s=fixture(zero_positive=True)
    s.safe_aabbs=torch.tensor([[19.,-1,-1,21,1,1]],device='cuda')
    with torch.no_grad():c[2]=100
    _,_,a=run(head,'center_inside_but_footprint_outside',c,b,s)
    assert a['samples'][0]['state'][2]==-1 and c.grad[2].abs().sum()==0


def test_boundary_contact_and_rotated_envelope(head):
    c,b,s=fixture()
    bounds,_=query_envelopes(b)
    s.unknown_aabbs=torch.cat([bounds[2,3:4],bounds[2,1:3],bounds[2,3:4]+1,bounds[2,4:]])[None]
    _,_,a=run(head,'unknown_boundary_contact',c,b,s)
    assert a['samples'][0]['state'][2]==-1
    rotated=b.detach().clone();rotated[:,6]=1;rotated[:,7]=0
    assert torch.equal(query_envelopes(rotated)[0],bounds)


def test_invalid_boxes_certificates_and_nonfinite_fail_closed(head):
    c,b,s=fixture();s.boxes[0,3]=0
    with pytest.raises(ValueError):IgnoreAwarePETRLoss(head).layer_loss(c[None],b[None],[s])
    c,b,s=fixture();s.negative_certificate={}
    with pytest.raises(ValueError):IgnoreAwarePETRLoss(head).layer_loss(c[None],b[None],[s])
    c,b,s=fixture();s.negative_certificate['class_ids']=[0,8]
    with pytest.raises(ValueError):IgnoreAwarePETRLoss(head).layer_loss(c[None],b[None],[s])
    c,b,s=fixture()
    with torch.no_grad():b[0,0]=float('nan')
    with pytest.raises(ValueError):IgnoreAwarePETRLoss(head).layer_loss(c[None],b[None],[s])


def test_batch_mixed_supervision_and_large_dimension_no_negative(head):
    c,b,s=fixture();c0,b0,s0=fixture(zero_positive=True);s0.safe_aabbs=s0.safe_aabbs[:0]
    _,_,audit=IgnoreAwarePETRLoss(head).layer_loss(torch.stack([c,c0]),torch.stack([b,b0]),[s,s0])
    assert audit['positive']==1 and audit['negative']==1 and audit['ignore']==6
    with torch.no_grad():b[2,2]=1000
    _,_,audit=IgnoreAwarePETRLoss(head).layer_loss(c[None],b[None],[s])
    assert audit['samples'][0]['state'][2]==-1


def test_no_real_negative_certificate_disables_background(head):
    c,b,s=fixture();s.safe_aabbs=s.safe_aabbs[:0];s.negative_certificate={}
    _,_,a=run(head,'no_certified_real_negatives_policy',c,b,s)
    assert a['positive']==1 and a['negative']==0 and a['ignore']==3


def test_two_stage_not_silently_ignored(head):
    c,b,s=fixture()
    with pytest.raises(ValueError):IgnoreAwarePETRLoss(head)(dict(all_cls_scores=c[None,None],all_bbox_preds=b[None,None],enc_cls_scores=c),[s])


def test_real_unsigned_adapter_and_coordinate_corner_contract(head):
    from datasets.m8j5_loss_adapter import build_reviewed_supervision
    from datasets.m8j5_supervision import hs_to_petr_center
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from datasets.m8b_petr_adapter import corners_hs
    root=Path(os.environ['M8J5_PRIVATE'])
    pack=json.loads((root/'golden_candidates.json').read_text())
    frame=json.loads((root/'trust_sidecar/00107.json').read_text())
    s=build_reviewed_supervision(frame,pack['candidates'],[])
    assert len(s.boxes)==0 and len(s.safe_aabbs)==0 and len(s.unknown_aabbs)>0
    assert not s.complete_annotation
    c,b,_=fixture();_,_,a=run(head,'actual_unsigned_sidecar_policy_synthetic_predictions',c,b,s)
    assert a['positive']==a['negative']==0 and a['ignore']==4
    # Actual candidate boxes: algebra and official corner convention, no update.
    maxerr=0.
    for candidate in pack['candidates']:
        obj=candidate['object'];t=torch.tensor(hs_to_petr_center(obj))[None]
        t[:,2]-=t[:,5]/2
        official=LiDARInstance3DBoxes(t,box_dim=9).corners.numpy()[0]
        hs=[*obj['center_hs_m'],*obj['dimensions_lwh_m'],obj['yaw_ccw_rad']]
        import numpy as np
        ref=corners_hs(hs)
        error=float(np.linalg.norm(official[:,None]-ref[None],axis=-1).min(1).max())
        maxerr=max(maxerr,error)
    assert maxerr<1e-4
    RECORDS.append(dict(name='actual_candidate_yaw_box_roundtrip',count=len(pack['candidates']),
        max_corner_error_m=maxerr,reference='YawOnlyBox_NotTiltedOriginalCorners',updates=0))


def test_synthetic_signed_adapter_positive_and_tamper_rejection(head):
    from datasets.m8j5_loss_adapter import build_reviewed_supervision
    from datasets.m8j5_supervision import digest
    obj=dict(geometry_status='Valid',model_class='car',yaw_only_comparable=True,reason='SYNTHETIC',
        center_hs_m=[0,0,0],dimensions_lwh_m=[4,2,2],yaw_ccw_rad=-1.5707963267948966)
    candidate=dict(candidate_id='SYNTHETIC_ONLY',ordinal='SYNTHETIC',object=obj,
        trust=dict(geometry_validity='Valid',class_validity='Valid',PETR_common_region=True))
    candidate['candidate_sha256']=digest(candidate)
    response=dict(candidate_id='SYNTHETIC_ONLY',candidate_sha256=candidate['candidate_sha256'],decision='ApprovePositive',
        reviewer='SYNTHETIC_TEST',reviewed_at='2026-10-26T00:00:00Z',review_source='SYNTHETIC_NOT_ACTUAL_HUMAN',
        geometry_confirmed=True,class_confirmed=True,identity_confirmed=True,visible_confirmed=True,notes='')
    frame=dict(ordinal='SYNTHETIC',objects=[dict(candidate_id='SYNTHETIC_ONLY',object=copy.deepcopy(obj))])
    s=build_reviewed_supervision(frame,[candidate],[response])
    c,b,_=fixture();_,_,a=run(head,'synthetic_signed_adapter_only',c,b,s)
    assert a['positive']==1 and a['negative']==0 and not s.velocity_valid.any()
    frame['objects'][0]['object']['center_hs_m']=[5,0,0]
    with pytest.raises(ValueError):build_reviewed_supervision(frame,[candidate],[response])
