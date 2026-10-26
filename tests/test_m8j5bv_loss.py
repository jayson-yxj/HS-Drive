"""Core official-PETR synthetic tests. No private data/Checkpoint access needed."""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from training.m8j5b_coverage_loss import CoverageSupervision,integral_mask
from training.m8j5bv_validated_loss import ValidatedCoveragePETRLoss

SEED=850541
RECORDS=[]


def state_hash(head):
    h=hashlib.sha256()
    for name,value in head.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope='module')
def head():
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_head
    torch.manual_seed(SEED);np.random.seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    runtime=Path(os.environ['M8J5_RUNTIME'])
    cfg=mmcv.Config.fromfile(str(runtime/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    h=build_head(dict(cfg.model.pts_bbox_head,train_cfg=cfg.model.train_cfg.pts)).cuda().eval()
    h.init_weights();before=state_hash(h)
    yield h
    after=state_hash(h);assert after==before
    path=os.environ.get('M8J5BV_REPORT')
    if path:
        with Path(path).open('x') as f:json.dump(dict(seed=SEED,records=RECORDS,
            model_parameters_before_sha256=before,model_parameters_after_sha256=after,
            parameter_updates=0,private_payload_used=False,atol=1e-6,rtol=1e-5,
            GPU=torch.cuda.get_device_name(),torch=torch.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated()),f,indent=2)


def fixture(complete=False,zero=False,queries=4):
    from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
    gt=torch.tensor([[0.,0,20,2,4,2,0,0,0]],device='cuda')
    box=normalize_bbox(gt,None).repeat(queries,1);box[:,0]=torch.arange(queries,device='cuda')*10.;box[0,0]=.25
    box.requires_grad_();cls=torch.zeros((queries,10),device='cuda',requires_grad=True)
    mask=torch.zeros((1,128,128),device='cuda',dtype=torch.bool)
    if not complete:mask[:,56:73,47:58]=True
    matrix=torch.tensor([[[40.,0,32,0],[0,40,64,0],[0,0,1,0]]],device='cuda')
    cert=dict(status='Verified',class_ids=list(range(10)),reviewer='SYNTHETIC',evidence_ref='ExactSyntheticScene_NotRealCertificate',
              scope='CompleteAnnotation' if complete else 'BoundedRegion')
    s=CoverageSupervision(gt[:0] if zero else gt,torch.zeros(0 if zero else 1,device='cuda',dtype=torch.long),
        torch.ones(0 if zero else 1,device='cuda',dtype=torch.bool),torch.empty((0,6),device='cuda'),
        torch.empty((0,6),device='cuda') if complete else torch.tensor([[16.,-4,16,24,4,24]],device='cuda'),
        cert,complete,matrix,integral_mask(mask),True,'SyntheticExactCoverage')
    return cls,box,s


def as_layers(c,b,layers=6):
    return dict(all_cls_scores=c.detach()[None,None].repeat(layers,1,1,1).requires_grad_(),
        all_bbox_preds=b.detach()[None,None].repeat(layers,1,1,1).requires_grad_(),enc_cls_scores=None,enc_bbox_preds=None)


def official_loss(head,pred,specs):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    gt=[]
    for s in specs:
        bottom=s.boxes.clone();bottom[:,2]-=bottom[:,5]/2;gt.append(LiDARInstance3DBoxes(bottom,box_dim=9))
    return head.loss(gt,[s.labels for s in specs],pred)


def assert_close(a,b):
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5),float((a-b).abs().max())


@pytest.mark.parametrize('batch',[1,2])
def test_seeded_full_matching_losses_gradients_and_disabled_mode(head,batch):
    torch.manual_seed(SEED);c,b,s=fixture(True,queries=12)
    c=torch.randn_like(c)*.3
    p=as_layers(c,b);p={k:v.repeat(1,batch,1,1).detach().requires_grad_() if isinstance(v,torch.Tensor) else v for k,v in p.items()}
    q={k:v.detach().clone().requires_grad_() if isinstance(v,torch.Tensor) else v for k,v in p.items()}
    specs=[s]*batch
    losses,a=ValidatedCoveragePETRLoss(head)(p,specs);original=official_loss(head,q,specs)
    sum(losses.values()).backward();sum(original.values()).backward()
    for k in original:assert_close(losses[k],original[k])
    for k in ('all_cls_scores','all_bbox_preds'):assert_close(p[k].grad,q[k].grad)
    for layer,record in enumerate(a):
        for bi,sample in enumerate(record['samples']):
            targets=head._get_target_single(q['all_cls_scores'][layer,bi],q['all_bbox_preds'][layer,bi],s.labels,s.boxes,None)
            official_assignment=head.assigner.assign(q['all_bbox_preds'][layer,bi],q['all_cls_scores'][layer,bi],s.boxes,s.labels,None)
            assert sample['matched_gt_indices']==official_assignment.gt_inds.cpu().tolist()
            assert targets[-2].cpu().tolist()==[i for i,x in enumerate(sample['state']) if x==1]
    d={k:v.detach().clone().requires_grad_() if isinstance(v,torch.Tensor) else v for k,v in p.items()}
    disabled,record=ValidatedCoveragePETRLoss(head,coverage_enabled=False)(d,specs)
    sum(disabled.values()).backward()
    for k in original:assert_close(disabled[k],original[k])
    assert_close(d['all_cls_scores'].grad,q['all_cls_scores'].grad);assert_close(d['all_bbox_preds'].grad,q['all_bbox_preds'].grad)
    RECORDS.append(dict(name='complete_and_disabled_parity',batch=batch,matching_exact=True,
        max_loss_abs_error=max(float(abs(losses[k]-original[k])) for k in original),
        max_class_grad_abs_error=float((p['all_cls_scores'].grad-q['all_cls_scores'].grad).abs().max()),
        max_box_grad_abs_error=float((p['all_bbox_preds'].grad-q['all_bbox_preds'].grad).abs().max()),layers=a))


def test_cannot_disable_on_unknown_real_like_labels(head):
    c,b,s=fixture();p=as_layers(c,b)
    with pytest.raises(ValueError,match='Cannot disable'):ValidatedCoveragePETRLoss(head,False)(p,[s])


@pytest.mark.parametrize('batch',[1,2])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
@pytest.mark.parametrize('case',['positive_only','safe_negative_only','all_ignore','empty_unknown','empty_safe','positive_ignore_overlap','out_of_view','fully_occluded','two_camera_overlap'])
def test_edge_loss_gradient_counts_every_layer(head,case,dtype,batch):
    c,b,s=fixture(zero=case in ('safe_negative_only','all_ignore','empty_unknown','empty_safe','out_of_view','fully_occluded','two_camera_overlap'))
    expected=[1,-1,0,-1]
    if case in ('positive_only','positive_ignore_overlap'):
        s.frame_negative_enabled=False;expected=[1,-1,-1,-1]
        if case=='positive_ignore_overlap':s.ignore_integral=integral_mask(torch.ones((1,128,128),device='cuda',dtype=torch.bool))
    elif case in ('all_ignore','empty_unknown'):
        s.frame_negative_enabled=False;expected=[-1]*4
    elif case in ('safe_negative_only','empty_safe'):expected=[-1,-1,0,-1]
    elif case=='out_of_view':
        b=b.detach().clone();b[:,4]=-20;b.requires_grad_();expected=[-1]*4
    elif case=='fully_occluded':
        # Image has no evidence about behind-surface object; absence never
        # enables negatives. Conservative all-view unknown supplied by scene.
        s.ignore_integral=integral_mask(torch.ones((1,128,128),device='cuda',dtype=torch.bool));expected=[-1]*4
    elif case=='two_camera_overlap':
        s.image_projection=s.image_projection.repeat(2,1,1)
        masks=torch.zeros((2,128,128),device='cuda',dtype=torch.bool);masks[1,50:80,60:85]=True
        s.ignore_integral=integral_mask(masks);expected=[-1]*4
    p=as_layers(c,b)
    p={k:v.repeat(1,batch,1,1).to(dtype).detach().requires_grad_() if isinstance(v,torch.Tensor) else v for k,v in p.items()}
    with torch.autocast('cuda',enabled=dtype==torch.float16):
        losses,a=ValidatedCoveragePETRLoss(head)(p,[s]*batch)
    scale=512 if dtype==torch.float16 else 1
    (sum(losses.values())*scale).backward()
    state=torch.tensor(expected,device='cuda');cg=p['all_cls_scores'].grad;bg=p['all_bbox_preds'].grad
    assert torch.isfinite(cg).all() and torch.isfinite(bg).all()
    assert cg[:,:,state==-1].abs().sum()==0 and bg[:,:,state!=1].abs().sum()==0
    if (state==0).any():assert (cg[:,:,state==0]>0).all()
    if (state==1).any():assert (cg[:,:,state==1,0]<0).all() and bg[:,:,state==1].abs().sum()>0
    assert all(sample['state']==expected for x in a for sample in x['samples'])
    assert all(x['classification_weight_sum']==batch*int((state!=-1).sum()) for x in a)
    expected_p=batch*int((state==1).sum());expected_n=batch*int((state==0).sum())
    assert all(x['positive']==expected_p and x['negative']==expected_n and
               x['ignore']==batch*int((state==-1).sum()) and
               x['cls_avg_factor']==max(1,expected_p+head.bg_cls_weight*expected_n) and
               x['reg_avg_factor']==max(1,expected_p) for x in a)
    assert all(abs(x['classification_loss']-x['classification_loss_positive']-x['classification_loss_negative'])<1e-5 for x in a)
    RECORDS.append(dict(name=case,dtype=str(dtype),batch=batch,scale=scale,layers=a,
        positive_class_grad_min=float(cg[:,:,state==1,0].float().min()/scale) if (state==1).any() else None,
        negative_class_grad_min=float(cg[:,:,state==0].float().min()/scale) if (state==0).any() else None,
        negative_class_grad_max=float(cg[:,:,state==0].float().max()/scale) if (state==0).any() else None,
        positive_regression_grad_max=float(bg[:,:,state==1].float().abs().max()/scale) if (state==1).any() else None,
        nonpositive_regression_grad_max=float(bg[:,:,state!=1].float().abs().max()/scale),
        max_direct_ignored_grad=float(cg[:,:,state==-1].abs().max()) if (state==-1).any() else 0.))


def test_unknown_query_shared_parameters_can_have_nonzero_grad(head):
    torch.manual_seed(SEED);c,b,s=fixture();linear=torch.nn.Linear(3,10).cuda()
    x=torch.randn(4,3,device='cuda');c=linear(x);c.retain_grad()
    lc,lb,a=ValidatedCoveragePETRLoss(head).layer_loss(c[None],b[None],[s]);(lc+lb).backward()
    assert c.grad[1].abs().sum()==0 and c.grad[3].abs().sum()==0
    assert linear.weight.grad.abs().sum()>0
    RECORDS.append(dict(name='query_zero_not_shared_parameter_zero',ignored_logits_grad=0.,shared_parameter_grad_norm=float(linear.weight.grad.norm())))


@pytest.mark.parametrize('class_id',[0,8])
@pytest.mark.parametrize('distance',[5.,20.,45.,51.199])
def test_classes_distances_identity_unknown_positive(head,class_id,distance):
    from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
    c,b,s=fixture();s.boxes[0,:3]=torch.tensor([distance,0,0],device='cuda');s.labels[:]=class_id
    s.frame_negative_enabled=False;s.image_projection=torch.tensor([[[64.,-60,0,0],[64,0,-60,0],[1,0,0,0]]],device='cuda')
    s.ignore_integral=integral_mask(torch.ones((1,128,128),device='cuda',dtype=torch.bool))
    b=normalize_bbox(s.boxes,None).repeat(4,1);b[:,1]=torch.tensor([.25,10,20,30],device='cuda');b.requires_grad_()
    p=as_layers(c,b);losses,a=ValidatedCoveragePETRLoss(head)(p,[s]);sum(losses.values()).backward()
    for layer,record in enumerate(a):
        sample=record['samples'][0];idx=sample['state'].index(1)
        assert sample['positive_class_ids']==[class_id]
        assert p['all_cls_scores'].grad[layer,0,idx,class_id]<0
        assert p['all_bbox_preds'].grad[layer,0,idx,:8].abs().sum()>0
    RECORDS.append(dict(name='class_distance_positive',class_id=class_id,distance_m=distance,identity='Unknown_NotRequired',all_layers_positive=True))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_autocast_loss_boundary_scaled_gradients(head,dtype):
    c,b,s=fixture();c=c.detach().to(dtype).requires_grad_();b=b.detach().to(dtype).requires_grad_()
    p=as_layers(c,b);ref={k:v.detach().float().requires_grad_() if isinstance(v,torch.Tensor) else v for k,v in p.items()}
    with torch.autocast('cuda',dtype=dtype):
        losses,a=ValidatedCoveragePETRLoss(head)(p,[s])
    baseline,ar=ValidatedCoveragePETRLoss(head)(ref,[s])
    for k in losses:assert losses[k].dtype==torch.float32;assert_close(losses[k],baseline[k])
    # Official FP16 scale512; BF16 checked additionally with same scale.
    (sum(losses.values())*512).backward();sum(baseline.values()).backward()
    for key in ('all_cls_scores','all_bbox_preds'):
        grad=p[key].grad.float()/512;expected=ref[key].grad
        assert torch.isfinite(grad).all()
        tol=.004 if dtype==torch.bfloat16 else .0005
        assert torch.allclose(grad,expected,atol=tol,rtol=tol)
    assert p['all_cls_scores'].grad[:,:,1].abs().sum()==0
    assert all(x['samples'][0]['state']==[1,-1,0,-1] for x in a)
    RECORDS.append(dict(name='mixed_precision_loss_boundary',dtype=str(dtype),scale=512,layers=a,
        max_class_grad_abs_error=float((p['all_cls_scores'].grad.float()/512-ref['all_cls_scores'].grad).abs().max()),
        gradient_atol=tol,gradient_rtol=tol,exact_masked_zero=True))


def test_weight_balance_nonzero_background_factor(head):
    c,b,s=fixture();original=head.bg_cls_weight;head.bg_cls_weight=.5
    try:
        lc,lb,a=ValidatedCoveragePETRLoss(head).layer_loss(c[None],b[None],[s]);(lc+lb).backward()
        assert a['cls_avg_factor']==1.5 and a['reg_avg_factor']==1
        assert a['classification_weight_sum']==2 and a['effective_fraction']==.5
        assert a['classification_loss_positive']>0 and a['classification_loss_negative']>0
        RECORDS.append(dict(name='negative_weight_normalization',audit=a))
    finally:head.bg_cls_weight=original


def test_official_head_synthetic_six_camera_forward_and_backward_parity(head):
    # Real official six-layer, 900-query head; synthetic features only, no image
    # backbone or real GT. Same parameters/forward graph used for both losses.
    torch.manual_seed(SEED);head.zero_grad(set_to_none=True)
    features=torch.randn((1,6,2048,2,3),device='cuda',requires_grad=True)*.03
    features.retain_grad()
    mat=np.array([[48.,-50,0,0],[32,0,-50,0],[1,0,0,0],[0,0,0,1]],dtype=np.float32)
    meta=dict(pad_shape=[(64,96,3)]*6,img_shape=[(64,96,3)]*6,lidar2img=[mat.copy() for _ in range(6)])
    pred=head([features],[meta])
    _,_,s=fixture(True);s.boxes[0,:3]=torch.tensor([10.,0,0],device='cuda')
    # Official source force_fp32 is conditional on fp16_enabled. Pass explicit
    # same float outputs to reference, matching the new precision contract.
    floatpred={k:v.float() if isinstance(v,torch.Tensor) else v for k,v in pred.items()}
    original=official_loss(head,floatpred,[s])
    losses,a=ValidatedCoveragePETRLoss(head)(pred,[s])
    for k in original:assert_close(losses[k],original[k])
    parameters=[features,head.reference_points.weight,head.cls_branches[-1][-1].weight]
    g0=torch.autograd.grad(sum(original.values()),parameters,retain_graph=True)
    g1=torch.autograd.grad(sum(losses.values()),parameters,retain_graph=True)
    for x,y in zip(g0,g1):assert_close(x,y);assert torch.isfinite(x).all()
    scaled=torch.autograd.grad(sum(losses.values())*512,parameters)
    assert all(torch.isfinite(x).all() for x in scaled)
    for x,y in zip(scaled,g1):assert torch.allclose(x/512,y,atol=.002,rtol=.005)
    assert pred['all_cls_scores'].shape==(6,1,900,10)
    RECORDS.append(dict(name='official_head_forward_parameter_gradient_parity',dtype='torch.float32',seed=SEED,
        feature_shape=list(features.shape),head_queries=900,layers=6,
        model_scope='OfficialHead_SyntheticFeatures_NoRealDataOrBackbone',
        max_parameter_grad_difference=max(float((x-y).abs().max()) for x,y in zip(g0,g1)),
        static_scale512_finite=True,losses={k:float(v) for k,v in losses.items()}))


def test_official_feature_fp16_head_fp32_full_synthetic_model(head):
    import mmcv
    from mmcv.runner import wrap_fp16_model
    from mmdet3d.models import build_model
    torch.manual_seed(SEED)
    cfg=mmcv.Config.fromfile(str(Path(os.environ['M8J5_RUNTIME'])/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    assert cfg.optimizer_config.type=='Fp16OptimizerHook' and cfg.optimizer_config.loss_scale==512.
    cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda().eval()
    model.pts_bbox_head.init_weights();wrap_fp16_model(model);before=state_hash(model)
    image=torch.randn((1,6,3,64,96),device='cuda',requires_grad=True)
    mat=np.array([[48.,-50,0,0],[32,0,-50,0],[1,0,0,0],[0,0,0,1]],np.float32)
    meta=dict(pad_shape=[(64,96,3)]*6,img_shape=[(64,96,3)]*6,lidar2img=[mat.copy() for _ in range(6)])
    # Exactly the official forward/feature decorators: backbone AMP, returned
    # features FP32, Head outside autocast. Do not AMP-wrap the whole detector.
    seen=[]
    handle=model.img_backbone.register_forward_hook(lambda m,i,o:seen.append(str(o[0].dtype)))
    with torch.autocast('cuda',enabled=False):
        feat=model.extract_feat(image.clone(),[meta]);assert feat[0].dtype==torch.float32
        pred=model.pts_bbox_head(feat,[meta])
    handle.remove();assert seen==['torch.float16']
    _,_,s=fixture(True);s.boxes[0,:3]=torch.tensor([10.,0,0],device='cuda')
    original=official_loss(model.pts_bbox_head,pred,[s])
    losses,a=ValidatedCoveragePETRLoss(model.pts_bbox_head)(pred,[s])
    for k in original:assert_close(losses[k],original[k])
    outputs=[pred['all_cls_scores'],pred['all_bbox_preds']]
    direct_original=torch.autograd.grad(sum(original.values()),outputs,retain_graph=True)
    direct_new=torch.autograd.grad(sum(losses.values()),outputs,retain_graph=True)
    for x,y in zip(direct_original,direct_new):assert_close(x,y)
    params=[image,model.img_backbone.conv1.weight,model.pts_bbox_head.reference_points.weight]
    # Official backbone uses reentrant activation checkpointing (with_cp=True),
    # which supports backward(), not autograd.grad(inputs=...). Preserve it.
    model.zero_grad(set_to_none=True);image.grad=None
    (sum(original.values())*512).backward(retain_graph=True)
    assert all(p.grad is not None for p in params)
    g0=[p.grad.detach().clone() for p in params]
    model.zero_grad(set_to_none=True);image.grad=None
    (sum(original.values())*512).backward(retain_graph=True)
    grepeat=[p.grad.detach().clone() for p in params]
    model.zero_grad(set_to_none=True);image.grad=None
    (sum(losses.values())*512).backward()
    assert all(p.grad is not None for p in params)
    g1=[p.grad.detach().clone() for p in params]
    for x,y,z in zip(g0,g1,grepeat):assert torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(z).all()
    assert state_hash(model)==before
    RECORDS.append(dict(name='official_full_model_mixed_precision_synthetic',seed=SEED,
        input_shape=[1,6,3,64,96],backbone_output=seen[0],head_input=str(feat[0].dtype),
        output_dtype=str(pred['all_cls_scores'].dtype),loss_scale=512.,all_requested_gradients_finite=True,
        max_unscaled_grad_difference=max(float((x-y).abs().max()/512) for x,y in zip(g0,g1)),
        max_direct_output_grad_difference=max(float((x-y).abs().max()) for x,y in zip(direct_original,direct_new)),
        repeated_original_grad_difference=max(float((x-y).abs().max()/512) for x,y in zip(g0,grepeat)),
        per_parameter_grad_differences=[dict(name=n,original_repeat=float((x-z).abs().max()/512),coverage=float((x-y).abs().max()/512),
            gradient_cosine=float(torch.nn.functional.cosine_similarity(x.float().reshape(1,-1),y.float().reshape(1,-1))),
            original_repeat_meets_fp32_tolerance=bool(torch.allclose(x/512,z/512,atol=1e-6,rtol=1e-5)),
            coverage_meets_fp32_tolerance=bool(torch.allclose(x/512,y/512,atol=1e-6,rtol=1e-5)))
            for n,x,y,z in zip(['image','conv1.weight','reference_points.weight'],g0,g1,grepeat)],
        weights_before_sha256=before,weights_after_sha256=before,optimizer_steps=0,
        scope='ActualOfficialResNetDCNv2AndPETR_SyntheticSmallImages_NoCheckpoint_NoRealTraining'))
    for x,y in zip(g0,g1):assert_close(x/512,y/512)
