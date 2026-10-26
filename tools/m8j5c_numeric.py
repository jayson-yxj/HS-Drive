"""Paired PETR diagnostics reusing J5BV fixtures; synthetic data, no updates."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import time

import numpy as np
import torch

from training.m8j5bv_validated_loss import ValidatedCoveragePETRLoss

ROOT = Path(os.environ['M8J5C_ROOT'])
OUT = Path(os.environ['M8J5C_OUTPUT'])
PROTOCOL = json.loads(Path('configs/m8j5c_protocol.json').read_text())
REUSE = runpy.run_path('tests/test_m8j5bv_loss.py')
state_hash = REUSE['state_hash']
fixture = REUSE['fixture']
official_loss = REUSE['official_loss']
ATOL, RTOL = PROTOCOL['numeric']['atol'], PROTOCOL['numeric']['rtol']


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(r):
    random.setstate(r[0]); np.random.set_state(r[1]); torch.set_rng_state(r[2]); torch.cuda.set_rng_state_all(r[3])


def rng_digest(r):
    return hashlib.sha256(pickle.dumps(r[:2]) + r[2].numpy().tobytes() +
                          b''.join(x.cpu().numpy().tobytes() for x in r[3])).hexdigest()


def metric(a, b):
    if a is None or b is None:
        return dict(passed=a is None and b is None, none_left=a is None, none_right=b is None)
    a=a.double().reshape(-1); b=b.double().reshape(-1)
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff=(a-b).abs(); na=float(a.norm()); nb=float(b.norm())
    return dict(passed=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)) and finite,
        finite=finite, max_abs=float(diff.max()) if len(diff) else 0.,
        relative_l2=float((a-b).norm())/max(na,nb,1e-30),
        cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if na and nb else (1. if na==nb else 0.),
        left_norm=na,right_norm=nb,violating_elements=int((diff>ATOL+RTOL*b.abs()).sum()),elements=len(a))


def comparisons(runs):
    result={}
    for label,left,right in [('original_original',0,1),('new_new',2,3),('original_new',0,2)]:
        a,b=runs[left],runs[right]
        groups={}
        for group in ('losses','outputs','gradients','activations'):
            if group in a:
                groups[group]={key:metric(a[group][key],b[group].get(key)) for key in a[group]}
        controlled=a['before']==b['before'] and a['input_sha']==b['input_sha'] and a['rng_before']==b['rng_before']
        result[label]=dict(same_initial_state=a['before']==b['before'],
            same_input=a['input_sha']==b['input_sha'],same_initial_rng=a['rng_before']==b['rng_before'],
            matching_equal=a['matching']==b['matching'],groups=groups,
            passed=controlled and all(v['passed'] for g in groups.values() for v in g.values()) and a['matching']==b['matching'])
    return result


def assignment(head,p,s):
    return [head.assigner.assign(b[0].detach(),c[0].detach(),s.boxes,s.labels,None).gt_inds.cpu().tolist()
            for c,b in zip(p['all_cls_scores'],p['all_bbox_preds'])]


def calculate(head,p,s,method):
    return official_loss(head,p,[s]) if method=='original' else ValidatedCoveragePETRLoss(head)(p,[s])[0]


def frozen_outputs(model,s,base,rstate):
    c,b,_=fixture(True,queries=900)
    c=torch.randn_like(c)*.3
    p0=REUSE['as_layers'](c,b)
    results={}
    for precision in ('FP32','OfficialAMP'):
        runs=[]
        for method in ('original','original','new','new'):
            model.load_state_dict(base);model.zero_grad(set_to_none=True);restore_rng(rstate)
            p={k:v.detach().clone().requires_grad_() if isinstance(v,torch.Tensor) else v for k,v in p0.items()}
            before=state_hash(model); rh=rng_digest(rng())
            # Actual AMP returns FP32 Head outputs; no artificial half cast.
            loss=calculate(model.pts_bbox_head,p,s,method)
            scale=512. if precision=='OfficialAMP' else 1.
            (sum(loss.values())*scale).backward()
            runs.append(dict(before=before,input_sha=digest(p['all_cls_scores']),rng_before=rh,
                matching=assignment(model.pts_bbox_head,p,s),
                losses={k:v.detach().cpu() for k,v in loss.items()},
                outputs={k:p[k].detach().cpu() for k in ('all_cls_scores','all_bbox_preds')},
                gradients={k:p[k].grad.detach().cpu()/scale for k in ('all_cls_scores','all_bbox_preds')}))
        results[precision]=dict(comparisons=comparisons(runs),output_dtype='torch.float32',
            scaler='NotUsed_ManualStaticScaleThenUnscaled',scale=scale)
    return results


def full_run(model,s,base,rstate,image0,meta0,method,precision,mode,keep_dcn=False):
    model.load_state_dict(base);model.zero_grad(set_to_none=True)
    model.train(mode=='train');restore_rng(rstate)
    image=image0.detach().clone().requires_grad_()
    meta=copy.deepcopy(meta0);scale=512. if precision=='OfficialAMP' else 1.
    before=state_hash(model);rh=rng_digest(rng());trace={};acts={};saved={};handles=[]
    names=['img_backbone.conv1','img_backbone.layer1','img_backbone.layer2',
           'img_backbone.layer3','img_backbone.layer4']
    names += [n for n,m in model.named_modules() if 'img_backbone' in n and 'ModulatedDeformConv' in type(m).__name__]
    def hook(name):
        def f(module,inputs,output):
            if not isinstance(output,torch.Tensor):return
            acts[name+'.output']=output.detach().cpu()
            if output.requires_grad:
                output.register_hook(lambda g:trace.__setitem__(name+'.output',g.detach().float().cpu()/scale))
            if inputs[0].requires_grad:
                inputs[0].register_hook(lambda g:trace.__setitem__(name+'.input',g.detach().float().cpu()/scale))
            if name=='img_backbone.layer4.2.conv2' and torch.is_grad_enabled():
                saved['input']=inputs[0].detach().cpu();saved['state']={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
                saved['type']=type(module).__name__;saved['output_dtype']=str(output.dtype)
                if output.requires_grad:output.register_hook(lambda g:saved.__setitem__('grad_output',g.detach().cpu()))
        return f
    for name in names:handles.append(dict(model.named_modules())[name].register_forward_hook(hook(name)))
    with torch.autocast('cuda',enabled=False):
        features=model.extract_feat(image.clone(),meta)
        features[0].retain_grad()
        p=model.pts_bbox_head(features,meta)
        for k in ('all_cls_scores','all_bbox_preds'):p[k].retain_grad()
        loss=calculate(model.pts_bbox_head,p,s,method)
    (sum(loss.values())*scale).backward();torch.cuda.synchronize()
    for h in handles:h.remove()
    grads={n:None if param.grad is None else param.grad.detach().float().cpu()/scale for n,param in model.named_parameters()}
    grads.update({'INPUT':image.grad.detach().float().cpu()/scale,'FEATURE':features[0].grad.detach().float().cpu()/scale})
    grads.update({'TRACE.'+k:v for k,v in trace.items()})
    grads.update({k:p[k].grad.detach().float().cpu()/scale for k in ('all_cls_scores','all_bbox_preds')})
    record=dict(method=method,before=before,after=state_hash(model),input_sha=digest(image0),rng_before=rh,rng_after=rng_digest(rng()),
        matching=assignment(model.pts_bbox_head,p,s),losses={k:v.detach().cpu() for k,v in loss.items()},
        outputs={k:p[k].detach().cpu() for k in ('all_cls_scores','all_bbox_preds')},gradients=grads,activations=acts,
        model_mode=mode,feature_dtype=str(features[0].dtype),logit_dtype=str(p['all_cls_scores'].dtype),
        scaler='NotApplicable_FP32' if precision=='FP32' else 'NoGradScaler_ManualStatic512UnscaledBeforeComparison',scale=scale,
        module_modes={n:m.training for n,m in model.named_modules() if n in ('img_backbone','pts_bbox_head','grid_mask','img_backbone.bn1')},
        buffers_equal=all(torch.equal(v.detach().cpu(),base[k]) for k,v in model.named_buffers()))
    if keep_dcn and saved:
        saved['scale']=scale
        torch.save(saved,ROOT/(precision+'_'+mode+'_dcn.pt'))
    return record


def primitive(model,precision,mode):
    path=ROOT/(precision+'_'+mode+'_dcn.pt')
    if not path.exists():return dict(status='NotCaptured')
    # Our own synthetic tensors; no external checkpoint deserialization.
    capture=torch.load(path,map_location='cpu')
    original=dict(model.named_modules())['img_backbone.layer4.2.conv2']
    module=copy.deepcopy(original).cuda().eval()
    state=capture['state'];results={}
    for dtype in (torch.float32,torch.float16):
        runs=[]
        for _ in range(3):
            module.load_state_dict(state);module.zero_grad(set_to_none=True)
            x=capture['input'].to(device='cuda',dtype=dtype).requires_grad_()
            grad=capture['grad_output'].to(device='cuda',dtype=dtype)/capture['scale']
            # Match conv-offset autocast of the official feature path.
            with torch.autocast('cuda',dtype=torch.float16,enabled=dtype==torch.float16):
                y=module(x)
            y.backward(grad)
            runs.append({'input':x.grad.detach().float().cpu(),
                         **{n:p.grad.detach().float().cpu() for n,p in module.named_parameters()}})
        results[str(dtype)]=[dict(repeat=i,gradients={n:metric(runs[0][n],runs[i][n]) for n in runs[0]}) for i in (1,2)]
    return dict(module='img_backbone.layer4.2.conv2',type=capture['type'],
        captured_input_shape=list(capture['input'].shape),upstream_gradient='IdenticalFrozenSyntheticTensor_Unscaled',
        comparisons=results,scope='IsolatedOfficialDCNv2Pack_NoLoss_NoDecoder_NoCheckpoint')


def main():
    import mmcv,mmdet,mmdet3d
    import projects.mmdet3d_plugin
    from mmcv.runner import wrap_fp16_model
    from mmdet3d.models import build_model
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    seed=PROTOCOL['numeric']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    deterministic = os.environ.get('M8J5C_DETERMINISTIC','1') == '1'
    torch.use_deterministic_algorithms(deterministic)
    cfg=mmcv.Config.fromfile(str(Path(os.environ['M8J5_RUNTIME'])/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    cfg.model.img_backbone.pretrained=None
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda().eval()
    model.pts_bbox_head.init_weights()
    base={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    image=torch.randn((1,6,3,64,96),device='cuda')
    mat=np.array([[48.,-50,0,0],[32,0,-50,0],[1,0,0,0],[0,0,0,1]],np.float32)
    meta=[dict(pad_shape=[(64,96,3)]*6,img_shape=[(64,96,3)]*6,lidar2img=[mat.copy() for _ in range(6)])]
    _,_,s=fixture(True);s.boxes[0,:3]=torch.tensor([10.,0,0],device='cuda')
    rstate=rng();start=time.perf_counter()
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256(Path('configs/m8j5c_protocol.json').read_bytes()).hexdigest(),
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmdet3d=mmdet3d.__version__,
        settings=PROTOCOL['numeric'],actual_torch_deterministic_algorithms=deterministic,
        actual_settings=dict(TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),GradScaler='NotInstantiated'),
        determinism_override_reason=os.environ.get('M8J5C_DETERMINISM_REASON','None'),
        initial_state_sha=state_hash(model),
        frozen_decoder=frozen_outputs(model,s,base,rstate),full_model={},isolated_operator={})
    for precision in ('FP32','OfficialAMP'):
        if precision=='OfficialAMP':wrap_fp16_model(model)
        for mode in ('eval','train'):
            key=precision+'_'+mode;print(key,flush=True)
            try:
                runs=[full_run(model,s,base,rstate,image,meta,method,precision,mode,keep_dcn=i==0)
                      for i,method in enumerate(('original','original','new','new'))]
                report['full_model'][key]=dict(comparisons=comparisons(runs),
                    runs=[{k:v for k,v in r.items() if k not in ('losses','outputs','gradients','activations','matching')} for r in runs])
                # Retain synthetic gradient tensors only in private experiment directory.
                torch.save(runs,ROOT/(key+'_runs.pt'))
                report['isolated_operator'][key]=primitive(model,precision,mode)
                del runs
            except Exception as e:
                report['full_model'][key]=dict(error_type=type(e).__name__,error=str(e),status='Failed_NoRelaxation')
                raise
            finally:
                report['elapsed_s']=time.perf_counter()-start
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                (OUT/'numeric_partial.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    with (OUT/'numeric_results.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('Completed diagnostic; comparisons determine gates, no optimizer steps',flush=True)


if __name__=='__main__':main()
