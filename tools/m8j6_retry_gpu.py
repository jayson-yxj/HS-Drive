"""One bounded phase of the independent 2+1 PETR FP32 retry.

Executed only by m8j6_retry_run in isolated fresh processes. Reuses the previous
official model/data/loss path; no dataset or model architecture changes.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from tools import m8j6_gpu_actual as old
from training.m8j6_retry_state import (parameter_mapping, head_policy, snapshot, fingerprint,
    assert_exact, rng_state, restore_rng, optimizer_summary, assert_frozen_unchanged)
from datasets.m8j6_selection import sha256


def training_mode(model):
    model.eval()
    model.pts_bbox_head.train()
    assert not model.img_backbone.training
    if model.with_img_neck: assert not model.img_neck.training


def evaluation_probe(model, prepared, frames):
    """A fixed eval probe must not change the continuing training RNG/modes."""
    state=rng_state();modes={n:m.training for n,m in model.named_modules()}
    model.eval()
    try:
        with torch.no_grad():
            img,meta,_,_=old.load_frame(prepared,frames[0])
            output=old.forward(model,img,meta)
            result={k:snapshot(v) for k,v in output.items() if isinstance(v,torch.Tensor)}
        torch.cuda.synchronize()
        return result
    finally:
        for n,m in model.named_modules():m.train(modes[n])
        restore_rng(state)


def tensor_output_compare(expected, actual, atol, rtol):
    assert set(expected)==set(actual)
    result={}
    for n,a in expected.items():
        b=actual[n];old.finite(a,'expected_output');old.finite(b,'restored_output')
        diff=(a.double()-b.double())
        result[n]=dict(max_abs_error=float(diff.abs().max()),
            relative_l2=float(diff.norm()/a.double().norm().clamp_min(1e-30)),
            bitwise_equal=torch.equal(a,b),passed=torch.allclose(a,b,atol=atol,rtol=rtol))
        assert result[n]['passed'],'EvalOutputRestore:'+n
    return result


def save_checkpoint(model, optimizer, prepared, frames, mapping, original, progress, private, run, code_commit):
    continuation_rng=rng_state()
    outputs=evaluation_probe(model,prepared,frames)
    assert_exact(continuation_rng,rng_state(),'PostProbeRNG')
    state=snapshot(model.state_dict());opt=snapshot(optimizer.state_dict())
    modes={n:m.training for n,m in model.named_modules()}
    payload=dict(model=state,optimizer=opt,global_step=progress,next_sample_position=progress,
        next_sample_index=[0,4,1][progress] if progress<3 else None,sample_order=[0,4,1],
        scheduler=run['scheduler'],rng=continuation_rng,module_modes=modes,mapping=mapping,
        fixed_eval_outputs=outputs,run_id=run['run_id'],code_commit=code_commit,
        manifest_sha256=sha256(prepared/'manifest.json'),original_state_sha256=old.state_sha(original),
        protocol_sha256=sha256(run['frozen_protocol']),
        state_fingerprint=fingerprint(state),optimizer_fingerprint=fingerprint(opt))
    path=private/f'step_{progress}.pth'
    if path.exists():raise FileExistsError(path)
    torch.save(payload,path)
    return dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size,global_step=progress,
        next_sample_position=progress,next_sample_index=payload['next_sample_index'],
        model_fingerprint=payload['state_fingerprint'],optimizer_fingerprint=payload['optimizer_fingerprint'],
        RNG_fingerprint=fingerprint(continuation_rng),mapping_fingerprint=fingerprint(mapping),
        scheduler=payload['scheduler'],saving_process_pid=os.getpid(),
        fixed_eval_output_fingerprint=fingerprint(outputs),saved_model_modes=modes)


def restore_checkpoint(model,optimizer,prepared,frames,mapping,original,path,expected,run,code_commit):
    assert path.is_file() and sha256(path)==expected['sha256']
    saved=torch.load(path,map_location='cpu')  # Own private run only, hash verified above.
    assert saved['run_id']==run['run_id'] and saved['code_commit']==code_commit
    assert saved['manifest_sha256']==sha256(prepared/'manifest.json')
    assert saved['protocol_sha256']==sha256(run['frozen_protocol'])
    assert saved['original_state_sha256']==old.state_sha(original)
    assert saved['sample_order']==[0,4,1]
    assert saved['global_step']==expected['global_step']==saved['next_sample_position']
    assert saved['scheduler']==run['scheduler'] and saved['scheduler']['state'] is None
    assert fingerprint(saved['model'])==expected['model_fingerprint']==saved['state_fingerprint']
    assert fingerprint(saved['optimizer'])==expected['optimizer_fingerprint']==saved['optimizer_fingerprint']
    assert_exact(mapping,saved['mapping'],'ReconstructedMappingBeforeLoad')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    assert_exact(saved['model'],model.state_dict(),'RestoredModelParametersAndBuffers')
    assert_exact(saved['optimizer'],optimizer.state_dict(),'RestoredOptimizerState')
    restored_mapping=parameter_mapping(model,optimizer,head_policy)
    assert_exact(mapping,restored_mapping,'SharedAliasAndOptimizerMappingAfterLoad')
    assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
    for n,m in model.named_modules():m.train(saved['module_modes'][n])
    assert_exact(saved['module_modes'],{n:m.training for n,m in model.named_modules()},'RestoredModes')
    restore_rng(saved['rng'])
    errors=tensor_output_compare(saved['fixed_eval_outputs'],evaluation_probe(model,prepared,frames),
        run['output_tolerance']['atol'],run['output_tolerance']['rtol'])
    # Loading, rebuilding, and eval consume no continuing random draws.
    assert_exact(saved['rng'],rng_state(),'ReadyToContinueRNG')
    assert fingerprint(rng_state())==expected['RNG_fingerprint']
    return saved['global_step'],dict(passed=True,checkpoint_sha256=expected['sha256'],
        saving_process_pid=expected['saving_process_pid'],restoring_process_pid=os.getpid(),
        new_process=os.getpid()!=expected['saving_process_pid'],
        model_parameters_and_buffers_exact=True,optimizer_exact=True,parameter_sharing_preserved=True,
        optimizer_mapping_exact=True,freeze_policy_exact=True,RNG_exact=True,module_modes_exact=True,
        scheduler=saved['scheduler'],restored_global_step=saved['global_step'],
        next_sample_position=saved['next_sample_position'],next_sample_index=saved['next_sample_index'],
        fixed_eval_output_comparison=errors,atol=run['output_tolerance']['atol'],rtol=run['output_tolerance']['rtol'],
        scope='SavedStateRestoreEvalAndContinue_NotAllUninterruptedTrainingBitwiseEquivalence')


def optimizer_step(model, optimizer, prepared, frame, global_step, original, mapping, rep, cfg):
    start=time.perf_counter();model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    assert parameter_mapping(model,optimizer,head_policy)==mapping
    assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
    before=snapshot(model.state_dict())
    img,meta,boxes,labels=old.load_frame(prepared,frame)
    raw=old.forward(model,img,meta)
    for key in ('all_cls_scores','all_bbox_preds'):raw[key].retain_grad()
    with torch.autocast('cuda',enabled=False):loss=model.pts_bbox_head.loss([boxes],[labels],raw)
    for name,value in loss.items():old.finite(value,'loss.'+name)
    layers=old.layer_stats(model.pts_bbox_head,raw,boxes,labels)
    total=sum(loss.values());total.backward();torch.cuda.synchronize()
    grads=old.gradient_stats(model,raw,layers)
    grad_norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    for n,p in model.named_parameters():
        if not p.requires_grad:assert p.grad is None,'FrozenGradient:'+n
    clipped=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['smoke']['gradient_clip_norm'])
    old.finite(clipped,'clip_norm')
    post_clip_norm=sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None)**.5
    pre_record=dict(global_step_about_to_execute=global_step+1,frame_index=frame['index'],
        sample_token=frame['sample_token'],losses={k:float(v.detach()) for k,v in loss.items()},
        total_loss=float(total.detach()),classification_loss=sum(float(v.detach()) for k,v in loss.items() if 'cls' in k),
        regression_loss=sum(float(v.detach()) for k,v in loss.items() if 'bbox' in k),gradients=grads,
        parameter_gradient_norms=grad_norms,gradient_norm_before_clip=float(clipped),gradient_norm_after_clip=post_clip_norm,
        layers=[{k:v for k,v in l.items() if k not in ('positive_indices','negative_indices')} for l in layers],
        before_state_sha256=old.state_sha(before),optimizer_before=optimizer_summary(optimizer),
        peak_allocated_before_step_bytes=torch.cuda.max_memory_allocated())
    old.save(rep/f'pre_step_{global_step+1}.json',pre_record)
    assert torch.cuda.max_memory_allocated()<=cfg['smoke']['maximum_allocated_GiB']*1024**3
    optimizer.step()
    # Ledger is persisted immediately after step returns; later failures cannot
    # erase the fact that an update has occurred.
    old.save(rep/f'optimizer_step_{global_step+1}_returned.json',dict(run_phase=os.environ['M8J6R_PHASE'],
        global_step=global_step+1,process_pid=os.getpid(),time_utc=datetime.now(timezone.utc).isoformat()))
    changed=[]
    for name,p in model.named_parameters():
        old.finite(p,'after_step.'+name)
        if p.requires_grad:
            delta=p.detach().cpu()-before[name]
            changed.append(dict(name=name,changed=bool(torch.count_nonzero(delta)),
                delta_l2=float(delta.double().norm()),delta_max_abs=float(delta.abs().max()),
                effective_gradient_norm=grad_norms.get(name,0.)))
    updates=[r for r in changed if r['changed'] and r['effective_gradient_norm']>0]
    assert updates,'NoGradientDrivenUpdate'
    assert any(r['name'].startswith('pts_bbox_head.cls_branches') for r in updates)
    assert any(r['name'].startswith('pts_bbox_head.reg_branches') for r in updates)
    assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
    assert_frozen_unchanged(model,before,mapping['frozen_state_names'])
    assert parameter_mapping(model,optimizer,head_policy)==mapping
    opt=optimizer_summary(optimizer);assert opt['step_values']==[global_step+1]
    assert opt['first_moment_l2']>0 and opt['weight_decay']==[0.]
    torch.cuda.synchronize()
    result=dict(pre_record,global_step=global_step+1,trainable_parameter_changes=changed,
        gradient_driven_changed_parameters=len(updates),frozen_state_checked=len(mapping['frozen_state_names']),
        frozen_parameter_and_buffer_max_abs_change=0.,all_buffers_max_abs_change=0.,
        after_state_sha256=old.state_sha(model.state_dict()),optimizer_after=opt,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_s=time.perf_counter()-start,memory_scope='torch CUDA allocator; excludes other processes/driver; reset per step',
        actual_amp=False,GradScaler='NotUsed',backbone_mode='eval',head_mode='train')
    # Every buffer is frozen, not just BN running statistics.
    assert all(torch.equal(v.detach().cpu(),original[n]) for n,v in model.named_buffers())
    old.save(rep/f'step_{global_step+1}.json',result)
    print('step',global_step+1,'frame',frame['index'],'loss',result['total_loss'],'peak',result['peak_allocated_bytes'],flush=True)
    return result


def main():
    run=json.loads(Path('configs/m8j6_retry_protocol.json').read_text())
    cfg=json.loads(Path(run['frozen_protocol']).read_text());phase=os.environ['M8J6R_PHASE']
    root=Path(os.environ['M8J6R_ROOT']);rep=Path(os.environ['M8J6R_REPORT']);private=root/phase
    private.mkdir(exist_ok=False)
    rt=Path(os.environ['M8J6_RUNTIME']);prepared=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    manifest=json.loads((prepared/'manifest.json').read_text());frames=manifest['frames']
    code_commit=os.environ['M8J6R_COMMIT'];progress=0;steps=[];model=None;optimizer=None
    report=dict(run_id=run['run_id'],phase=phase,code_commit=code_commit,process_pid=os.getpid(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),status='Running',steps=[],
        optimizer_calls_this_process=0,old_attempt_optimizer_steps=1,precision='FP32',
        torch=torch.__version__,cuda=torch.version.cuda,mmcv=old.mmcv.__version__,mmdet=old.mmdet.__version__,
        mmdet3d=old.mmdet3d.__version__,GPU=torch.cuda.get_device_name(),
        scheduler=run['scheduler'],seed=cfg['smoke']['seed'],sample_order=[0,4,1],
        optimizer=dict(type='AdamW',lr=1e-5,betas=[.9,.999],weight_decay=0.,gradient_clip=35.),
        TF32=False,autocast=False,GradScaler='NotUsed',cudnn_deterministic=True,cudnn_benchmark=False,
        deterministic_algorithms=False,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    start=time.perf_counter();original_nan_to_num=torch.nan_to_num
    try:
        assert sha256(run['frozen_protocol'])==run['frozen_protocol_sha256']==manifest['protocol_sha256']
        assert len(frames)==8
        baseline=json.loads(Path('reports/m8j6_sources/actual_baseline/result.json').read_text())
        assert baseline['status']=='FrozenFP32BaselineAndGradientsPassed'
        assert cfg['smoke']['sample_manifest_indices']==[0,4,1]
        seed=cfg['smoke']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(False)
        torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory)
        def checked_nan(x,*args,**kwargs):old.finite(x,'BeforeOfficialNanToNum');return original_nan_to_num(x,*args,**kwargs)
        torch.nan_to_num=checked_nan
        sys.addaudithook(old.input_guard)
        old.PERCEPTION_ACTIVE=True;blocked=0
        for canary in [prepared/'supervision/00.npz',prepared.parent/'data/samples/LIDAR_TOP/forbidden.bin']:
            try:canary.read_bytes()
            except PermissionError:blocked+=1
        old.PERCEPTION_ACTIVE=False;assert blocked==2
        archive=rt/'artifacts/safe_extract/model_tensors.npz';assert sha256(archive)==cfg['official']['safe_tensor_archive_sha256']
        model=old.build(rt,cfg,archive)
        original=snapshot(model.state_dict());assert old.state_sha(original)==baseline['initial_state_sha256']
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.999),weight_decay=0.)
        assert len(optimizer.state)==0
        mapping=parameter_mapping(model,optimizer,head_policy)
        report.update(initial_weight_sha256=cfg['official']['weights_sha256'],safe_archive_sha256=sha256(archive),
            original_state_sha256=old.state_sha(original),fresh_optimizer_initial_entries=0,input_canaries_rejected=blocked,
            initial_snapshot_is_independent_CPU_clone=True,mapping_fingerprint=fingerprint(mapping),
            trainable_numel=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_numel=sum(p.numel() for p in model.parameters() if not p.requires_grad))
        old.save(rep/'parameter_mapping.json',mapping)
        training_mode(model)
        if phase=='first':
            order=[0,4]
        else:
            prior='first' if phase=='resume' else 'resume'
            record=json.loads((rep.parent/prior/'result.json').read_text());assert record['status']=='PhasePassed'
            checkpoint=record['checkpoint']
            progress,restore=restore_checkpoint(model,optimizer,prepared,frames,mapping,original,
                Path(checkpoint['path']),checkpoint,run,code_commit)
            assert restore['new_process']
            report['restore']=restore
            old.save(rep/'restore.json',restore)
            order=[1] if phase=='resume' else []
            assert progress==(2 if phase=='resume' else 3)
        for index in order:
            assert index==cfg['smoke']['sample_manifest_indices'][progress] and progress<3
            result=optimizer_step(model,optimizer,prepared,frames[index],progress,original,mapping,rep,cfg)
            progress+=1;steps.append(result);report['optimizer_calls_this_process']+=1
        assert progress==(2 if phase=='first' else 3)
        report['global_step']=progress
        report['steps']=[{k:v for k,v in s.items() if k!='parameter_gradient_norms'} for s in steps]
        if phase!='verify':
            report['checkpoint']=save_checkpoint(model,optimizer,prepared,frames,mapping,original,progress,private,run,code_commit)
        assert sha256(archive)==cfg['official']['safe_tensor_archive_sha256']
        assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
        report.update(status='PhasePassed',frozen_final_check=True)
    except Exception as exc:
        # Derive actual returns from on-disk ledgers even if post-step checking
        # raises before the in-memory global counter is incremented.
        ledgers=[json.loads(p.read_text()) for p in rep.glob('optimizer_step_*_returned.json')]
        report['optimizer_calls_this_process']=len(ledgers)
        report['global_step']=max([progress]+[l['global_step'] for l in ledgers])
        report.update(status='FailedStopped',error_type=type(exc).__name__,error=str(exc))
        if model is not None:
            path=private/'failure.pth'
            torch.save(dict(model=snapshot(model.state_dict()),optimizer=snapshot(optimizer.state_dict()) if optimizer else None,
                global_step=report['global_step'],rng=rng_state(),run_id=run['run_id']),path)
            report['failure_checkpoint_sha256']=sha256(path)
        raise
    finally:
        torch.nan_to_num=original_nan_to_num
        report.update(elapsed_s=time.perf_counter()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
        old.save(rep/'result.json',report)


if __name__=='__main__':main()
