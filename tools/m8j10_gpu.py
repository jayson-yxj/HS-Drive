"""One C training run: J7 update function unchanged, new fixed sample schedule."""
from datetime import datetime,timezone
import json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from tools import m8j6_gpu_actual as old
from tools.m8j6_retry_gpu import training_mode,optimizer_step,evaluation_probe,tensor_output_compare
from tools.m8j8_gpu import query_provenance
from tools.m8j7_gpu import visualize
from training.m8j6_retry_state import (parameter_mapping,head_policy,snapshot,fingerprint,assert_exact,
    rng_state,restore_rng,optimizer_summary,assert_frozen_unchanged)
from datasets.m8j6_selection import sha256


def evaluate(model,protocol,private,report,step):
    prepared=Path(protocol['evaluation_prepared']);manifest=json.loads((prepared/'manifest.json').read_text())
    frozen=json.loads(Path(protocol['evaluation_manifest']).read_text())
    assert [f['sample_token'] for f in manifest['frames']]==[f['sample_token'] for f in frozen['frames']]
    state=rng_state();modes={n:m.training for n,m in model.named_modules()};before=fingerprint(model.state_dict())
    model.eval();directory=private/f'C{step}'/'mini_val';directory.mkdir(parents=True,exist_ok=False)
    rows=[];start=time.perf_counter()
    try:
        with torch.no_grad():
            for frame in manifest['frames']:
                t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                img,meta,gt,labels=old.load_frame(prepared,frame)
                raw=old.forward(model,img,meta)
                loss=model.pts_bbox_head.loss([gt],[labels],raw)
                for key,value in loss.items():old.finite(value,'eval_loss.'+key)
                layers=old.layer_stats(model.pts_bbox_head,raw,gt,labels)
                decoded,scores,pred_labels=model.pts_bbox_head.get_bboxes(raw,meta)[0]
                old.finite(decoded.tensor,'decoded');old.finite(scores,'scores')
                assert len(gt)==frame['retained_gt'] and (decoded.tensor[:,3:6]>0).all()
                qi,flat=query_provenance(model.pts_bbox_head,raw,decoded,scores,pred_labels)
                record=dict(boxes=decoded.tensor.cpu().numpy(),scores=scores.cpu().numpy(),labels=pred_labels.cpu().numpy(),
                    gt=gt.tensor.cpu().numpy(),gt_labels=labels.cpu().numpy(),query_indices=qi,flat_indices=flat,
                    raw_cls=raw['all_cls_scores'][-1,0].cpu().numpy(),raw_bbox=raw['all_bbox_preds'][-1,0].cpu().numpy())
                path=directory/f'{frame["index"]:02}.npz';np.savez_compressed(path,**record)
                if frame['index'] in (0,20,39,40,60,80):
                    visualize(directory,frame['index'],img,meta,gt,decoded,record['scores'],.2,manifest['six_camera_order'],step)
                rows.append(dict(index=frame['index'],sample_token=frame['sample_token'],scene_name=frame['scene_name'],
                    retained_gt=len(gt),decoded=len(scores),cache_path=str(path),cache_sha256=sha256(path),
                    losses={k:float(v) for k,v in loss.items()},total_loss=float(sum(loss.values())),
                    layers=[{k:v for k,v in layer.items() if k not in ('positive_indices','negative_indices')} for layer in layers],
                    elapsed_s=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                assert fingerprint(model.state_dict())==before,'EvaluationStateMutation'
                print('C eval',step,frame['index'],'GT',len(gt),'decoded',len(scores),flush=True)
                del img,meta,gt,labels,raw,loss,decoded,scores,pred_labels,record
        assert len(rows)==81 and sum(f['retained_gt'] for f in rows)==3559
        out=dict(step=step,scope='MiniValDevelopment_NotTrainingSet',frames=rows,model_state_unchanged=True,
            mean_total_loss=float(np.mean([f['total_loss'] for f in rows])),
            mean_losses={k:float(np.mean([f['losses'][k] for f in rows])) for k in rows[0]['losses']},
            elapsed_s=time.perf_counter()-start)
        old.save(report/f'eval_{step:03}.json',out)
        return out
    finally:
        for n,m in model.named_modules():m.train(modes[n])
        restore_rng(state)
        assert_exact(state,rng_state(),'EvalRNG')
        assert {n:m.training for n,m in model.named_modules()}==modes


def save_checkpoint(model,optimizer,prepared,frames,mapping,original,step,private,report,protocol,commit):
    outputs=evaluation_probe(model,prepared,frames)
    payload=dict(model=snapshot(model.state_dict()),optimizer=snapshot(optimizer.state_dict()),global_step=step,
        next_sample_position=step,sample_order_sha256=protocol['manifest_sha256'],scheduler=None,rng=rng_state(),
        module_modes={n:m.training for n,m in model.named_modules()},mapping=mapping,original_state_sha256=old.state_sha(original),
        run_id=protocol['run_id'],code_commit=commit,protocol_sha256=sha256('configs/m8j10_protocol.json'),
        prepared_manifest_sha256=sha256(prepared/'manifest.json'),fixed_eval_outputs=outputs)
    path=private/f'checkpoint_{step:03}.pth';assert not path.exists();torch.save(payload,path)
    record=dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path),step=step,
        model_fingerprint=fingerprint(payload['model']),optimizer_fingerprint=fingerprint(payload['optimizer']),
        mapping_fingerprint=fingerprint(mapping),rng_fingerprint=fingerprint(payload['rng']),saving_pid=os.getpid())
    old.save(report/f'checkpoint_{step:03}.json',record)
    return record


def main():
    protocol=json.loads(Path('configs/m8j10_protocol.json').read_text());cfg=json.loads(Path('configs/m8j6_protocol.json').read_text())
    prepared=Path(protocol['prepared_root']);manifest=json.loads((prepared/'manifest.json').read_text());frames=manifest['frames']
    fixed=json.loads(Path(protocol['manifest']).read_text());order=fixed['order']
    phase=os.environ['M8J10_PHASE'];report=Path(os.environ['M8J10_REPORT']);private=Path(protocol['private_root']);rt=Path(protocol['runtime'])
    commit=os.environ['M8J10_COMMIT'];tr=protocol['training'];start=time.perf_counter();old_nan=torch.nan_to_num
    result=dict(run_id=protocol['run_id'],phase=phase,code_commit=commit,started_at_utc=datetime.now(timezone.utc).isoformat(),
        status='Running',optimizer_steps=0,FP32=True,AMP=False,TF32=False,GradScaler='NotUsed',seed=tr['seed'],scheduler=None,
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,mmcv=old.mmcv.__version__,
        mmdet=old.mmdet.__version__,mmdet3d=old.mmdet3d.__version__)
    model=optimizer=None;checkpoints=[]
    try:
        for p,h in protocol['source_sha256'].items():assert sha256(p)==h,p
        assert sha256(protocol['manifest'])==protocol['manifest_sha256']
        prep_check=json.loads(Path('reports/m8j10_preflight.json').read_text())
        assert sha256(prepared/'manifest.json')==prep_check['prepared_manifest_sha256']
        assert len(frames)==242 and len(order)==400
        assert [f['sample_token'] for f in frames]==[f['sample_token'] for f in fixed['frames']]
        random.seed(tr['seed']);np.random.seed(tr['seed']);torch.manual_seed(tr['seed']);torch.cuda.manual_seed_all(tr['seed'])
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory)
        def checked_nan(x,*a,**kw):old.finite(x,'BeforeOfficialNanToNum');return old_nan(x,*a,**kw)
        torch.nan_to_num=checked_nan;sys.addaudithook(old.input_guard)
        old.PERCEPTION_ACTIVE=True;blocked=0
        for p in [prepared/'supervision/00.npz',Path(protocol['dataset_root'])/'samples/LIDAR_TOP/forbidden.bin']:
            try:p.read_bytes()
            except PermissionError:blocked+=1
        old.PERCEPTION_ACTIVE=False;assert blocked==2
        archive=Path(protocol['A']['path']);assert sha256(archive)==protocol['A']['sha256']
        model=old.build(rt,cfg,archive);original=snapshot(model.state_dict())
        assert old.state_sha(original)==json.loads(Path('reports/m8j6_sources/actual_baseline/result.json').read_text())['initial_state_sha256']
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=tr['lr'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        assert len(optimizer.state)==0
        mapping=parameter_mapping(model,optimizer,head_policy)
        assert mapping['canonical_trainable_count']==141 and len(mapping['frozen_state_names'])==337
        training_mode(model);coder=model.pts_bbox_head.bbox_coder
        assert coder.max_num==300 and coder.score_threshold is None
        assert list(coder.post_center_range)==protocol['evaluation']['decoder_post_center_range']
        old.save(report/'parameter_mapping.json',mapping)
        result.update(initial_weight_sha256=sha256(archive),original_state_sha256=old.state_sha(original),
            fresh_optimizer_initial_entries=0,input_canaries_rejected=blocked,
            dropout_modules={n:float(m.p) for n,m in model.pts_bbox_head.named_modules() if isinstance(m,torch.nn.Dropout)},
            attention_dropout={n:float(m.dropout) for n,m in model.pts_bbox_head.named_modules() if isinstance(m,torch.nn.MultiheadAttention)})
        if phase=='verify':
            cp=json.loads((report.parent/'train/checkpoint_400.json').read_text());assert sha256(cp['path'])==cp['sha256']
            saved=torch.load(cp['path'],map_location='cpu')
            assert saved['global_step']==400 and saved['run_id']==protocol['run_id'] and saved['protocol_sha256']==sha256('configs/m8j10_protocol.json')
            assert saved['sample_order_sha256']==protocol['manifest_sha256'] and saved['prepared_manifest_sha256']==sha256(prepared/'manifest.json')
            assert saved['original_state_sha256']==old.state_sha(original) and saved['scheduler'] is None
            assert os.getpid()!=cp['saving_pid']
            assert_exact(mapping,saved['mapping'],'MappingBeforeLoad')
            assert fingerprint(saved['model'])==cp['model_fingerprint'] and fingerprint(saved['optimizer'])==cp['optimizer_fingerprint']
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            assert_exact(saved['model'],model.state_dict(),'ModelRestore');assert_exact(saved['optimizer'],optimizer.state_dict(),'OptimizerRestore')
            assert_exact(mapping,parameter_mapping(model,optimizer,head_policy),'SharedMappingRestore')
            for n,m in model.named_modules():m.train(saved['module_modes'][n])
            restore_rng(saved['rng'])
            errors=tensor_output_compare(saved['fixed_eval_outputs'],evaluation_probe(model,prepared,frames),1e-6,1e-5)
            assert_exact(saved['rng'],rng_state(),'RNGRestore');assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
            result.update(status='FinalReloadPassed',checkpoint_sha256=cp['sha256'],model_buffers_optimizer_exact=True,
                shared_mapping_preserved=True,RNG_exact=True,fixed_eval_output_comparison=errors,new_process=True,
                optimizer_state=optimizer_summary(optimizer),frozen_final_unchanged=True)
        else:
            steps_dir=private/'steps';steps_dir.mkdir(exist_ok=False)
            with (report/'steps.jsonl').open('x') as stream:
                for row in order:
                    step=row['step'];frame=frames[row['index']]
                    assert 1<=step<=400 and frame['scene_name']==row['scene_name']
                    assert not model.img_backbone.training and model.pts_bbox_head.training
                    s=optimizer_step(model,optimizer,prepared,frame,step-1,original,mapping,steps_dir,cfg)
                    record={k:v for k,v in s.items() if k not in ('parameter_gradient_norms','trainable_parameter_changes')}
                    record.update(scene_name=frame['scene_name'],sample_token=frame['sample_token'],scene_cycle=row['scene_cycle'],
                        actual_lr=optimizer.param_groups[0]['lr'],parameter_delta_l2=sum(x['delta_l2']**2 for x in s['trainable_parameter_changes'])**.5,
                        private_detail_sha256=sha256(steps_dir/f'step_{step}.json'))
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();result['optimizer_steps']=step
                    if step in (200,400):
                        # Save before dev evaluation so an evaluation exception cannot lose completed updates.
                        checkpoints.append(save_checkpoint(model,optimizer,prepared,frames,mapping,original,step,private,report,protocol,commit))
                        evaluate(model,protocol,private,report,step)
            assert len(list(steps_dir.glob('optimizer_step_*_returned.json')))==400
            assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
            assert sha256(archive)==protocol['A']['sha256']
            result.update(status='Completed400Steps',global_step=400,checkpoints=checkpoints,
                frozen_final_unchanged=True,optimizer_state=optimizer_summary(optimizer))
    except Exception as exc:
        result.update(status='FailedStopped',optimizer_steps=len(list((private/'steps').glob('optimizer_step_*_returned.json'))),error_type=type(exc).__name__,error=str(exc))
        if model is not None:
            p=private/f'failure_{phase}.pth';assert not p.exists()
            torch.save(dict(model=snapshot(model.state_dict()),optimizer=snapshot(optimizer.state_dict()) if optimizer else None,rng=rng_state(),actual_optimizer_steps=result['optimizer_steps']),p)
            result['failure_checkpoint']=dict(path=str(p),sha256=sha256(p))
        raise
    finally:
        torch.nan_to_num=old_nan;result.update(elapsed_s=time.perf_counter()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
        old.save(report/'result.json',result)


if __name__=='__main__':main()
