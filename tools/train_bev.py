"""Train the RGB-only single-frame Depth/BEV engineering baseline on CUDA."""
import argparse,json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from datasets.training import HSDriveDataset,make_dataloader
from simulator.recorder import atomic_json,file_hash
from tools.provenance import source_hashes
from training.runner import (seed_all,select_indices,to_device,initialize,evaluate,write_preview,
                             save_checkpoint,load_checkpoint,selection_score)
from training.losses import compute_losses,loss_options


_ACTIVE_OUTPUT = None


def main():
    global _ACTIVE_OUTPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/m4_baseline.yaml');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int);p.add_argument('--overfit',action='store_true')
    p.add_argument('--stop-after',type=int,help='Stop after this global step, saving resume state without changing the full step plan')
    p.add_argument('--resume',help='Resume into a NEW output directory with identical config/index')
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text())
    if config['schema_version']!='hsd-m4-baseline-v1':raise ValueError('Unknown config')
    if a.steps is not None:config['steps']=a.steps
    if config['steps']<1 or config['batch_size']!=1:raise ValueError('Baseline requires positive steps and batch=1')
    if not torch.cuda.is_available() or (config['amp_bf16'] and not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Validated CUDA/BF16 environment is required')
    from training.runtime import prepare_cuda
    prepare_cuda()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);_ACTIVE_OUTPUT=root
    seed_all(config['seed'])
    model=initialize(config,pretrained=not a.resume).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    train=HSDriveDataset(config['index'],'train',road_overlay=config.get('road_overlay'));val=HSDriveDataset(config['index'],'val',road_overlay=config.get('road_overlay'))
    if config.get('drivable_head') and train.index.get('drivable_bev_config_sha256')!=file_hash(config['bev_config']):
        raise ValueError('Static lane labels use different BEV geometry')
    train_ids=[0] if a.overfit else select_indices(train,config['train_samples_per_scene'])
    val_ids=select_indices(val,config['val_samples_per_scene'])
    index_sha=file_hash(Path(config['index'])/'index.json')
    config['experiment_mode']='single_sample_overfit' if a.overfit else 'small_sample_baseline'
    start_step=0;best_score=float('inf')
    if a.resume:
        previous=json.loads(Path(a.resume).with_name('manifest.json').read_text())
        if file_hash(config['bev_config'])!=previous['source_sha256'][config['bev_config']]:
            raise ValueError('Resume BEV geometry differs')
        if config.get('road_overlay') and file_hash(Path(config['road_overlay'])/'manifest.json')!=previous.get('road_overlay_manifest_sha256'):raise ValueError('Resume road overlay changed')
        state=load_checkpoint(a.resume,model,optimizer)
        if state['index_sha256']!=index_sha or state['config']!=config:raise ValueError('Resume config/index differs')
        start_step=state['step'];best_score=state['best_score']
        best_path=Path(a.resume).with_name('best.pt')
        prior_best=torch.load(best_path,map_location='cpu',weights_only=True)
        if prior_best['config']!=config or prior_best['index_sha256']!=index_sha:
            raise ValueError('Resume best checkpoint is incompatible')
        shutil.copy2(best_path,root/'best.pt')
    manifest={'schema_version':'hsd-m4-run-v1','status':'running','config':config,'index_sha256':index_sha,
        'source_sha256':source_hashes(),'train_sample_ids':[train.windows[i][-1]['sample_id'] for i in train_ids],
        'val_sample_ids':[] if a.overfit else [val.windows[i][-1]['sample_id'] for i in val_ids],
        'pretrained_sha256':file_hash(config['pretrained_weights']),
        'road_overlay_manifest_sha256':file_hash(Path(config['road_overlay'])/'manifest.json') if config.get('road_overlay') else None,
        'resume_sha256':file_hash(a.resume) if a.resume else None,
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
        'determinism':'fixed seeds/order; CUDA index_add and grid_sample backward can be nondeterministic',
        'loss_semantics':'depth CE + optional range balancing/log depth regression + observed surface MSE + optional masked static Drivable BCE/Dice; exact weights in config'}
    atomic_json(root/'manifest.json',manifest);(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (root/'env.lock').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    start=time.monotonic();initial,_=evaluate(model,train,train_ids,config)
    atomic_json(root/'initial_train.json',initial)
    if not a.overfit:
        initial_val,_=evaluate(model,val,val_ids,config);atomic_json(root/'initial_val.json',initial_val)
    generator=np.random.RandomState(config['seed'])
    planned=[]
    while len(planned)<config['steps']:planned.extend(generator.permutation(train_ids).tolist())
    planned=planned[:config['steps']]
    end_step=config['steps'] if a.stop_after is None else a.stop_after
    if not start_step < end_step <= config['steps']:raise ValueError('Need start_step < stop_after <= planned steps')
    subset=torch.utils.data.Subset(train,planned[start_step:end_step])
    loader=make_dataloader(subset,workers=config['workers'],shuffle=False,seed=config['seed'])
    torch.cuda.reset_peak_memory_stats();times=[];finite=True
    with (root/'metrics.jsonl').open('w') as log:
        for step,batch in enumerate(loader,start_step+1):
            now=time.monotonic();model.train();inputs,targets=to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp_bf16']):
                output=model(**inputs.as_model_kwargs())
                losses,_=compute_losses(model,output,inputs,targets,**loss_options(config))
            losses['total'].backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip_norm'],error_if_nonfinite=True)
            finite=finite and bool(torch.isfinite(losses['total']))
            if not finite:raise RuntimeError('Nonfinite loss')
            optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-now)
            row={'step':step,**{k:float(v.detach()) for k,v in losses.items()},'grad_norm':float(grad_norm),
                 'step_compute_s':times[-1],'sample_id':batch.metadata[0]['sample_id']}
            log.write(json.dumps(row)+'\n');log.flush()
            if step%10==0 or step==1:print(json.dumps(row),flush=True)
            if step%config['eval_every']==0 or step==end_step:
                report,_=evaluate(model,train if a.overfit else val,train_ids if a.overfit else val_ids,config)
                atomic_json(root/f'eval_{step:06d}.json',report)
                score=selection_score(report,config)
                if score<best_score:
                    best_score=score;save_checkpoint(root/'best.pt',model,optimizer,step,config,index_sha,best_score)
                save_checkpoint(root/'last.pt',model,optimizer,step,config,index_sha,best_score)
                print(f"Eval step={step} {config.get('selection_metric','total_loss')}={score:.4f}",flush=True)
    before={k:v.detach().cpu() for k,v in model.state_dict().items()}
    state=load_checkpoint(root/'last.pt',model)
    reload_ok=all(torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items())
    best=load_checkpoint(root/'best.pt',model)
    final,preview=evaluate(model,train,train_ids,config)
    atomic_json(root/'final_train.json',final);write_preview(preview,root/'previews','train')
    if not a.overfit:
        final_val,preview=evaluate(model,val,val_ids,config);atomic_json(root/'final_val.json',final_val)
        write_preview(preview,root/'previews','val')
    manifest.update(status='complete' if end_step==config['steps'] else 'checkpointed',best_step=best['step'],last_step=state['step'],
        all_gradients_finite=finite,checkpoint_reload_exact=reload_ok,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        step_compute_p50_s=float(np.median(times)),step_compute_p95_s=float(np.quantile(times,.95)),
        wall_s=time.monotonic()-start,
        best_sha256=file_hash(root/'best.pt'),last_sha256=file_hash(root/'last.pt'))
    atomic_json(root/'manifest.json',manifest);print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        import sys
        if _ACTIVE_OUTPUT is not None:
            manifest_path=_ACTIVE_OUTPUT/'manifest.json'
            if manifest_path.exists():
                record=json.loads(manifest_path.read_text())
                if record.get('status')=='running':
                    record.update(status='failed',error=repr(exc));atomic_json(manifest_path,record)
        raise
