"""Bounded, offline, frozen planning replay. No CARLA import/vehicle handle."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types
import numpy as np
import torch
from datasets.m8j12_contract import CAMERAS, DryRunController, official_camera_contract, planning_to_hs, measured_navigation_target
from tools.m8j12_runtime import protocol,build,reset,state_digest,observation_batch


def plain(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().numpy().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def finite(x):
    if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all())
    if isinstance(x,np.ndarray):return bool(np.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    return True


def append(path,obj):
    with path.open('a') as f:f.write(json.dumps(plain(obj),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def memory(model):
    bank=model.head.det_head.instance_bank;q=model.head.motion_plan_head.instance_queue
    return dict(cached=bank.cached_anchor is not None,history_timestamp=None if bank.metas is None else plain(bank.metas['timestamp']),
        valid=plain(bank.mask),instance_queue_length=len(q.anchor_queue),ego_queue_length=len(q.ego_anchor_queue))


def install_selection_log(model,store):
    decoder=model.head.motion_plan_head.planning_decoder;original=decoder.decode_v2
    def wrapped(self,*args,**kwargs):
        prior=sys.getprofile()
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is original.__func__.__code__:
                v=frame.f_locals
                for k in ('traj_mode_idx','traj_cls','num_lat_mode','num_lon_mode'):
                    if k in v:store[k]=plain(v[k])
                # Exact native candidate set; no duplicate algorithm or reselection.
                store['candidates']=v['traj_reg'].detach().cpu()
        sys.setprofile(profile)
        try:return original(*args,**kwargs)
        finally:sys.setprofile(prior)
    decoder.decode_v2=types.MethodType(wrapped,decoder)


def main():
    os.umask(0o077);c=protocol();root=Path(c['private_root']);out=root/'replay';out.mkdir()
    ledger=root/'forward_budget.jsonl'
    observation_root=Path(json.loads(Path('reports/m8j12_sources/capture_result.json').read_text())['observation_root'])
    count=sum(1 for _ in ledger.open()) if ledger.exists() else 0
    torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model,cfg,load=build();model.cuda();baseline=state_digest(model);versions={n:p._version for n,p in model.named_parameters()}
    camera=official_camera_contract(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
    selection={};install_selection_log(model,selection)
    observed_dtypes={}
    def dtype_hook(name):
        def hook(module,args,output):
            def walk(v):
                if isinstance(v,torch.Tensor):return {str(v.dtype)}
                if isinstance(v,dict):return set().union(*(walk(x) for x in v.values()))
                if isinstance(v,(list,tuple)):return set().union(*(walk(x) for x in v))
                return set()
            observed_dtypes.setdefault(name,set()).update(walk(output))
        return hook
    for name,module in [('backbone',model.img_backbone),('neck',model.img_neck),('planning_head',model.head.motion_plan_head)]:
        module.register_forward_hook(dtype_hook(name))
    postprocess=model.head.post_process
    def timed_postprocess(*args,**kwargs):
        torch.cuda.synchronize();start=time.perf_counter()
        result=postprocess(*args,**kwargs)
        torch.cuda.synchronize();selection['decode_ms']=(time.perf_counter()-start)*1000
        return result
    model.head.post_process=timed_postprocess
    records=[];comparisons=[];error=None;torch.cuda.reset_peak_memory_stats()
    try:
        for replay in range(2):
            for sequence in c['sequence_ids']:
                reset(model);ctrl=DryRunController(Path(c['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py')
                manifest=json.loads((observation_root/sequence/'manifest.json').read_text())
                assert len(manifest)==32 and not memory(model)['cached']
                folder=out/(str(replay)+'_'+sequence);folder.mkdir()
                previous_time=None
                for index,row in enumerate(manifest):
                    started=time.perf_counter();selection.clear()
                    images=dict(np.load(observation_root/sequence/row['images_file'],allow_pickle=False))
                    batch=observation_batch(cfg,row,images,camera);torch.cuda.synchronize();prep=time.perf_counter()-started
                    before=memory(model)
                    if previous_time is not None and abs(row['timestamp_s']-previous_time-.1)>1e-5:raise RuntimeError('TemporalGap')
                    if before['history_timestamp'] is not None and not all(t<row['timestamp_s'] for t in before['history_timestamp']):raise RuntimeError('FutureMemory')
                    if count>=c['max_full_forwards']:raise RuntimeError('ForwardBudgetExhausted')
                    count+=1
                    append(ledger,dict(attempt=count,replay=replay,sequence=sequence,index=index,frame=row['frame_id'],timestamp=row['timestamp_s']))
                    cuda_start=torch.cuda.Event(enable_timing=True);cuda_end=torch.cuda.Event(enable_timing=True)
                    start=time.perf_counter();cuda_start.record()
                    with torch.no_grad():prediction=model(**batch)[0]['img_bbox']
                    cuda_end.record();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    if not finite(prediction):raise RuntimeError('NonfiniteNativeOutput')
                    trajectory,meta=planning_to_hs(prediction,row['timestamp_s'])
                    control_start=time.perf_counter();control=ctrl.command(prediction,row['speed_mps'],measured_navigation_target(row));control_ms=(time.perf_counter()-control_start)*1000
                    if any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('ParameterVersionChanged')
                    payload=dict(prediction=prediction,selection=selection.copy(),hs_points=trajectory.points,
                        hs_meta=meta,control=control,memory_before=before,memory_after=memory(model))
                    torch.save(payload,folder/(str(index)+'.pt'))
                    if replay:
                        prior=torch.load(out/('0_'+sequence)/(str(index)+'.pt'),map_location='cpu')
                        errors={}
                        for key in ('traj_final','lat_reg_final','lon_reg_final'):
                            a=prior['prediction'][key].numpy();b=prediction[key].numpy();errors[key]=float(np.max(np.abs(a-b)))
                            if not np.allclose(a,b,atol=c['atol'],rtol=c['rtol']):raise RuntimeError('ReplayMismatch:'+key)
                        a=list(prior['control']['native_agent_control'].values());b=list(control['native_agent_control'].values())
                        if not np.allclose(a,b,atol=c['atol'],rtol=c['rtol']):raise RuntimeError('ControlReplayMismatch')
                        comparisons.append(dict(sequence=sequence,index=index,max_abs=errors,control_max_abs=float(np.max(np.abs(np.array(a)-b)))))
                    record=dict(replay=replay,sequence=sequence,index=index,frame=row['frame_id'],timestamp=row['timestamp_s'],
                        preprocess_ms=prep*1000,model_and_decode_ms=elapsed*1000,decode_ms=selection['decode_ms'],cuda_event_ms=cuda_start.elapsed_time(cuda_end),control_ms=control_ms,
                        control=control['native_agent_control'],raw_pid=control['raw_pid'],memory_before=before,memory_after=memory(model),
                        output_keys=sorted(prediction),selected_index=selection.get('traj_mode_idx'),
                        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
                    records.append(record);append(out/'frames.jsonl',record);previous_time=row['timestamp_s']
                digest=state_digest(model)
                changed=[k for k in baseline if baseline[k]!=digest[k]]
                if changed:raise RuntimeError('FixedStateChanged:'+str(changed[:10]))
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        final=state_digest(model);changed=[k for k in baseline if baseline[k]!=final[k]]
        summary=dict(status='Passed' if len(records)==128 and not error and not changed else 'Partial',
            successful_forwards=len(records),full_forward_attempts=count,limit=c['max_full_forwards'],error=error,
            fixed_state_changed=changed,repeated_frame_comparisons=len(comparisons),
            max_replay_abs={k:max((r['max_abs'][k] for r in comparisons),default=None) for k in ('traj_final','lat_reg_final','lon_reg_final')},
            precision='FP32; TF32 disabled; no autocast',observed_output_dtypes={k:sorted(v) for k,v in observed_dtypes.items()},load=load,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            model_control_applied=0,optimizer_steps=0,closed_loop_driving='NotRun',training='NotRun',
            timing_boundary='model includes extract_feat, native heads and post_process; CUDA events include native CPU/GPU synchronization; collection stopped before replay')
        for label,rows in [('all',records),('cold',[r for r in records if r['index']==0]),('subsequent',[r for r in records if r['index']>0])]:
            if rows:summary[label+'_model_ms']={k:float(v) for k,v in zip(('p50','p95','max'),np.percentile([r['model_and_decode_ms'] for r in rows],[50,95,100]))}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        Path('reports/m8j12_sources/planning_replay.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
