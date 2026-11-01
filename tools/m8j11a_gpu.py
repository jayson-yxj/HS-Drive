"""Bounded frozen-weight interventions, reusing J11 input/model/output contracts."""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import mmcv, mmdet, mmdet3d
import projects.mmdet3d_plugin
from mmdet3d.models import build_model
from tools.m8j11_gpu import load, guard
from datasets.m8j11_contract import to_legacy_boxes
from datasets.m8j11a_temporal import HistoryIntervention, reset_official
from tools.m8j11a_common import read,write,sha,load_cache,compare_arrays,PROTOCOL
from tools.m8j11a_budget import reserve
from training.m8j6_retry_state import snapshot,fingerprint,assert_exact


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=['T2','T1','T3'],required=True);a=ap.parse_args()
    os.umask(0o077);p=read(PROTOCOL);c=read(p['source_protocol']);root=Path(p['private_root']);rt=Path(p['runtime_root'])
    out=root/a.condition;out.mkdir(exist_ok=False)
    rows=p['schedule'][a.condition];prepared=read(p['prepared_manifest'])['frames']
    result=dict(status='Running',condition=a.condition,run_id=p['run_ids'][a.condition],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol_sha256=sha(PROTOCOL),optimizer_steps=0,backward_calls=0,actual_forwards=0,
        precision='FP32_compute_FP64_timestamps',warmup_forwards=0,AMP=False,TF32=False,GT_inputs=False,
        torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmdet3d=mmdet3d.__version__,
        GPU=torch.cuda.get_device_name(),checks=[],frames=[])
    start=time.perf_counter()
    try:
        for path,h in p['fixed_assets_sha256'].items():assert sha(path)==h, 'FixedAssetChanged:'+path
        assert (torch.__version__,mmcv.__version__,mmdet.__version__,mmdet3d.__version__)==('2.0.0+cu118','1.7.0','2.28.2','1.0.0rc6')
        for f in p['T0_cache']:assert sha(f['path'])==f['sha256']
        if a.condition=='T1':assert read(root/'T2/result.json')['status']=='InterventionSequencePassed'
        if a.condition=='T3':assert read(root/'T1/result.json')['status']=='InterventionSequencePassed'
        sys.addaudithook(guard)
        seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(.9)
        conf=mmcv.Config.fromfile(str(rt/'runtime-source'/c['config']));conf.model.img_backbone.pretrained=None
        attn=conf.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1]
        assert attn.type=='PETRMultiheadFlashAttention';attn.type='PETRMultiheadAttention';attn.fp16=False
        conf.dump(str(out/'resolved_config.py'))
        old_result=read('reports/m8j11_sources/full/result.json')
        assert sha(out/'resolved_config.py')==old_result['resolved_config_sha256'],'ResolvedConfigChanged'
        model=build_model(conf.model,train_cfg=conf.get('train_cfg'),test_cfg=conf.get('test_cfg'))
        checkpoint=rt/'artifacts/stream_petr_r50_flash_704_bs2_seq_90e.pth'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(payload['state_dict'],strict=True);del payload
        model.float().cuda().eval()
        for param in model.parameters():param.requires_grad_(False)
        assert all(param.dtype==torch.float32 for param in model.parameters())
        initial=snapshot(model.state_dict());result['initial_state_sha256']=fingerprint(initial)
        assert result['initial_state_sha256']==old_result['initial_state_sha256'],'InitialStateMismatch'
        result.update(checkpoint_sha256=sha(checkpoint),resolved_config_sha256=sha(out/'resolved_config.py'),
            fp16_enabled_flags={n:bool(m.fp16_enabled) for n,m in model.named_modules() if hasattr(m,'fp16_enabled')},
            modes='all eval; no optimizer/scaler/hooks constructed; inherited config optimizer section unused')
        assert result['fp16_enabled_flags']==old_result['fp16_enabled_flags']
        reset_official(model)
        monitor=HistoryIntervention(model,{'T1':'cold','T2':'midpoint','T3':'normal'}[a.condition],p['atol'],p['rtol'])
        raw_holder={};dtypes=set()
        def headhook(mod,inp,output):
            for k in ('all_cls_scores','all_bbox_preds'):
                assert output[k].shape==(6,1,900,10)
                assert output[k].dtype==torch.float32 and torch.isfinite(output[k]).all()
            raw_holder['raw']=output
        model.pts_bbox_head.register_forward_hook(headhook)
        def dtypehook(mod,inp,output):
            for x in inp:
                if torch.is_tensor(x) and x.is_floating_point():
                    dtypes.add(str(x.dtype));assert x.dtype==torch.float32
        for mod in model.modules():
            if isinstance(mod,(torch.nn.Conv2d,torch.nn.Linear,torch.nn.MultiheadAttention)):mod.register_forward_hook(dtypehook)
        original_nan_to_num=torch.nan_to_num
        def checked_nan_to_num(x,*args,**kwargs):
            assert torch.isfinite(x).all(),'NonfiniteBeforeNanToNum'
            return original_nan_to_num(x,*args,**kwargs)
        torch.nan_to_num=checked_nan_to_num
        with (out/'memory.jsonl').open('x') as memorylog:
            for row in rows:
                i=row['index'];frame=prepared[i]
                assert all(frame[k]==row[k] for k in ('index','sample_token','scene_token','timestamp_us'))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                data,meta=load(frame);torch.cuda.synchronize();loaded=time.perf_counter()
                monitor.before(row)
                reservation=reserve(root/'forward_budget.jsonl',result['run_id'],a.condition,i,
                                    p['budget']['total_max'],p['budget']['per_condition'][a.condition])
                ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev0.record()
                result['actual_forwards']+=1
                with torch.no_grad(),torch.autocast('cuda',enabled=False):answer=model.simple_test(meta,**data)[0]['pts_bbox']
                ev1.record();ev1.synchronize()
                state=monitor.after()
                boxes=answer['boxes_3d'].tensor.detach().cpu().numpy()
                scores=answer['scores_3d'].cpu().numpy();labels=answer['labels_3d'].cpu().numpy();raw=raw_holder.pop('raw')
                assert np.isfinite(scores).all() and (boxes[:,3:6]>0).all() and len(scores)<=300
                ss,idx=raw['all_cls_scores'][-1,0].sigmoid().flatten().topk(300);rr=raw['all_bbox_preds'][-1,0,idx//10]
                coder=model.pts_bbox_head.bbox_coder
                keep=((rr[:,:3]>=rr.new_tensor(coder.post_center_range[:3]))&(rr[:,:3]<=rr.new_tensor(coder.post_center_range[3:]))).all(1)
                np.testing.assert_array_equal(ss[keep].cpu().numpy(),scores);np.testing.assert_array_equal((idx%10)[keep].cpu().numpy(),labels)
                np.testing.assert_allclose(rr[keep,2].cpu().numpy(),boxes[:,2]+boxes[:,5]/2,atol=1e-5)
                np.testing.assert_array_equal(rr[keep,8:10].cpu().numpy(),boxes[:,7:9])
                record=dict(boxes=to_legacy_boxes(boxes),native_boxes=boxes,scores=scores,labels=labels,
                    query_indices=(idx//10)[keep].cpu().numpy(),raw_cls=raw['all_cls_scores'][-1,0].cpu().numpy(),
                    raw_bbox=raw['all_bbox_preds'][-1,0].cpu().numpy())
                np.savez_compressed(out/('{:02}.npz'.format(i)),**record)
                references=[]
                if row['scene_index']==0 or (a.condition=='T2' and row['scene_index']<row['reset_scene_index']):
                    references.append(('T0',p['T0_cache'][i]['path']))
                if a.condition=='T1' and row['scene_index']==row['reset_scene_index']:
                    references.append(('T2_reset',str(root/'T2'/('{:02}.npz'.format(i)))))
                for name,path in references:
                    error=compare_arrays(record,load_cache(path),p['atol'],p['rtol'])
                    result['checks'].append(dict(index=i,reference=name,max_abs_error=error))
                assert_exact(initial,model.state_dict(),'FrozenWeightsAndRegisteredBuffers')
                assert all(param.grad is None for param in model.parameters()) and all(not m.training for m in model.modules())
                state.update(run_id=result['run_id'],**row)
                memorylog.write(__import__('json').dumps(state)+'\n');memorylog.flush()
                result['frames'].append(dict(index=i,scene_name=row['scene_name'],scene_index=row['scene_index'],
                    reservation=reservation['reservation'],gpu_instrumented_ms=ev0.elapsed_time(ev1),
                    cache_load_h2d_ms=(loaded-t)*1000,instrumented_frame_wall_ms=(time.perf_counter()-t)*1000,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    actual_prev_exists=state['actual_head_prev_exists'],observed_slots=state['observed_slots_read'],
                    cache_sha256=sha(out/('{:02}.npz'.format(i))),parameters_buffers_unchanged=True))
                print(a.condition,i,'prev',state['actual_head_prev_exists'],'history',state['observed_slots_read'],flush=True)
                del data,meta,answer,raw,rr,ss,ev0,ev1
        monitor.close();torch.nan_to_num=original_nan_to_num
        result.update(status='InterventionSequencePassed',final_state_sha256=fingerprint(model.state_dict()),
                      observed_compute_dtypes=sorted(dtypes),memory_log_sha256=sha(out/'memory.jsonl'))
        assert result['actual_forwards']==len(rows)
    except Exception as e:
        result.update(status='FailedStopped',error_type=type(e).__name__,error=str(e));raise
    finally:
        result['elapsed_s']=time.perf_counter()-start
        write(out/'result.json',result)
        write('reports/m8j11a_sources/'+a.condition+'_execution.json',{k:v for k,v in result.items() if k not in ('frames','checks')})


if __name__=='__main__':main()
