"""Frozen StreamPETR sequential inference; private outputs, no optimizer or backward."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch,mmcv,mmdet,mmdet3d
import projects.mmdet3d_plugin
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from datasets.m8j11_contract import INPUT_FIELDS,to_legacy_boxes
from training.m8j6_retry_state import snapshot,fingerprint,assert_exact


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    p=os.fsdecode(args[0])
    if '/m8j6-nuscenes-mini/data/' in p or '/supervision/' in p:
        raise PermissionError('NoRawGTOrSensorPayloadInInference')
def load(frame):
    p=Path(frame['path']);assert sha(p)==frame['sha256']
    with np.load(p,allow_pickle=False) as z:
        assert set(z.files)==INPUT_FIELDS
        data={k:torch.from_numpy(np.array(z[k],copy=True)).cuda().unsqueeze(0) for k in INPUT_FIELDS-{'images'}}
        data['img']=torch.from_numpy(z['images'].copy()).cuda()
    meta=[dict(scene_token=frame['scene_token'],box_type_3d=LiDARInstance3DBoxes,
        img_shape=[(256,704,3)]*6,pad_shape=[(256,704,3)]*6,ori_shape=[(900,1600,3)]*6)]
    assert data['img'].shape==(1,6,3,256,704)
    return data,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['probe','full'],required=True);a=ap.parse_args()
    os.umask(0o077);c=json.loads(Path('configs/m8j11_protocol.json').read_text());rt=Path(c['runtime_root']);root=Path(c['private_root'])
    out=root/a.phase;out.mkdir(exist_ok=False);rep=Path('reports/m8j11_sources')/a.phase;rep.mkdir(exist_ok=False)
    if a.phase=='full':assert json.loads(Path('reports/m8j11_sources/probe_export.json').read_text())['status']=='ShortSequenceExportPassed'
    mf=json.loads((root/'prepared/manifest.json').read_text());frames=mf['frames']
    result=dict(status='Running',phase=a.phase,started_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=os.popen('git rev-parse HEAD').read().strip(),checkpoint_sha256=c['initial_asset_sha256'],
        optimizer_steps=0,backward_calls=0,precision='FP32_tensor_compute_FP64_timestamps',AMP=False,TF32=False,
        torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmdet3d=mmdet3d.__version__,GPU=torch.cuda.get_device_name(),
        warmup='Probe separate process; full starts fresh scene memory; no warmup frames dropped',frames=[])
    start=time.perf_counter();sys.addaudithook(guard)
    try:
        random.seed(20261029);np.random.seed(20261029);torch.manual_seed(20261029);torch.cuda.manual_seed_all(20261029)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction(.9)
        conf=mmcv.Config.fromfile(str(rt/'runtime-source'/c['config']))
        conf.model.img_backbone.pretrained=None
        attn=conf.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1]
        assert attn.type=='PETRMultiheadFlashAttention';attn.type='PETRMultiheadAttention';attn.fp16=False
        conf.dump(str(out/'resolved_config.py'))
        model=build_model(conf.model,train_cfg=conf.get('train_cfg'),test_cfg=conf.get('test_cfg'))
        cp=rt/'artifacts/stream_petr_r50_flash_704_bs2_seq_90e.pth';assert sha(cp)==c['initial_asset_sha256']
        payload=torch.load(cp,map_location='cpu',weights_only=True)
        model.load_state_dict(payload['state_dict'],strict=True);del payload
        model.float().cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        # No wrap_fp16_model, optimizer, loss, scaler or training hook constructed.
        assert all(p.dtype==torch.float32 for p in model.parameters())
        initial=snapshot(model.state_dict());initial_hash=fingerprint(initial)
        result.update(parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial_hash,
            strict_checkpoint_load=True,model_mode='eval_all_modules',resolved_config_sha256=sha(out/'resolved_config.py'),
            fp16_enabled_flags={n:bool(m.fp16_enabled) for n,m in model.named_modules() if hasattr(m,'fp16_enabled')})
        raw_holder={};dtype_seen=set()
        def headhook(mod,inp,output):
            raw_holder['raw']=output
            for k in ('all_cls_scores','all_bbox_preds'):
                assert output[k].shape==(6,1,900,10);assert output[k].dtype==torch.float32 and torch.isfinite(output[k]).all()
        model.pts_bbox_head.register_forward_hook(headhook)
        def dtypehook(mod,inp,output):
            for x in inp:
                if torch.is_tensor(x) and x.is_floating_point():dtype_seen.add(str(x.dtype));assert x.dtype==torch.float32
        for mod in model.modules():
            if isinstance(mod,(torch.nn.Conv2d,torch.nn.Linear,torch.nn.MultiheadAttention)):mod.register_forward_hook(dtypehook)
        memory_names=['memory_embedding','memory_reference_point','memory_timestamp','memory_egopose','memory_velo']
        def reset():
            model.prev_scene_token=None;model.pts_bbox_head.reset_memory()
            assert all(getattr(model.pts_bbox_head,n) is None for n in memory_names)
        schedule=[('first',[0,1,2,40,41]),('repeat',[0,1,2,40,41]),('fresh_second_scene',[40,41])] if a.phase=='probe' else [('formal',list(range(81)))]
        reference={};repeat_error=0.
        original_nan_to_num=torch.nan_to_num
        def checked_nan_to_num(x,*args,**kwargs):
            assert torch.isfinite(x).all(),'Nonfinite before official nan_to_num'
            return original_nan_to_num(x,*args,**kwargs)
        torch.nan_to_num=checked_nan_to_num
        for run,ids in schedule:
            reset();last_scene=None
            for i in ids:
                frame=frames[i];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                data,meta=load(frame);torch.cuda.synchronize();loaded=time.perf_counter()
                ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
                ev0.record()
                with torch.no_grad(),torch.autocast('cuda',enabled=False):answer=model.simple_test(meta,**data)[0]['pts_bbox']
                ev1.record();ev1.synchronize()
                boxes=answer['boxes_3d'].tensor.detach().cpu().numpy();scores=answer['scores_3d'].cpu().numpy();labels=answer['labels_3d'].cpu().numpy()
                wall=time.perf_counter()-t;raw=raw_holder.pop('raw')
                legacy=to_legacy_boxes(boxes);assert np.isfinite(scores).all() and (boxes[:,3:6]>0).all() and len(scores)<=300
                # Raw -> official decoded: new layout xyz/logLWH; decoded bottom z.
                coder=model.pts_bbox_head.bbox_coder
                ss,idx=raw['all_cls_scores'][-1,0].sigmoid().flatten().topk(300);rr=raw['all_bbox_preds'][-1,0,idx//10]
                lo=rr.new_tensor(coder.post_center_range[:3]);hi=rr.new_tensor(coder.post_center_range[3:])
                keep=((rr[:,:3]>=lo)&(rr[:,:3]<=hi)).all(1)
                np.testing.assert_array_equal(ss[keep].cpu().numpy(),scores);np.testing.assert_array_equal((idx%10)[keep].cpu().numpy(),labels)
                np.testing.assert_allclose(rr[keep,2].cpu().numpy(),boxes[:,2]+boxes[:,5]/2,atol=1e-5)
                np.testing.assert_array_equal(rr[keep,8:10].cpu().numpy(),boxes[:,7:9])
                record=dict(boxes=legacy,native_boxes=boxes,scores=scores,labels=labels,query_indices=(idx//10)[keep].cpu().numpy(),
                    raw_cls=raw['all_cls_scores'][-1,0].cpu().numpy(),raw_bbox=raw['all_bbox_preds'][-1,0].cpu().numpy())
                if run in ('first','formal'):
                    if a.phase=='probe':reference[i]=record
                    np.savez_compressed(out/f'{i:02}.npz',**record)
                else:
                    for k in record:
                        err=float(np.max(abs(record[k]-reference[i][k]))) if record[k].size else 0.
                        repeat_error=max(repeat_error,err)
                        np.testing.assert_allclose(record[k],reference[i][k],atol=1e-5,rtol=1e-5)
                assert_exact(initial,model.state_dict(),'FrozenWeightsAndRegisteredBuffers')
                assert all(p.grad is None for p in model.parameters());assert all(not m.training for m in model.modules())
                memory={n:dict(shape=list(getattr(model.pts_bbox_head,n).shape),dtype=str(getattr(model.pts_bbox_head,n).dtype)) for n in memory_names}
                assert all(torch.isfinite(getattr(model.pts_bbox_head,n)).all() for n in memory_names)
                row=dict(index=i,sample_token=frame['sample_token'],scene_name=frame['scene_name'],timestamp_us=frame['timestamp_us'],run=run,
                    cold_scene=last_scene!=frame['scene_token'],decoded=len(scores),gpu_ms=ev0.elapsed_time(ev1),cached_input_to_cpu_prediction_ms=wall*1000,
                    cache_load_transfer_ms=(loaded-t)*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    memory=memory,parameters_buffers_unchanged=True)
                result['frames'].append(row);last_scene=frame['scene_token'];print(run,i,len(scores),row['gpu_ms'],flush=True)
                del data,meta,answer,raw,rr,ss,ev0,ev1
        result.update(status='ReadOnlySequencePassed',repeat_max_abs_error=repeat_error,compute_input_dtypes=sorted(dtype_seen),
            final_state_sha256=fingerprint(model.state_dict()),parameters_buffers_unchanged=True,reset_verified=True)
    except Exception as e:result.update(status='FailedStopped',error_type=type(e).__name__,error=str(e));raise
    finally:
        result['elapsed_s']=time.perf_counter()-start;write(rep/'result.json',result)
if __name__=='__main__':main()
