"""Same cached-input timing boundaries as candidate; A unchanged and no GT read."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from tools import m8j6_gpu_actual as old
from training.m8j6_retry_state import snapshot,assert_exact,fingerprint
from training.m8j7_metrics import distribution

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='reports/m8j11_sources/A_timing.json');args=ap.parse_args()
    c=json.loads(Path('configs/m8j6_protocol.json').read_text());j=json.loads(Path('configs/m8j9_retry1_protocol.json').read_text())
    frames=json.loads(Path(j['manifest']).read_text())['frames'];rt=Path(j['runtime'])
    torch.set_num_threads(2);torch.manual_seed(20261029);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=old.build(rt,c,Path(j['A']['path']));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    initial=snapshot(model.state_dict());rows=[];input_manifest=json.loads(Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared/manifest.json').read_text());root=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared/rgb_input')
    for f in frames:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        ip=root/f'{f["index"]:02}.npz';assert hashlib.sha256(ip.read_bytes()).hexdigest()==input_manifest['frames'][f['index']]['input_sha256']
        with np.load(ip,allow_pickle=False) as z:
            img=torch.from_numpy(z['images'].copy()).cuda();meta=[dict(lidar2img=z['lidar2img'].copy(),img_timestamp=(z['camera_timestamp_us']/1e6).tolist(),img_shape=[(512,1408,3)]*6,pad_shape=[(512,1408,3)]*6,ori_shape=[(900,1600,3)]*6,box_type_3d=old.LiDARInstance3DBoxes,scale_factor=1.,flip=False)]
        torch.cuda.synchronize();loaded=time.perf_counter();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record()
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            raw=old.forward(model,img,meta);boxes,scores,labels=model.pts_bbox_head.get_bboxes(raw,meta)[0]
        e.record();e.synchronize();arrays=(boxes.tensor.cpu().numpy(),scores.cpu().numpy(),labels.cpu().numpy());wall=time.perf_counter()-t
        # Verify timing run reproduces the exact frozen quality asset (not new selection).
        with np.load(Path(j['cache_root'])/f'{f["index"]:02}.npz',allow_pickle=False) as z:
            for key,value in zip(('boxes','scores','labels'),arrays):np.testing.assert_allclose(value,z[key],atol=1e-5,rtol=1e-5)
        assert_exact(initial,model.state_dict(),'ATimingFrozen')
        rows.append(dict(index=f['index'],scene_name=f['scene_name'],cold_scene=f['index'] in (0,40),gpu_ms=s.elapsed_time(e),cached_input_to_cpu_prediction_ms=wall*1000,
            cache_load_transfer_ms=(loaded-t)*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(f['index'],rows[-1]['gpu_ms'],flush=True)
        del img,meta,raw,boxes,scores,labels
    out=dict(status='ReadOnlyATimingPassed',frames=rows,parameters=sum(p.numel() for p in model.parameters()),FP32=True,model_state_sha256=fingerprint(model.state_dict()),cache_equivalence=True,optimizer_steps=0,backward_calls=0,
        cold_note='A is single-frame; indices0/40 reported for same scene grouping; only index0 is process first forward',summary={})
    for key,rr in [('all',rows),('cold_scene',[r for r in rows if r['cold_scene']]),('subsequent',[r for r in rows if not r['cold_scene']])]:
        out['summary'][key]={k:distribution([r[k] for r in rr]) for k in ('gpu_ms','cached_input_to_cpu_prediction_ms','cache_load_transfer_ms','peak_allocated_bytes','peak_reserved_bytes')}
    with Path(args.output).open('x') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()
