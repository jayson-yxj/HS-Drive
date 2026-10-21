"""Dedicated synchronized sixcamera timing after all evaluation GPU jobs end."""
import json,time,subprocess
from pathlib import Path
import numpy as np,torch
from tools.m5f_common import inputs,checkpoint
from tools.m5a_learning import encoder_load
from models.m5f_pixel import PixelHead,ForegroundModel
from models.m5f_lift import lift
from models.m5b_cached_encoder import cache_rgb
from models.m5d_stride8 import cache_p8
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_isolated_latency.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels/index.json').read_text())['records'];inp=inputs(rows[3]);study=json.loads(Path('reports/m5f_image_foreground_training.json').read_text())['records'];net=encoder_load();records=[];process=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    for r in study:
        ck=checkpoint(r);pn=encoder_load();pn.encoder.lateral.load_state_dict(ck['lateral']);pn.encoder.output.load_state_dict(ck['output']);head=PixelHead().cuda();head.load_state_dict(ck['head']);model=ForegroundModel(pn.encoder,head,r['stride']).eval();times=[];torch.cuda.reset_peak_memory_stats()
        for i in range(35):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.cuda.synchronize();t0=time.perf_counter();p=model(**inp).float().sigmoid();torch.cuda.synchronize();t1=time.perf_counter();s=r['stride'];c=(cache_rgb if s==16 else cache_p8)(net,inp);torch.cuda.synchronize();t2=time.perf_counter();out=lift(net.splat,p[:,:,s//2::s,s//2::s],c['probs'],c['K'],c['T'],c['image_hw']);torch.cuda.synchronize();t3=time.perf_counter()
            if i>=5:times.append(dict(pixel_ms=(t1-t0)*1000,depth_encoder_ms=(t2-t1)*1000,lift_ms=(t3-t2)*1000,total_ms=(t3-t0)*1000))
        records.append(dict(name=r['name'],seed=r['seed'],samples=times,summary={k:dict(mean=float(np.mean([t[k] for t in times])),p50=float(np.median([t[k] for t in times])),p95=float(np.quantile([t[k] for t in times],.95)),p99=float(np.quantile([t[k] for t in times],.99)),max=float(max(t[k] for t in times))) for k in times[0]},peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model,pn,head
    atomic_json(dest,dict(kind='actual_isolated_RTX4090_CUDA_synchronized_inference',records=records,gpu=torch.cuda.get_device_name(),other_process_snapshot=process,source_sha256=file_hash(__file__),limits='Onefixedrecordedsixcamera sample, 5warmup30timedrepeat each. Pixel+duplicatefrozendepthencoders+FP32lift. Excludesdisk/network/CARLA/control. Dedicatedtiming supplements full36frame shadow timings. Not strict20Hzclaim.'))

if __name__=='__main__':main()
