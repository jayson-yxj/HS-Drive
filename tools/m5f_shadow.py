"""Unmasked legalRGB-only shadow with synchronized latency; never sends controls."""
import json,time
from pathlib import Path
import numpy as np,torch
from models.m5f_pixel import PixelHead,ForegroundModel
from models.m5f_lift import lift
from models.m5b_cached_encoder import cache_rgb
from models.m5d_stride8 import cache_p8
from models.m5e_paths import factorize
from tools.m5a_learning import encoder_load
from tools.m5f_common import inputs,pixel_label,checkpoint
from tools.m5f_factorization import reference
from diagnostics.m5f_metrics import histogram,metrics
from diagnostics.m5c_localization import matching
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_shadow_evaluation.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);root=Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels');rows=json.loads((root/'index.json').read_text())['records'];study=json.loads(Path('reports/m5f_image_foreground_training.json').read_text());choices=json.loads(Path('reports/m5f_geometry_selection.json').read_text());records=[];latency=[];common=[];net=encoder_load()
    for record in study['records']:
        ck=checkpoint(record);pn=encoder_load();pn.encoder.lateral.load_state_dict(ck['lateral']);pn.encoder.output.load_state_dict(ck['output']);head=PixelHead().cuda();head.load_state_dict(ck['head']);model=ForegroundModel(pn.encoder,head,record['stride']).eval();th=choices[f"{record['name']}_{record['seed']}_D_learned2d_learneddepth"]['threshold'];hist=np.zeros((2,256),np.int64);frames=[];times=[]
        torch.cuda.reset_peak_memory_stats()
        for i,row in enumerate(rows):
            inp=inputs(row);start=time.perf_counter()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.cuda.synchronize();start=time.perf_counter();p=model(**inp).float().sigmoid();torch.cuda.synchronize();t1=time.perf_counter();c=(cache_rgb if record['stride']==16 else cache_p8)(net,inp);torch.cuda.synchronize();t2=time.perf_counter();s=record['stride'];fg=p[:,:,s//2::s,s//2::s];out=lift(net.splat,fg,c['probs'],c['K'],c['T'],c['image_hw']);torch.cuda.synchronize();t3=time.perf_counter()
            times.append(dict(pixel_ms=(t1-start)*1000,depth_encoder_ms=(t2-t1)*1000,lift_ms=(t3-t2)*1000,total_ms=(t3-start)*1000));score=out['event_score'][0].cpu().numpy();d=pixel_label(row);target,valid,actors=reference(d,inp,row,row['camera_names']);h=histogram(score,target,valid);hist+=h;pred=score>=int(th*256)/256;matches=matching(pred&valid,target&valid)
            frames.append(dict(sample_id=row['sample_id'],group=row['group'],configuration=row['configuration'],metrics=metrics(h,th),matching=matches,unknown_high_fraction=float(pred[~valid].mean()),actor_visible=sum(int(a['mask'].sum())>0 for a in actors),oracle_state_reference=row['raw_frame'],control_sent=False))
            # Common S16 UV/depth for both S16 and P8Pixel: isolate pixel recognition.
            if record['stride']==8:
                c16=cache_rgb(net,inp)
                with torch.no_grad():co=lift(net.splat,p[:,:,8::16,8::16],c16['probs'],c16['K'],c16['T'],c16['image_hw'])
                cs=co['event_score'][0].cpu().numpy();common.append(dict(name=record['name'],seed=record['seed'],sample_id=row['sample_id'],histogram=histogram(cs,target,valid).tolist(),matching=matching((cs>=int(th*256)/256)&valid,target&valid)))
        records.append(dict(name=record['name'],seed=record['seed'],threshold=th,metrics=metrics(hist,th),frames=frames));latency.append(dict(name=record['name'],seed=record['seed'],timings=times,summary={k:dict(p50=float(np.median([x[k] for x in times[1:]])),p95=float(np.quantile([x[k] for x in times[1:]],.95)),max=float(max(x[k] for x in times[1:]))) for k in times[0]},peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20));print(record['name'],record['seed'],records[-1]['metrics'],flush=True)
    atomic_json(dest,dict(kind='actual_RTX4090_offline_recorded_RGB_shadow_no_control',records=records,commonS16=common,source_sha256=file_hash(__file__),limits='NoGTinferenceMask; Oracle states onlymetrics. PixelandDepthencoder separatecopies, timing includes redundant computation. CommonS16 nativeDepth comparedP8Pixel features, no newlytrainedhead. Thresholdchosenmaskedvalidationthenusedunmasked maydistribution-shift, never retunedontest. Notvisualdriving success.'))
    atomic_json('reports/m5f_inference_latency.json',dict(kind='actual_CUDA_synchronized_offline_sixcamera_inference',records=latency,gpu=torch.cuda.get_device_name(),source_sha256=file_hash(__file__),limits='Diskdecode excluded. Full6camera pixel+originaldepth+foregroundlift. Firstframewarmupexcluded. CARLAstopped beforebenchmark. Not sharedGPUrender/control or realtimeguarantee.'))

if __name__=='__main__':main()
