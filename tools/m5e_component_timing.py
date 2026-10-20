"""Separate actual Splat, BEV encoder and Head CUDA time; fixed val frames."""
import json,time
from pathlib import Path
import torch
import numpy as np
from tools.m5e_common import ROOT
from tools.m5a_learning import encoder_load,stack
from datasets.m5a_training import VisualEvidenceDataset
from models.m5b_cached_encoder import cache_rgb
from models.m5d_stride8 import cache_p8
from models.m5e_paths import factorize
from models.m5c_obstacle_head import VisibleObstacleHead
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5e_component_timing.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();head=VisibleObstacleHead().cuda().eval();ds=VisualEvidenceDataset(ROOT,'val',supervision=False);records=[]
    for i in [0,3,6]:
        inp=stack([ds[i]],'inputs');c16=cache_rgb(net,inp);c8=cache_p8(net,inp)
        for mode in ['s16','p8_s16depth','p8','p8_down16']:
            c=factorize(c16,c8,mode);f=c['features'];b,n,chan,h,w=f.shape
            for rep in range(12):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    ctx=net.context_head(f.flatten(0,1)).reshape(b,n,64,h,w);torch.cuda.synchronize();t0=time.perf_counter();height,mass=net.splat(ctx,c['probs'],c['K'],c['T'],c['image_hw']);torch.cuda.synchronize();t1=time.perf_counter();bev=net.bev_encoder(torch.cat((height.flatten(1,2),torch.log1p(mass)),1));torch.cuda.synchronize();t2=time.perf_counter();head(bev);torch.cuda.synchronize();t3=time.perf_counter()
                if rep>=2:records.append(dict(sample_id=ds.rows[i]['sample_id'],mode=mode,splat_ms=(t1-t0)*1000,bev_encoder_ms=(t2-t1)*1000,evidence_head_ms=(t3-t2)*1000))
    summary={m:{k:dict(mean=float(np.mean([r[k] for r in records if r['mode']==m])),p95=float(np.quantile([r[k] for r in records if r['mode']==m],.95))) for k in ['splat_ms','bev_encoder_ms','evidence_head_ms']} for m in ['s16','p8_s16depth','p8','p8_down16']}
    atomic_json(dest,dict(kind='actual_RTX4090_CUDA_synchronized_component_timing',records=records,summary=summary,source_sha256=file_hash(__file__),limits='FrozenM4 encoder weights and untrained fixedarchitecture head; operationcost only noqualityresults. Threevalframes samegroup. CARLA idle, noCPUlabeljob while measured if contemporaneouslogconfirms; no hardrealtime claim. Component syncadds launchbarriers.'))

if __name__=='__main__':main()
