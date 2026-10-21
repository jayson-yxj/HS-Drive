"""Actual original M4 training frames only; P8 and original S16 near replay."""
import json,time
from pathlib import Path
import torch
from datasets.training import HSDriveDataset
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb
from tools.m5g_depth_data import conditioned
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    out=Path('runs/m5h_replay');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());net=encoder_load();records=[];start=time.perf_counter()
    for split in ['train','val']:
        for stride in [8,16]:
            ds=HSDriveDataset(man['config']['index'],split,feature_stride=stride);lookup={w[-1]['sample_id']:i for i,w in enumerate(ds.windows)}
            for sid in man[split+'_sample_ids']:
                item=ds[lookup[sid]];x={k:v[None].cuda()for k,v in item.inputs.items()};c=cache_p8(net,x)if stride==8 else cache_rgb(net,x);f=conditioned(net,c);d=item.targets
                dep=d['depth_m'];valid=d['depth_valid']&(dep>=1.5)&(dep<10)
                path=out/(sid.replace('/','_')+f'_s{stride}.pt');torch.save(dict(rgb_conditioned=f[0].cpu(),depth=dep,valid=valid),path);records.append(dict(sample_id=sid,split=split,stride=stride,path=str(path),sha256=file_hash(path),near_n=int(valid.sum())));print(split,stride,sid,flush=True)
    atomic_json(out/'index.json',dict(kind='actual_GPU_original_M4_frozen_RGB_replay',records=records,original_manifest_sha256=file_hash('runs/m4_expanded_joint/manifest.json'),source_sha256=file_hash(__file__),elapsed_s=time.perf_counter()-start,limits='Original selected train ids only in optimization; original val ids evaluationonly. Temporal driving frames correlated. P8 extra diagnostic, S16 original geometry. No class/actor sampling.'))
if __name__=='__main__':main()
