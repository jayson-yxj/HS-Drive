"""RGB-only conditioned features stored separately from depth supervision arrays."""
import json,time
from pathlib import Path
import torch,numpy as np
from models.m5e_paths import geometry
from models.m5d_stride8 import cache_p8
from tools.m5a_learning import encoder_load
from tools.m5g_common import rows
from tools.m5f_common import inputs,pixel_label
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def supervision(net,c,d):
    dep=torch.from_numpy(d['depth'][:,4::8,4::8][None]).cuda();v=d['valid'][:,4::8,4::8].copy();full=d['depth'];patch=np.stack([full[:,4+dy:384:8,4+dx:640:8]for dy in [-1,0,1]for dx in [-1,0,1]])
    v &= np.isfinite(patch).all(0)&((patch.max(0)-patch.min(0))<=.5);v=torch.from_numpy(v[None]).cuda();g=geometry(net,c,dep);spatial=g['same_layer']&(g['xy']<=1);return dep,v,g,spatial


def conditioned(net,c):
    f=c['features'];b,n,ch,h,w=f.shape;K=c['K'];T=c['T'];ih,iw=c['image_hw'];camera=torch.stack((K[...,0,0]/iw,K[...,1,1]/ih,K[...,0,2]/iw,K[...,1,2]/ih),-1);ext=torch.cat((T[...,:3,:3].flatten(-2),T[...,:3,3]/10),-1)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cond=net.camera_mlp(torch.cat((camera,ext),-1)).reshape(b,n,ch,1,1)
    return f+cond


def main():
    out=Path('runs/m5g_depth_cache');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);net=encoder_load();records=[]
    for row in rows():
        x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);dep,v,g,sp=supervision(net,c,d);p=out/(row['sample_id'].replace('/','_')+'.pt');torch.save(dict(rgb_conditioned=conditioned(net,c)[0].cpu(),depth=dep[0].cpu(),valid=v[0].cpu(),target=torch.from_numpy(d['target'][:,4::8,4::8]),spatial=sp[0].cpu()),p);records.append(dict(sample_id=row['sample_id'],path=str(p),sha256=file_hash(p)));print(row['sample_id'],flush=True)
    atomic_json(out/'index.json',dict(records=records,source_sha256=file_hash(__file__),limits='CachedRGBconditionedfield distinctGTsupervision; noGTinputtoDepthHead. P8originalencoder frozen.'))

if __name__=='__main__':main()
