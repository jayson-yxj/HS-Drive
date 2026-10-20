"""Matched per-ray spatial probability metrics; expected error != error of expectation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tools.m5e_common import ROOT,sidecars,load_checkpoint
from tools.m5a_learning import encoder_load,stack
from datasets.m5a_training import VisualEvidenceDataset
from models.m5b_cached_encoder import cache_rgb
from models.m5e_paths import depth_logits,geometry
from models.view_transform import feature_uv,unproject
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--index',default=ROOT);pa.add_argument('--split',default='val');pa.add_argument('--output',default='reports/m5e_depth_spatial_probability.json');pa.add_argument('--models',action='store_true');a=pa.parse_args();dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);ds=VisualEvidenceDataset(a.index,a.split,supervision=False);sup=sidecars(a.index,ds.rows,'m5e_supervision.json');net=encoder_load();caches=[cache_rgb(net,stack([ds[i]],'inputs')) for i in range(len(ds))];models=[dict(name='original_depth',seed=0,checkpoint=None)]
    if a.models:models+=json.loads(Path('reports/m5e_model_ablation.json').read_text())['records']
    records=[]
    for model in models:
        net=encoder_load()
        if model['checkpoint']:net.depth_head.load_state_dict(load_checkpoint(model['checkpoint'])['depth'])
        for row,c,s in zip(ds.rows,caches,sup):
            depth=torch.from_numpy(s['depth16'][None]).cuda();reliable=torch.from_numpy(s['valid16'][None]).cuda();actor=torch.from_numpy(s['actor16'][None]).cuda();g=geometry(net,c,depth);b,n,h,w=depth.shape;centers=net.splat.depth_centers
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=depth_logits(net,c).float().softmax(2)
            nearest=(depth[:,:,None]-centers[None,None,:,None,None]).abs().argmin(2);onehot=torch.nn.functional.one_hot(nearest,48).permute(0,1,4,2,3).float();expected=(p*centers[None,None,:,None,None]).sum(2)
            with torch.autocast('cuda',enabled=False):ep=unproject(feature_uv(h,w,c['image_hw'],'cuda').round().expand(b,n,h,w,2),expected,c['K'],c['T'])
            erank,einside=net.splat.ranks(ep);entropy=-(p*p.clamp_min(1e-12).log()).sum(2)/np.log(48)
            for mode,weights in [('continuous_gt',None),('nearest_lid',onehot),('learned_probability',p),('expected_depth',None)]:
                if weights is not None:
                    dz=(g['points']-g['gt'][:,:,None]).norm(dim=-1);same=g['same_layer'];xy=g['xy'];inside=g['bin_inside'];rank=g['ranks']
                    delta_cell=(torch.floor((g['points'][...,:2]+torch.tensor([20,30],device='cuda'))/.5)-torch.floor((g['gt'][:,:,None,...,:2]+torch.tensor([20,30],device='cuda'))/.5)).abs()
                    neighbor=same&(delta_cell<=1).all(-1)&inside
                    vals=dict(correct_layer_mass=(weights*same).sum(2),gt_voxel_mass=(weights*(rank==g['ids'][:,:,None])*inside).sum(2),neighbor_voxel_mass=(weights*neighbor).sum(2),expected_abs_xy_m=(weights*xy).sum(2),expected_abs_xyz_m=(weights*dz).sum(2),out_of_bev_mass=(weights*~inside).sum(2),outside_gt_entering_bev_mass=(weights*inside).sum(2)*~g['inside'])
                    for t in [.5,1.,2.]:vals[f'xy_{t}m_mass']=(weights*(xy<=t)*same*inside).sum(2)
                else:
                    xy=torch.zeros_like(depth) if mode=='continuous_gt' else (ep[...,:2]-g['gt'][...,:2]).norm(dim=-1);same=torch.ones_like(reliable) if mode=='continuous_gt' else torch.bucketize(ep[...,2].contiguous(),net.splat.z_edges,right=True)==torch.bucketize(g['gt'][...,2].contiguous(),net.splat.z_edges,right=True);inside=g['inside'] if mode=='continuous_gt' else einside;rank=g['ids'] if mode=='continuous_gt' else erank
                    point=g['gt'] if mode=='continuous_gt' else ep
                    delta_cell=(torch.floor((point[...,:2]+torch.tensor([20,30],device='cuda'))/.5)-torch.floor((g['gt'][...,:2]+torch.tensor([20,30],device='cuda'))/.5)).abs()
                    vals=dict(correct_layer_mass=same.float(),gt_voxel_mass=((rank==g['ids'])&inside).float(),neighbor_voxel_mass=(same&(delta_cell<=1).all(-1)&inside).float(),expected_abs_xy_m=xy,expected_abs_xyz_m=torch.zeros_like(depth) if mode=='continuous_gt' else (ep-g['gt']).norm(dim=-1),out_of_bev_mass=(~inside).float(),outside_gt_entering_bev_mass=(inside&~g['inside']).float())
                    for t in [.5,1.,2.]:vals[f'xy_{t}m_mass']=((xy<=t)&same&inside).float()
                vals.update(entropy=entropy,signed_depth_bias_m=expected-depth,expected_depth_point_xy_m=(ep[...,:2]-g['gt'][...,:2]).norm(dim=-1))
                for target,m0 in [('all_reliable',reliable),('actor',reliable&actor)]:
                    for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]:
                        m=m0&(g['rad']>=lo)&(g['rad']<hi)
                        if not m.any():continue
                        records.append(dict(name=model['name'],seed=model['seed'],group=row['group'],configuration=row['configuration'],target=target,distance=f'{lo}_{hi}',mode=mode,n=int(m.sum()),**{k:float(v[m].mean()) for k,v in vals.items()}))
        print(model['name'],model['seed'],flush=True)
    summary=[]
    for name,seed,target,mode,band in sorted({(r['name'],r['seed'],r['target'],r['mode'],r['distance']) for r in records}):
        rr=[r for r in records if (r['name'],r['seed'],r['target'],r['mode'],r['distance'])==(name,seed,target,mode,band)];count=sum(r['n'] for r in rr);summary.append(dict(name=name,seed=seed,target=target,mode=mode,distance=band,n=count,groups=len({r['group'] for r in rr}),**{k:sum(r[k]*r['n'] for r in rr)/count for k in vals}))
    atomic_json(dest,dict(kind='actual_GPU_fixed_UV_sameMask_spatial_probability',records=records,summary=summary,source_sha256=file_hash(__file__),scope=a.split,limits='FeatureUV rounded identically for all diagnostic projections; probabilities belong to original continuous featurecenter. Neighbor metric same-layer3x3cells includingGTcell. Entropy/bias columns always learned distribution diagnostic even on oracle rows. OutsideGT enteringBEV reported separately, not proof falsefree. No additive causal subtraction.'))


if __name__=='__main__':main()
