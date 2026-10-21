"""All high-cell contribution attribution, additive source categories + overlap flags."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from models.m5j_evidence import evidence
from tools.m5j_common import rows,write
from tools.m5a_learning import encoder_load
from tools.m5f_common import inputs,pixel_label
from models.m5d_stride8 import cache_p8
from models.view_transform import unproject,feature_uv
from tools.m5f_factorization import reference
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='val');a=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();idx=json.loads(Path(f'runs/m5j_{a.split}/index.json').read_text())['records'];records=[]
    for row in idx:
        x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);z=np.load(row['cache']);valid=torch.from_numpy(d['valid'][:,4::8,4::8])[None].cuda();fg=torch.from_numpy(d['target'][:,4::8,4::8])[None].cuda();dep=torch.from_numpy(d['depth'][:,4::8,4::8])[None].cuda();uv=feature_uv(48,80,(384,640),'cuda').round();points=unproject(uv.expand(1,6,48,80,2),dep,c['K'],c['T']);gr,gi=net.splat.ranks(points);truth=z['target'];known=z['valid']
        for seed in [101,102]:
            q=torch.from_numpy(z[f'{seed}_pixel'][None,:,4::8,4::8].astype(np.float32)).cuda();e=evidence(net.splat,q,c['probs'],c['K'],c['T'],c['image_hw'],trace=True);tr=e['trace'];rank=tr['ranks'];inside=tr['inside'];mass=tr['probability']*q[:,:,None];types={'A_wrong2D':(valid&~fg)[:,:,None].expand_as(inside),'B_target_depth_othercell':(valid&fg)[:,:,None]&((rank!=gr[:,:,None])|~gi[:,:,None]),'target_samecell':(valid&fg)[:,:,None]&(rank==gr[:,:,None])&gi[:,:,None],'E_source_unknown':(~valid)[:,:,None].expand_as(inside)};source={}
            for key,m in types.items():
                arr=torch.zeros(4*160*120,device='cuda');keep=inside&m;arr.index_add_(0,rank[keep],mass[keep]);source[key]=arr.reshape(4,160,120).cpu().numpy()
            fields={k:e[k][0].cpu().numpy()for k in ['foreground_mass','total_mass','low_pixel_mass','max_ray_mass','ray_bin_count','ray_count','camera_count','entropy']};high=z[f'{seed}_raw']>=.2;xx=-20+(np.arange(160)+.5)*.5;yy=-30+(np.arange(120)+.5)*.5;distance=np.hypot(xx[:,None],yy[None,:]);areas={'all':np.ones_like(known)};areas.update({f'layer{k}':np.broadcast_to((np.arange(4)==k)[:,None,None],known.shape)for k in range(4)});areas.update({f'range{lo}_{hi}':np.broadcast_to(((distance>=lo)&(distance<hi))[None],known.shape)for lo,hi in [(0,10),(10,20),(20,35),(35,80)]})
            for domain,dm in [('target',truth&known),('background',known&~truth),('unknown',~known)]:
                for area,am in areas.items():
                    mask=dm&am&high;n=int(mask.sum());tot=float(fields['foreground_mass'][mask].sum());records.append(dict(group=row['group'],sample_id=row['sample_id'],seed=seed,domain=domain,area=area,high_cells=n,mass=tot,source_mass={k:float(v[mask].sum())for k,v in source.items()},low_q_mass=float(fields['low_pixel_mass'][mask].sum()),D_all_individual_rays_subthreshold=int((mask&(fields['max_ray_mass']<-np.log(.8))).sum()),multi_camera_cells=int((mask&(fields['camera_count']>1)).sum()),ray_bin_count=int(fields['ray_bin_count'][mask].sum()),distinct_ray_cells=int(fields['ray_count'][mask].sum()),mean_entropy=float(fields['entropy'][mask].mean())if n else None))
        print(row['sample_id'],flush=True)
    write(f'reports/m5j_attribution_population_{a.split}.json',dict(kind='actual_GPU_all_highcells_additive_attribution',records=records,source_sha256=file_hash(__file__),limits='A is sourceGTnonforeground even lowq; B means targetray probability outside exactGTvoxel, notnecessarilyindependentphysicalfalsepositive. Eunknown cannotadjudicate. D accumulates subthresholdrays butnotnecessarilyalllowq. C broaddepthspread separatelytopraytrace; categories aredescriptive notexclusivecauses.'))
if __name__=='__main__':main()
