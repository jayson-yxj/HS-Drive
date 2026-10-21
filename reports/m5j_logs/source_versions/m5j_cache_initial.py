"""Frozen RGB dualbranch, explicit surface-reference cache and source attribution."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from models.m5j_candidate import MobileBEVCandidate
from models.m5j_evidence import evidence
from tools.m5j_common import rows,write,quantile
from tools.m5f_common import inputs,pixel_label
from tools.m5f_factorization import reference
from models.view_transform import unproject,feature_uv
from simulator.recorder import file_hash
from training.runtime import prepare_cuda


def audit(spatial,d,net,x,seed,row):
    tr=spatial['trace'];rank=tr['ranks'][0].cpu().numpy();inside=tr['inside'][0].cpu().numpy();pr=tr['probability'][0].cpu().numpy();q=tr['foreground'][0].cpu().numpy();v=d['valid'][:,4::8,4::8];gt=d['target'][:,4::8,4::8];dep=d['depth'][:,4::8,4::8]
    uv=feature_uv(48,80,(384,640),'cuda').round()
    pts=unproject(uv.expand(1,6,48,80,2),torch.from_numpy(dep[None]).cuda(),x['intrinsics'],x['cam_to_ego']);gr,gi=net.depth.splat.ranks(pts);gr=gr[0].cpu().numpy();gi=gi[0].cpu().numpy()
    truth,valid,_=reference(d,x,row,row['camera_names']);mass=spatial['foreground_mass'][0].cpu().numpy();total=spatial['total_mass'][0].cpu().numpy();raw=spatial['scores']['raw'][0].cpu().numpy();domains={'target':truth&valid,'background':valid&~truth,'unknown':~valid};summary=[];examples=[]
    for domain,mask in domains.items():
        summary.append(dict(seed=seed,sample_id=row['sample_id'],group=row['group'],domain=domain,fields={k:quantile(spatial[k][0].cpu().numpy()[mask])for k in ['foreground_mass','total_mass','ray_bin_count','ray_count','camera_count','entropy','max_ray_mass','low_pixel_mass']},high_count=int((mask&(raw>=.2)).sum())))
        ids=np.flatnonzero(mask&(raw>=.2))
        if not len(ids):continue
        # One deterministic highest-score example per domain/frame/seed; representative not frequency estimate.
        cell=int(ids[np.argmax(raw.reshape(-1)[ids])]);at=inside&(rank==cell);where=np.argwhere(at);weights=(pr*q[:,None])[at];summ={};raymap={}
        for (c,b,y,z),weight in zip(where,weights):
            key=(int(c),int(y),int(z));rec=raymap.setdefault(key,dict(mass=0.,bins=[]));rec['mass']+=float(weight);rec['bins'].append(int(b))
            tag='A_2D_wrong'if v[c,y,z]and not gt[c,y,z]else'B_target_depth_misprojected'if v[c,y,z]and gt[c,y,z]and(not gi[c,y,z]or int(gr[c,y,z])!=cell)else'correct_target_endpoint'if v[c,y,z]and gt[c,y,z]else'E_unknown_source'
            summ[tag]=summ.get(tag,0.)+float(weight)
        top=[]
        for (c,y,z),rec in sorted(raymap.items(),key=lambda x:x[1]['mass'],reverse=True)[:8]:
            p=pr[c,:,y,z];outside=~inside[c,:,y,z];top.append(dict(camera=row['camera_names'][c],pixel_uv=[4+8*z,4+8*y],foreground=float(q[c,y,z]),gt_valid=bool(v[c,y,z]),gt_target=bool(gt[c,y,z]),gt_depth=float(dep[c,y,z]),gt_rank=int(gr[c,y,z])if gi[c,y,z]else None,probabilities=p.tolist(),contributing_bins=rec['bins'],depth_cell_count=int(len(np.unique(rank[c,inside[c,:,y,z],y,z]))),mass=rec['mass'],outside_probability=float(p[outside].sum())))
        z,xx,yy=np.unravel_index(cell,(4,160,120));m=float(mass.reshape(-1)[cell]);low=float(spatial['low_pixel_mass'].reshape(-1)[cell]);examples.append(dict(seed=seed,sample_id=row['sample_id'],group=row['group'],domain=domain,cell=[int(z),int(xx),int(yy)],range_xyz=[[-20+xx*.5,-20+(xx+1)*.5],[-30+yy*.5,-30+(yy+1)*.5],[-2+z*1.5,-2+(z+1)*1.5]],mass=m,total=float(total.reshape(-1)[cell]),ray_bins=int(at.sum()),distinct_rays=len(raymap),camera_mass=spatial['camera_mass'][0,:,z,xx,yy].cpu().tolist(),source_mass=summ,low_q_mass=low,D_low_accumulation=low>=-np.log(.8)and max((r['mass']for k,r in raymap.items()if q[k]<.2),default=0)<-np.log(.8),C_diffuse_ray_count=sum(len(np.unique(rank[c,inside[c,:,y,x],y,x]))>1 for c,y,x in raymap),top_rays=top))
    return summary,examples


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','historical','test'],default='val');a=p.parse_args()
    if a.split=='test':
        from tools.m5j_freeze import verify
        verify()
    dest=Path('runs/m5j_'+a.split);dest.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);rr=rows(a.split);records=[];stats=[];examples=[];net=MobileBEVCandidate(enable=True,seed=101);base=net.depth
    models=json.loads(Path('configs/m5j_models.json').read_text());from tools.m5i_data import load
    pixels={s:load(next(r for r in models['pixel']if r['name']=='mobile_gt'and r['seed']==s))[0]for s in [101,102]};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for row in rr:
        x=inputs(row);d=pixel_label(row);truth,valid,actors=reference(d,x,row,row['camera_names'])
        from models.m5d_stride8 import cache_p8
        c=cache_p8(base,x);v=torch.from_numpy(d['valid'][:,4::8,4::8][None]).cuda();gt=torch.from_numpy(d['target'][:,4::8,4::8][None]).float().cuda();dep=torch.from_numpy(d['depth'][:,4::8,4::8][None]).cuda();payload=dict(target=truth,valid=valid,actor_masks=np.array([ac['mask']for ac in actors],dtype=bool).reshape(-1,4,160,120));actor_meta=[{k:v for k,v in ac.items()if k!='mask'}for ac in actors]
        for seed,pixel in pixels.items():
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):q=torch.clamp((pixel(**x).float().sigmoid()*256).floor(),max=255)/256
            sample=q[:,:,4::8,4::8];e=evidence(base.splat,sample,c['probs'],c['K'],c['T'],c['image_hw'],trace=True)
            for k in ['foreground_mass','total_mass','ray_bin_count','ray_count','camera_count','camera_mass','camera_total','entropy','max_ray_mass','low_pixel_mass']:payload[f'{seed}_{k}']=e[k][0].cpu().numpy()
            for k,s in e['scores'].items():payload[f'{seed}_{k}']=s[0].cpu().numpy()
            payload[f'{seed}_pixel']=q[0].cpu().numpy().astype(np.float16)
            ss,ee=audit(e,d,net,x,seed,row);stats+=ss;examples+=ee
            # Matched reliable P8 diagnostic masks; formal unmasked main output remains separate.
            for mode,fg,depth in [('mobile_learned',sample,None),('oracle_learned',gt,None),('mobile_oracle',sample,dep),('oracle_oracle',gt,dep)]:
                z=evidence(base.splat,fg,c['probs'],c['K'],c['T'],c['image_hw'],depth=depth,valid=v)
                for k,s in z['scores'].items():payload[f'{seed}_diag_{mode}_{k}']=s[0].cpu().numpy()
            payload[f'{seed}_gate_target_pixels']=np.array([int((v& (gt>0)).sum()),int((v&(gt>0)&(sample>=.9)).sum())])
        path=dest/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,**payload);records.append(dict(**row,cache=str(path),cache_sha256=file_hash(path),actor_meta=actor_meta));print(row['sample_id'],flush=True)
    write(dest/'index.json',dict(kind='actual_RTX4090_fixed_dual_RGB_offline_reference_separate',records=records,elapsed_s=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,source_sha256=file_hash(__file__),models_sha256=file_hash('configs/m5j_models.json')))
    write(f'reports/m5j_evidence_attribution_{a.split}.json',dict(kind='offline_actual_GPU_source_contribution_audit',statistics=stats,examples=examples,source_sha256=file_hash(__file__),limits='A/B endpoint-based additive source classes; C/D overlap; E cannot certify physicalFP. Distinct rays identified camera/pixel, spatial correlation remains. Highest-cell examples not categoryprevalence. Rawgeometrysupport includes tinyprobabilities.'))
if __name__=='__main__':main()
