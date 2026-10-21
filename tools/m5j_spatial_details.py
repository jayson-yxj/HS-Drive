"""Locked workpoints, paired removal, camera coverage and distance/layer stratification."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m5j_common import write
from tools.m5j_metrics import evaluate,aggregate
from diagnostics.m5f_metrics import histogram,metrics
from simulator.recorder import file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='val');a=p.parse_args();idx=json.loads(Path(f'runs/m5j_{a.split}/index.json').read_text())['records'];locked=json.loads(Path('reports/m5j_selection.json').read_text());records=[];pairs=[];diags=[];gate=[]
    for group in sorted({r['group']for r in idx}):
        rr=sorted([r for r in idx if r['group']==group],key=lambda r:r['configuration']);first=np.load(rr[0]['cache']);last=np.load(rr[-1]['cache'])
        for row in rr:
            d=np.load(row['cache']);truth=d['target'];valid=d['valid'];actors=d['actor_masks'];xx=-20+(np.arange(160)+.5)*.5;yy=-30+(np.arange(120)+.5)*.5;distance=np.hypot(xx[:,None],yy[None,:]);areas={f'layer{z}':np.broadcast_to((np.arange(4)==z)[:,None,None],truth.shape)for z in range(4)};areas.update({f'range{lo}_{hi}':np.broadcast_to(((distance>=lo)&(distance<hi))[None],truth.shape)for lo,hi in [(0,10),(10,20),(20,35),(35,80)]})
            for seed in [101,102]:
                for c,name in enumerate(row['camera_names']):areas['camera_coverage_'+name]=d[f'{seed}_camera_total'][c]>0
                gate.append(dict(group=group,seed=seed,sample_id=row['sample_id'],target_rays=int(d[f'{seed}_gate_target_pixels'][0]),retained=int(d[f'{seed}_gate_target_pixels'][1])))
                for method in ['raw','gate','regularized']:
                    th=locked['workpoints'][f'{seed}_{method}']['selected'];score=d[f'{seed}_{method}']
                    for category,area in areas.items():records.append(dict(group=group,sample_id=row['sample_id'],seed=seed,method=method,threshold=th,category=category,**evaluate(score,truth,valid&area,actors&area,th)))
                    if row!=rr[0]and row!=rr[-1]:
                        m=truth&valid&last['valid']&~last['target'];n=int(m.sum());pairs.append(dict(group=group,sample_id=row['sample_id'],seed=seed,method=method,n=n,removed_fp=int(((last[f'{seed}_{method}']>=th)&m).sum()),present_hits=int(((score>=th)&m).sum()),mean_fall=float((score-last[f'{seed}_{method}'])[m].mean())if n else None))
                    for mode in ['mobile_learned','oracle_learned','mobile_oracle','oracle_oracle']:
                        diags.append(dict(group=group,sample_id=row['sample_id'],seed=seed,method=method,mode=mode,threshold=th,**evaluate(d[f'{seed}_diag_{mode}_{method}'],truth,valid,actors,th)))
    write(f'reports/m5j_spatial_details_{a.split}.json',dict(kind='offline_fixed_workpoint_stratified_and_paired_diagnostics',records=records,pairs=pairs,oracle_diagnostics=diags,gate_target_loss=gate,source_sha256=file_hash(__file__),limits='Camera regions geometric overlapping support, not independentlyattributableFP; sources per-camera inattribution. Unknownmasked. OriginalReference/paircurrentfirstsurface, not full3Dabsence.'))
if __name__=='__main__':main()
