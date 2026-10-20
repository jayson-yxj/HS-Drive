"""Descriptive frozen-test actor strata; no threshold/model reselection."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash
from diagnostics.m5c_localization import matching
from tools.m5e_common import sidecars


def main():
    dest=Path('reports/m5e_actor_localization.json')
    if dest.exists():raise FileExistsError(dest)
    source=json.loads(Path('reports/m5e_presence_absence.json').read_text());root='/root/gpufree-data/hs-drive-data/processed/m5e_test_index';rows=json.loads(Path(root,'index.json').read_text())['samples'];refs=sidecars(root,rows,'m5e_surface_index.json');records=[]
    x=-20+(np.arange(160)+.5)*.5;y=-30+(np.arange(120)+.5)*.5;rad=np.hypot(x[:,None],y[None,:]);corridor=(x[:,None]>0)&(x[:,None]<35)&(abs(y[None,:])<=1.5)
    for r in source['records']:
        scores=np.load(r['predictions'])['score'];frames=[]
        for row,d,s,f in zip(rows,refs,scores,r['frames']):
            p=s>r['threshold'];actor=d['actor_masks'].any(0)&d['valid'];strata={}
            masks={f'distance_{lo}_{hi}':actor&((rad>=lo)&(rad<hi))[None] for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]}
            masks.update({f'height_{z}':actor&(np.arange(4)==z)[:,None,None] for z in range(4)})
            masks.update(actor=actor,actor_corridor=actor&corridor[None],ground=d['ground']&d['valid']&~d['target'],unknown=~d['valid'],mixed=actor&d['mixed'],unmixed=actor&~d['mixed'])
            for name,m in masks.items():strata[name]=dict(n=int(m.sum()),high=int((p&m).sum()),mean_probability=float(s[m].mean()) if m.any() else None)
            frames.append(dict(group=row['group'],configuration=row['configuration'],strata=strata,actor_matching=matching(p&d['valid'],actor),actors=f['metrics']['actors'],full_known_surface_spatial=f['metrics']['spatial']))
        agg={}
        for key in frames[0]['strata']:
            n=sum(f['strata'][key]['n'] for f in frames);hit=sum(f['strata'][key]['high'] for f in frames);agg[key]=dict(n=n,high=hit,fraction=hit/n if n else None)
        scene={}
        for g in source['groups']:
            ff=[f for f in frames if f['group']==g];n=sum(f['strata']['actor']['n'] for f in ff);hit=sum(f['strata']['actor']['high'] for f in ff);scene[g]=dict(actor_n=n,actor_exact_recall=hit/n if n else None,gold_witness=r['presence_absence']['by_scene'][g])
        records.append(dict(name=r['name'],seed=r['seed'],threshold=r['threshold'],aggregate=agg,by_scene=scene,frames=frames))
    atomic_json(dest,dict(kind='offline_diagnostic_frozen_GPU_test_predictions',records=records,source_sha256=file_hash(__file__),prediction_report_sha256=file_hash('reports/m5e_presence_absence.json'),limits='Actorfirsthit heldout rendered surfaces, AABB association not meshtruth. Exactrecall on reliablevisiblecells only. Actor1m matching weak manytoone, not objectassociation. Unknown high rate notFPR. Allmodels/thresholds unchanged after test. No independent voxel statisticalsamples.'))

if __name__=='__main__':main()
