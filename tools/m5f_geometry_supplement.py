"""Fixed-threshold paired BEV diagnostic; never selects models or thresholds."""
import json
from pathlib import Path
import numpy as np
from diagnostics.m5f_metrics import histogram,metrics
from diagnostics.m5c_localization import matching
from diagnostics.m5d_metrics import voxel_3d_matching
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_geometry_supplement.json')
    if dest.exists():raise FileExistsError(dest)
    source=json.loads(Path('reports/m5f_2d_3d_factorization.json').read_text())['records'];fixed=[];pairs=[];threshold=.2
    for name,seed,mode in sorted({(r['name'],r['seed'],r['mode']) for r in source}):
        rr=[r for r in source if (r['name'],r['seed'],r['mode'])==(name,seed,mode)];hist=np.zeros((2,256),np.int64);ngt=hits=0;tol={'.5':0,'1':0};layers={};scenes={};distances={};loc=[]
        for r in rr:
            d=np.load(r['path']);p=d['score'];t=d['target'];v=d['valid'];pred=(p>=int(threshold*256)/256)&v;truth=t&v;h=histogram(p,t,v);hist+=h;ngt+=int(truth.sum());hits+=int((truth&pred).sum());m=matching(pred,truth);tol['.5']+=m['tolerances']['0.5']['matched_gt'];tol['1']+=m['tolerances']['1.0']['matched_gt'];loc.append(voxel_3d_matching(pred,truth));scenes[r['group']]=scenes.get(r['group'],np.zeros((2,256),np.int64))+h
            for z in range(4):layers[str(z)]=layers.get(str(z),np.zeros((2,256),np.int64))+histogram(p[z],t[z],v[z])
            rad=np.hypot(-20+(np.arange(160)[:,None]+.5)*.5,-30+(np.arange(120)[None,:]+.5)*.5)
            for lo,hi in [(5,10),(10,15),(15,20),(20,35)]:
                key=f'{lo}_{hi}';distances[key]=distances.get(key,np.zeros((2,256),np.int64))+histogram(p,t,v&((rad>=lo)&(rad<hi))[None])
        def location(key):
            support=sum(a[key]['n']-a[key]['no_candidate'] for a in loc);return dict(support=support,no_candidate=sum(a[key]['no_candidate'] for a in loc),mean_m=sum((a[key]['mean_m'] or 0)*(a[key]['n']-a[key]['no_candidate']) for a in loc)/support if support else None)
        fixed.append(dict(name=name,seed=seed,mode=mode,threshold=threshold,metrics=metrics(hist,threshold),gt_cells=ngt,exact_recall=hits/ngt if ngt else None,recall_05=tol['.5']/ngt if ngt else None,recall_1m=tol['1']/ngt if ngt else None,by_layer={k:metrics(h,threshold) for k,h in layers.items()},by_scene={k:metrics(h,threshold) for k,h in scenes.items()},by_distance={k:metrics(h,threshold) for k,h in distances.items()},gt_to_pred=location('gt_to_pred'),pred_to_gt=location('pred_to_gt')))
        for group in sorted({r['group'] for r in rr}):
            rs=sorted([r for r in rr if r['group']==group],key=lambda r:r['configuration']);a=np.load(rs[0]['path']);z=np.load(rs[-1]['path'])
            for r in rs[1:-1]:
                d=np.load(r['path']);m=d['target']&d['valid'];n=int(m.sum());pr=d['score'][m];pe=a['score'][m];pz=z['score'][m];cut=int(threshold*256)/256
                pairs.append(dict(name=name,seed=seed,mode=mode,group=group,configuration=r['configuration'],n=n,mean_present=float(pr.mean()) if n else None,mean_empty=float(pe.mean()) if n else None,mean_removed=float(pz.mean()) if n else None,direction_rise=float((pr>pe).mean()) if n else None,direction_fall=float((pr>pz).mean()) if n else None,present_rate=float((pr>=cut).mean()) if n else None,empty_event_residue=float((pe>=cut).mean()) if n else None,removed_event_residue=float((pz>=cut).mean()) if n else None,empty_current_surface_supported=int(a['valid'][m].sum()),removed_current_surface_supported=int(z['valid'][m].sum())))
    atomic_json(dest,dict(kind='offline_saved_GPU_projection_analysis',fixed_common_threshold=fixed,pairs=pairs,source_sha256=file_hash(__file__),limits='Fixed .2 reported for all models/modes, no test selection. Pair mask=present visible target cells; empty/removed score there is event residue diagnostic, NOT physical absence FPR: oldsurface may be unknown now. Nativegrid differs S16/P8. Same-renderer visible surfaces, 3D distances coarsevoxelcenters not preciseMesh.'))

if __name__=='__main__':main()
