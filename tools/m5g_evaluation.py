"""Common 2D current-surface metrics, offline paired masks and no GT inference."""
import numpy as np
from diagnostics.m5f_metrics import histogram,metrics,select


def summarize(rows,labels,probabilities,threshold=None,classes=(1,2,3)):
    groups={};empty={};strata={};actors=[];pred=[];pairs=[]
    for row,d,prob in zip(rows,labels,probabilities):
        target=d['target']&np.isin(d['classes'],classes);valid=d['valid']&(~d['target']|target);g=row['group'];h=histogram(prob,target,valid);groups[g]=groups.get(g,np.zeros((2,256),np.int64))+h;empty.setdefault(g,np.zeros((2,256),np.int64))
        configs=[r['configuration']for r in rows if r['group']==g]
        if row['configuration']in[min(configs),max(configs)]:empty[g]+=histogram(prob,target,valid&~target)
        for c,name in enumerate(row['camera_names']):
            for key,mask in [('all',valid[c])]+[(f'distance_{lo}_{hi}',valid[c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi))for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]]:
                hh=histogram(prob[c],target[c],mask)
                for category in [f'camera:{name}/{key}',f'weather:{row["weather"]}/{key}',f'pooled/{key}']:strata[category]=strata.get(category,np.zeros((2,256),np.int64))+hh
    choosing=threshold is None
    selection=None
    if choosing:
        selection,_=select(groups,empty,[x/10 for x in range(1,10)])
        threshold=selection['threshold']
    cut=int(threshold*256)/256
    for row,d,prob in zip(rows,labels,probabilities):
        for c,name in enumerate(row['camera_names']):
            for a in row['actors'][c]['actors']:
                mask=(d['actor'][c]==a['local_id'])&np.isin(d['classes'][c],classes);n=int(mask.sum());hits=int(((prob[c]>=cut)&mask).sum());actors.append(dict(group=row['group'],configuration=row['configuration'],camera=name,actor_id=a['actor_id'],type_id=a['type_id'],distance_m=a['distance_m'],pixels=n,hits=hits,detected=n>=3 and hits>=3 and hits/n>=.5,mean_probability=float(prob[c][mask].mean())if n else None))
    for g in sorted(groups):
        ids=sorted([i for i,r in enumerate(rows)if r['group']==g],key=lambda i:rows[i]['configuration']);a,z=ids[0],ids[-1]
        for i in ids[1:-1]:
            d=labels[i];da=labels[a];dz=labels[z];m=d['target']&d['valid']&np.isin(d['classes'],classes)&(d['actor']>0)&da['valid']&~da['target'];mz=m&dz['valid']&~dz['target'];n=int(m.sum());nr=int(mz.sum());p=probabilities[i];pa=probabilities[a];pz=probabilities[z]
            pairs.append(dict(group=g,configuration=rows[i]['configuration'],n=n,removed_n=nr,present_hits=int(((p>=cut)&m).sum()),empty_fp=int(((pa>=cut)&m).sum()),both_correct=int(((p>=cut)&(pa<cut)&m).sum()),removed_fp=int(((pz>=cut)&mz).sum()),mean_rise=float((p-pa)[m].mean())if n else None,mean_fall=float((p-pz)[mz].mean())if nr else None,mean_removed=float(pz[mz].mean())if nr else None,direction_rise=float((p[m]>pa[m]).mean())if n else None,direction_fall=float((p[mz]>pz[mz]).mean())if nr else None))
    pooled=sum(groups.values());allcurve=[dict(threshold=x/100,**metrics(pooled,x/100))for x in range(1,100)];allowed=[r for r in allcurve if(r['fpr']or 0)<=.01]
    return dict(threshold=threshold,selection=selection,metrics=metrics(pooled,threshold),by_scene={g:metrics(h,threshold)for g,h in groups.items()},empty_by_scene={g:metrics(h,threshold)for g,h in empty.items()},strata={k:metrics(h,threshold)for k,h in strata.items()},actors=actors,pairs=pairs,histogram=pooled.tolist(),low_fpr_validation_workpoint=max(allowed,key=lambda x:x['recall']or 0)if allowed and choosing else None,limits='Histogrampositivefirst; conditionalcurrentfirstsurface. Unsupportedclassesignore. Suppliedlockedthreshold neverreselected; lowFPRworkpointonlywhenchoosingonvalidation. Actorcoveragealone notdetectionprecision.')
