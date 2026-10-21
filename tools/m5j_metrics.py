"""Conditional surface and spatial scores; unknown never enters negative counts."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from diagnostics.m5f_metrics import histogram,metrics


def distance_to(mask):
    return np.stack([distance_transform_edt(~m)*.5 if m.any()else np.full(m.shape,np.inf)for m in mask])


def evaluate(score,target,valid,actors,threshold):
    predicted=(score>=threshold)&valid;truth=target&valid;dp=distance_to(predicted);dg=distance_to(truth);tp=int((predicted&truth).sum());fp=int((predicted&~truth).sum());ng=int(truth.sum());npred=int(predicted.sum());nneg=int((valid&~truth).sum());ah=[]
    for a in actors:
        n=int(a.sum());hits=int((a&(dp<=1)).sum());ah.append(dict(n=n,hits=hits,detected=n>=3 and hits>=3 and hits/n>=.2))
    eligible=[a for a in ah if a['n']>=3]
    return dict(tp=tp,fp=fp,positive=ng,negative=nneg,predicted=npred,precision=tp/max(tp+fp,1),recall=tp/ng if ng else None,fpr=fp/nneg if nneg else None,matched05=int((truth&(dp<=.5)).sum()),matched1=int((truth&(dp<=1)).sum()),matchedpred1=int((predicted&(dg<=1)).sum()),actor_cells=sum(a['n']for a in ah),actor_hits=sum(a['hits']for a in ah),actor_views=len(eligible),actor_detected=sum(a['detected']for a in eligible),actors=ah,unknown_high=int(((score>=threshold)&~valid).sum()),unknown_n=int((~valid).sum()))


def aggregate(rr):
    keys=['tp','fp','positive','negative','predicted','matched05','matched1','matchedpred1','actor_cells','actor_hits','actor_views','actor_detected','unknown_high','unknown_n'];s={k:sum(r[k]for r in rr)for k in keys}
    for k,num,den in [('precision',s['tp'],s['tp']+s['fp']),('recall',s['tp'],s['positive']),('fpr',s['fp'],s['negative']),('recall05',s['matched05'],s['positive']),('recall1',s['matched1'],s['positive']),('spatial_precision1',s['matchedpred1'],s['predicted']),('actor_recall',s['actor_hits'],s['actor_cells']),('actor_detection',s['actor_detected'],s['actor_views']),('unknown_high_fraction',s['unknown_high'],s['unknown_n'])]:s[k]=num/den if den else None
    return s
