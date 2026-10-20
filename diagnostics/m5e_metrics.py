"""Scene Pareto selection, preserving missing support and actor/absence tradeoffs."""
import numpy as np
from diagnostics.m5d_metrics import witness_metrics,aggregate_witness


def selection(prob,rows,refs,pairs,lookup,thresholds,near_ratio=1.):
    candidates=[]
    for th in thresholds:
        events=[dict(group=r['group'],witness=witness_metrics(prob[lookup[r['a']]],prob[lookup[r['b']]],p,th)) for r,p in pairs]
        groups=aggregate_witness(events)['by_scene'];scene={}
        for g in sorted({r['group'] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r['group']==g];hit=pos=fp=neg=0
            for i in ids:
                d=refs[i];m=d['valid'];a=d['actor_masks'].any(0)&m
                hit+=int(((prob[i]>th)&a).sum());pos+=int(a.sum());ground=m&~d['target'];fp+=int(((prob[i]>th)&ground).sum());neg+=int(ground.sum())
            w=groups[g];scene[g]=dict(actor_recall=hit/pos if pos else 0,presence=w['presence_recall'] or 0,absence_fpr=w['absence_fpr'] if w['n'] else 1,ground_fpr=fp/neg if neg else 1,actor_cells=pos,witness_n=w['n'])
        violations=[max((.7-v['actor_recall'])/.7,(.8-v['presence'])/.8,(v['absence_fpr']-.1)/.1,(v['ground_fpr']-.05)/.05,near_ratio/1.1-1,0) for v in scene.values()]
        vector=[np.mean([v[k] for v in scene.values()]) for k in ['actor_recall','presence','absence_fpr','ground_fpr']]
        candidates.append(dict(threshold=th,scenes=scene,worst_violation=float(max(violations)),feasible=max(violations)<=0,near_depth_ratio=near_ratio,pareto_vector=vector))
    for c in candidates:
        v=np.array(c['pareto_vector'])*[1,1,-1,-1]
        c['pareto']=not any(np.all(np.array(o['pareto_vector'])*[1,1,-1,-1]>=v)&np.any(np.array(o['pareto_vector'])*[1,1,-1,-1]>v) for o in candidates)
    best=min(candidates,key=lambda c:(c['worst_violation'],-c['pareto_vector'][0],c['threshold']))
    return best,candidates
