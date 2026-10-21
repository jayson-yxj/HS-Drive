"""Streaming pixel PR counts avoid treating background accuracy as success."""
import numpy as np

BINS=256


def histogram(score,target,valid):
    ids=np.minimum((score[valid]*BINS).astype(int),BINS-1);t=target[valid]
    return np.stack([np.bincount(ids[t],minlength=BINS),np.bincount(ids[~t],minlength=BINS)])


def metrics(hist,threshold):
    pos,neg=hist;cut=int(threshold*BINS);tp=int(pos[cut:].sum());fp=int(neg[cut:].sum());p=int(pos.sum());n=int(neg.sum());fn=p-tp
    tpc=np.cumsum(pos[::-1]);fpc=np.cumsum(neg[::-1]);precision=tpc/np.maximum(tpc+fpc,1);ap=float((precision*pos[::-1]).sum()/p) if p else None
    return dict(positive=p,negative=n,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/p if p else None,iou=tp/(tp+fp+fn) if tp+fp+fn else None,fpr=fp/n if n else None,histogram_ap=ap)


def select(groups,empty,thresholds):
    rows=[]
    for th in thresholds:
        scenes={g:dict(**metrics(h,th),empty_fpr=metrics(empty[g],th)['fpr']) for g,h in groups.items()}
        violation=max(max((.8-v['precision'])/.8,(.8-(v['recall'] or 0))/.8,((v['empty_fpr'] or 0)-.01)/.01,0) for v in scenes.values())
        rows.append(dict(threshold=th,scenes=scenes,violation=violation,macro_iou=float(np.mean([v['iou'] or 0 for v in scenes.values()])),feasible=violation==0))
    return min(rows,key=lambda r:(r['violation'],-r['macro_iou'],r['threshold'])),rows
