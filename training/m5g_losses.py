"""Independent visible surface pixel and depth supervision; no inference GT."""
import torch
from torch.nn import functional as F


def foreground_loss(logits,target,valid,mode='bce',hard=None):
    p=valid&target;n=valid&~target;terms=[];weights=[]
    if p.any():terms.append(F.softplus(-logits.float()[p]).mean());weights.append(1.)
    if n.any():
        loss=F.softplus(logits.float());neg=loss[n].mean()
        if mode=='hard':
            values=loss[n];cut=torch.quantile(values.detach(),.9);mask=n&(loss.detach()>=cut)
            if hard is not None:mask |= n&hard
            neg=.5*neg+.5*loss[mask].mean()
        terms.append(neg);weights.append(2. if mode=='weighted'else 1.)
    return sum(t*w for t,w in zip(terms,weights))/sum(weights) if terms else logits.sum()*0


def stratified_depth_loss(logits,depth,valid,target,edges,spatial=None):
    """Equal supported optical-Z bands, equal actor/background within each band."""
    ids=torch.bucketize(depth.contiguous(),edges,right=True)-1
    ce=F.cross_entropy(logits.float(),ids.clamp(0,47),reduction='none');probs=logits.float().softmax(1)
    neighborhood=None if spatial is None else (probs*spatial).sum(1).clamp_min(1e-8)
    bands=[]
    for lo,hi in [(1.5,10),(10,20),(20,35),(35,60),(60,80)]:
        m=valid&(depth>=lo)&(depth<hi);parts=[]
        for is_actor in [False,True]:
            mask=m&(target if is_actor else ~target)
            if mask.any():
                term=ce[mask].mean()
                if neighborhood is not None:
                    support=mask&spatial.any(1)
                    if support.any():term=term+.2*(-neighborhood[support].log()).mean()
                parts.append(term)
        if parts:bands.append(torch.stack(parts).mean())
    return torch.stack(bands).mean() if bands else logits.sum()*0
