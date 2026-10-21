"""Masked optical-Z and Actor spatial probability; GT only in loss path."""
import torch
from torch.nn import functional as F

def losses(logits, original, depth, valid, actor, spatial, edges):
    bins=(torch.bucketize(depth.contiguous(),edges,right=True)-1).clamp(0,47)
    ce=F.cross_entropy(logits.float(),bins,reduction='none');prob=logits.float().softmax(1)
    masks=[valid&actor&(depth<10),valid&actor&(depth>=10),valid&~actor&(depth<10),valid&~actor&(depth>=10)]
    terms=[ce[m].mean()for m in masks if m.any()];balanced=torch.stack(terms).mean()if terms else logits.sum()*0
    candidates=spatial.any(1);positive=valid&actor&candidates
    space=-(prob*spatial).sum(1).clamp_min(1e-8).log()[positive].mean()if positive.any()else logits.sum()*0
    near=valid&(depth<10);old=original.float().softmax(1);kl=(old*(old.clamp_min(1e-8).log()-logits.float().log_softmax(1))).sum(1)
    kl=kl[near].mean()if near.any()else logits.sum()*0
    return balanced,space,kl
