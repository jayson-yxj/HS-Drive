"""Separate supervision-only teacher agreement and near-field preservation losses."""
import torch
from torch.nn import functional as F

def teacher_mask(target,valid,classes,trusted):
    """Only positive GT-supported semantic agreement, never teacher negatives.

    V AND Y AND ((vehicle AND q_vehicle>=.7) OR (person AND q_person>=.7)).
    q already excludes instances with detector confidence<.7. Cone/Unknown excluded.
    """
    score=torch.where(classes==1,trusted[0],trusted[1])
    mask=valid&target&(((classes==1)|(classes==2)))&(score>=.7)
    return score,mask

def distillation_loss(logits,score,mask):
    return F.binary_cross_entropy_with_logits(logits.float()[mask],score.float()[mask]) if mask.any() else logits.sum()*0

def depth_ce(logits,depth,mask,edges):
    ids=(torch.bucketize(depth.contiguous(),edges,right=True)-1).clamp(0,47)
    ce=F.cross_entropy(logits.float(),ids,reduction='none')
    return ce[mask].mean() if mask.any() else logits.sum()*0

def near_kl(logits,old_logits,mask):
    log=F.log_softmax(logits.float(),1);oldlog=F.log_softmax(old_logits.detach().float(),1)
    kl=(oldlog.exp()*(oldlog-log)).sum(1)
    return kl[mask].mean() if mask.any() else logits.sum()*0
