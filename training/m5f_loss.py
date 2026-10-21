"""Reliable visible pixel BCE; no 3D negative inference or pair input."""
import torch
from torch.nn import functional as F


def pixel_loss(logits,target,valid):
    pos=valid&target;neg=valid&~target
    terms=[]
    if pos.any():terms.append(F.softplus(-logits.float()[pos]).mean())
    if neg.any():terms.append(F.softplus(logits.float()[neg]).mean())
    return torch.stack(terms).mean() if terms else logits.float().sum()*0
