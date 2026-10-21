"""GT-gated output transfer and independent normalized teacher-feature matching."""
import torch
from torch.nn import functional as F

def masks(target,valid,classes,score,trusted):
    same=torch.where(classes==1,trusted[0],trusted[1]);positive=valid&target&((classes==1)|(classes==2))&(same>=.7);negative=valid&~target&(score.max(0).values<=.05);soft=torch.where(positive,same,score.max(0).values);return soft,positive,negative

def output_kd(logits,soft,positive,negative):
    terms=[F.binary_cross_entropy_with_logits(logits.float()[m],soft[m].float())for m in [positive,negative]if m.any()];return torch.stack(terms).mean()if terms else logits.sum()*0

def feature_regions(target,valid,classes):
    # Feature anchors correspond to native0::8. Conservative3x3 support sampled
    # at receptive center; no fabricated annotation at ignored renderboundaries.
    v=valid.float()[:,None];interior=-F.max_pool2d(-v,3,1,1)>.5;t=target.float()[:,None];boundary=(F.max_pool2d(t,5,1,2)-(-F.max_pool2d(-t,5,1,2)))>.5;supported=((classes==1)|(classes==2))[:,None];ok=interior&(~t.bool()|supported);pos=ok&t.bool()&~boundary;neg=ok&~t.bool()&~boundary;edge=ok&boundary
    return [m[:,0,::8,::8]for m in [pos,neg,edge]]

def feature_kd(projected,teacher,regions):
    error=1-(F.normalize(projected.float(),dim=1,eps=1e-6)*F.normalize(teacher.detach().float(),dim=1,eps=1e-6)).sum(1);terms=[error[m].mean()for m in regions if m.any()];return torch.stack(terms).mean()if terms else projected.sum()*0
