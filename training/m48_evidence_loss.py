"""Independent masked binary event losses; unknown never becomes a negative."""
import torch
from torch.nn import functional as F


def evidence_targets(labels):
    sem_mask=labels['surface_semantic_valid'][:,None].expand_as(labels['surface_semantic']).clone()
    sem_mask[:,1]=labels['obstacle_valid']
    return dict(surface=(labels['surface_evidence'],labels['surface_valid']),
                free=(labels['free_ray_evidence'],labels['free_evidence_valid']),
                semantic=(labels['surface_semantic'],sem_mask))


def evidence_loss(outputs,labels):
    parts={}
    for key,(target,mask) in evidence_targets(labels).items():
        logits=outputs[key].float()
        # Class balance within each task; empty classes do not create negatives.
        pos=mask & target.bool();neg=mask & ~target.bool()
        terms=[]
        if pos.any():terms.append(F.softplus(-logits[pos]).mean())
        if neg.any():terms.append(F.softplus(logits[neg]).mean())
        parts[key]=sum(terms)/len(terms) if terms else logits.sum()*0
    return sum(parts.values())/len(parts),parts
