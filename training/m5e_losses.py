"""Reliable positive-only targets and FP32 spatial distribution supervision."""
import torch
from torch.nn import functional as F


def presence_only(logits,mask):
    """No absent/Unknown negative manufactured; base ground/gold losses required."""
    x=-20+(torch.arange(160,device=logits.device)+.5)*.5
    y=-30+(torch.arange(120,device=logits.device)+.5)*.5
    rad=torch.sqrt(x[:,None]**2+y[None,:]**2);terms=[]
    for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]:
        m=mask&((rad>=lo)&(rad<hi))[None,None]
        if m.any():terms.append(F.softplus(-logits.float()[m]).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def spatial_loss(logits,xy_error,same_layer,valid,radial_distance,*,sigma_m=1.):
    """Soft spatial target over bins; no requirement to hit exact coarse voxel.

    Soft normalized exp(-XY^2/(2 sigma^2)) on correct layer. Unsupported rays
    (no correct-layer bin within2m) masked, never assigned wrong-layer target.
    Equal nonempty radial strata; each valid ray is a reliable rendered surface.
    """
    with torch.autocast(device_type=logits.device.type,enabled=False):
        cost=xy_error.float().square()/(2*sigma_m**2)
        support=same_layer&(xy_error<=2)
        valid=valid&support.any(2)
        weights=torch.exp(-cost)*support
        target=weights/weights.sum(2,keepdim=True).clamp_min(1e-12)
        ce=-(target*logits.float().log_softmax(2)).sum(2)
        terms=[]
        for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]:
            m=valid&(radial_distance>=lo)&(radial_distance<hi)
            if m.any():terms.append(ce[m].mean())
        return torch.stack(terms).mean() if terms else logits.float().sum()*0
