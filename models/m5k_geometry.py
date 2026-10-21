"""Offline probability alignment and RGB-only adaptive sampling; no default change."""
import torch
from torch.nn import functional as F

def resample_probability(probability, size):
    """Pixel-center bilinear resampling with border replication; FP32 normalization.

    S16 rounded centers 8+16i map to P8 centers 4+8j. Their normalized
    center offset equals align_corners=False. No depth/logit interpolation.
    """
    b,n,d,h,w=probability.shape
    p=F.interpolate(probability.float().flatten(0,1),size=size,mode='bilinear',align_corners=False).reshape(b,n,d,*size)
    return p/p.sum(2,keepdim=True).clamp_min(1e-20)

def sampling_masks(foreground, threshold=.9):
    """Extra S4 rays chosen only from predicted foreground, never GT.

    Uniform exact-budget control spans camera-major S4 array evenly. P8 phase4
    and S4 phase2 have disjoint pixel centers, so their union has no duplicates.
    """
    roi=foreground>=threshold
    uniform=torch.zeros_like(roi)
    for batch in range(len(roi)):
        budget=int(roi[batch].sum());total=roi[batch].numel()
        if budget:
            ids=torch.floor((torch.arange(budget,device=roi.device)+.5)*total/budget).long()
            uniform[batch].view(-1)[ids]=True
        assert int(uniform[batch].sum())==budget
    return roi,uniform
