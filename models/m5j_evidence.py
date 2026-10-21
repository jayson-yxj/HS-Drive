"""Inference-only spatial evidence and exact distinct feature-ray accounting.

Scores are candidate surface-event evidence, never occupancy or traversable free.
Distinct means unique camera/pixel identity, not statistically independent samples.
"""
import math
import torch
from models.view_transform import feature_uv, unproject


def reduce_evidence(ranks, inside, probabilities, foreground, shape, *, trace=False):
    """Monotone depth samples on straight rays visit each convex voxel contiguously.

    Therefore first valid rank or rank change along D counts each ray/cell once.
    No probability threshold is used to fabricate support. Camera counts count
    geometric support, even if its probability is tiny; mass is provided separately.
    """
    b,n,d,h,w = probabilities.shape
    cells=math.prod(shape); device=probabilities.device
    pr=probabilities.float(); q=foreground.float()
    ids=ranks+torch.arange(b,device=device).view(b,1,1,1,1)*cells
    keep=inside & (pr>0)
    first=keep.clone()
    first[:,:,1:] &= (~keep[:,:,:-1]) | (ranks[:,:,1:]!=ranks[:,:,:-1])
    # Softmax can underflow to zero; contiguous equal-rank zero-probability bins
    # must not create a second independent ray. Count geometry, not positive mass.
    first=inside.clone()
    first[:,:,1:] &= (~inside[:,:,:-1]) | (ranks[:,:,1:]!=ranks[:,:,:-1])
    def scatter(value,mask=inside):
        z=torch.zeros(b*cells,device=device,dtype=torch.float32)
        z.index_add_(0,ids[mask],value.expand_as(pr)[mask])
        return z.reshape(b,*shape)
    fg=pr*q[:,:,None]; mass=scatter(fg); total=scatter(pr)
    entropy=-(pr*pr.clamp_min(1e-30).log()).sum(2)/math.log(d) if d>1 else torch.zeros_like(q)
    entmass=scatter(pr*entropy[:,:,None]); ray_count=scatter(torch.ones_like(pr),first)
    cams=[]; cam_total=[]
    for c in range(n):
        mask=inside.clone(); mask[:,:c]=False; mask[:,c+1:]=False
        cams.append(scatter(fg,mask));cam_total.append(scatter(pr,mask))
    camera_mass=torch.stack(cams,1);camera_total=torch.stack(cam_total,1)
    cam_count=torch.zeros_like(mass)
    for c in range(n):
        mask=first.clone();mask[:,:c]=False;mask[:,c+1:]=False
        cam_count+=(scatter(torch.ones_like(pr),mask)>0)
    # Group adjacent same-rank bins within each ray for maximum per-ray mass.
    rayids=torch.arange(b*n*h*w,device=device).reshape(b,n,1,h,w).expand_as(pr)
    flatkeys=ids[inside]*(b*n*h*w)+rayids[inside]
    keys,inv=torch.unique(flatkeys,return_inverse=True)
    rm=torch.zeros(len(keys),device=device);rm.index_add_(0,inv,fg[inside])
    maxray=torch.zeros(b*cells,device=device);maxray.scatter_reduce_(0,keys//(b*n*h*w),rm,reduce='amax',include_self=True)
    raw=-torch.expm1(-mass)
    result=dict(foreground_mass=mass,total_mass=total,ray_bin_count=scatter(torch.ones_like(pr)),
                ray_count=ray_count,camera_count=cam_count,camera_mass=camera_mass,camera_total=camera_total,
                entropy=entmass/total.clamp_min(1e-8),max_ray_mass=maxray.reshape(b,*shape),
                low_pixel_mass=scatter(fg*(q[:,:,None]<.2)),
                scores=dict(raw=raw,gate=-torch.expm1(-scatter(fg*(q[:,:,None]>=.9))),
                            regularized=raw*mass/(total+1.)))
    if trace:result['trace']=dict(ranks=ranks,inside=inside,rayids=rayids,probability=pr,foreground=q)
    return result


def evidence(splat, foreground, probabilities, K, T, image_hw, *, depth=None, valid=None, uv=None, trace=False):
    """GT depth/valid are restricted to offline diagnostic callers, not Candidate."""
    b,n,h,w=foreground.shape
    with torch.autocast(device_type=foreground.device.type,enabled=False):
        if uv is None:uv=feature_uv(h,w,image_hw,foreground.device).round()
        if depth is None:
            depths=splat.depth_centers[None,None,:,None,None].expand(b,n,-1,h,w);pr=probabilities.float()
        else:
            depths=depth[:,:,None].float();pr=torch.ones_like(depths)
        points=unproject(uv.expand(b,n,depths.shape[2],h,w,2),depths,K.float(),T.float())
        rank,inside=splat.ranks(points)
        if valid is not None:inside &= valid[:,:,None]
        return reduce_evidence(rank,inside,pr,foreground,(splat.nz,splat.nx,splat.ny),trace=trace)
