"""Visible foreground event mass, deliberately not calibrated physical occupancy."""
import torch
from models.view_transform import feature_uv,unproject


def lift(splat,foreground,probabilities,K,T,image_hw,*,depth=None,valid=None,uv=None):
    """No GT necessary for learned mode. Optional masks only diagnostic interventions.

    [B,N,H,W] foreground; [B,N,D,H,W] probability; FP32geometry.
    No totalmass division: tiny probability cannot become confident via normalization.
    """
    b,n,h,w=foreground.shape;cells=splat.nz*splat.nx*splat.ny
    with torch.autocast(device_type=foreground.device.type,enabled=False):
        if uv is None:uv=feature_uv(h,w,image_hw,foreground.device).round()
        if depth is None:
            d=splat.depth_centers[None,None,:,None,None].expand(b,n,-1,h,w);p=probabilities.float()
        else:d=depth[:,:,None].float();p=torch.ones_like(d)
        pts=unproject(uv.expand(b,n,d.shape[2],h,w,2),d,K.float(),T.float());rank,inside=splat.ranks(pts)
        if valid is not None:inside &= valid[:,:,None]
        rank=rank+torch.arange(b,device=foreground.device).view(b,1,1,1,1)*cells
        mass=torch.zeros(b*cells,device=foreground.device);total=mass.clone();count=mass.clone()
        mass.index_add_(0,rank[inside],(p*foreground[:,:,None].float())[inside]);total.index_add_(0,rank[inside],p[inside]);count.index_add_(0,rank[inside],torch.ones_like(p)[inside])
        mass=mass.reshape(b,splat.nz,splat.nx,splat.ny);total=total.reshape_as(mass);count=count.reshape_as(mass)
    return dict(foreground_mass=mass,total_depth_mass=total,ray_bin_support_count=count,event_score=-torch.expm1(-mass))
