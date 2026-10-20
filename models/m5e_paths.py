"""Explicit factorized visual-only diagnostic paths; default model unmodified."""
import torch
from torch.nn import functional as F
from models.m5b_cached_encoder import cache_rgb,cached_forward
from models.m5d_stride8 import cache_p8
from models.view_transform import feature_uv,unproject


def depth_logits(net,cache):
    f=cache['features'];b,n,c,h,w=f.shape;ih,iw=cache['image_hw'];K=cache['K'];T=cache['T']
    camera=torch.stack((K[...,0,0]/iw,K[...,1,1]/ih,K[...,0,2]/iw,K[...,1,2]/ih),-1)
    ext=torch.cat((T[...,:3,:3].flatten(-2),T[...,:3,3]/10),-1)
    cond=net.camera_mlp(torch.cat((camera,ext),-1)).reshape(b*n,c,1,1)
    return net.depth_head(f.flatten(0,1)+cond).reshape(b,n,48,h,w)


def geometry(net,cache,depth):
    b,n,h,w=depth.shape
    with torch.autocast(device_type='cuda',enabled=False):
        uv=feature_uv(h,w,cache['image_hw'],'cuda').round()
        gt=unproject(uv.expand(b,n,h,w,2),depth,cache['K'],cache['T'])
        centers=net.splat.depth_centers
        points=unproject(uv.expand(b,n,48,h,w,2),centers[None,None,:,None,None].expand(b,n,48,h,w),cache['K'],cache['T'])
        xy=(points[...,:2]-gt[:,:,None,...,:2]).norm(dim=-1)
        z=torch.bucketize(points[...,2].contiguous(),net.splat.z_edges,right=True)
        gz=torch.bucketize(gt[...,2].contiguous(),net.splat.z_edges,right=True)
        ids,inside=net.splat.ranks(gt);ranks,valid=net.splat.ranks(points)
    return dict(xy=xy,same_layer=(z==gz[:,:,None])&valid,rad=gt[...,:2].norm(dim=-1),gt=gt,points=points,inside=inside,bin_inside=valid,ids=ids,ranks=ranks)


def factorize(c16,c8,mode):
    if mode=='s16':return c16
    if mode=='p8':return c8
    if mode=='p8_s16depth':
        b,n,d,h,w=c16['probs'].shape
        p=F.interpolate(c16['probs'].flatten(0,1),size=c8['features'].shape[-2:],mode='bilinear',align_corners=False).reshape(b,n,d,*c8['features'].shape[-2:]);p=p/p.sum(2,keepdim=True)
        return dict(c8,probs=p)
    if mode=='p8_down16':
        b,n,c,h,w=c8['features'].shape
        f=F.adaptive_avg_pool2d(c8['features'].flatten(0,1),c16['features'].shape[-2:]).reshape(b,n,c,*c16['features'].shape[-2:])
        return dict(c16,features=f)
    raise ValueError(mode)
