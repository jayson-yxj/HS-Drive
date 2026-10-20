"""Depth reprojection diagnostics, with explicit visibility/discontinuity counts."""
import cv2
import numpy as np
from .reference import transform_points, unproject_pixels_to_optical


def compare_depths(source, target, source_K, target_K, T_target_from_source, config, stride=4):
    v,u=np.mgrid[0:source.shape[0]:stride,0:source.shape[1]:stride]
    d=source[::stride,::stride]
    good=np.isfinite(d)&(d>=config['min_depth_m'])&(d<config['max_depth_m'])
    p=unproject_pixels_to_optical(u[good],v[good],d[good],source_K)
    p=transform_points(p,T_target_from_source)
    K=np.asarray(target_K)
    positive=p[:,2]>.1
    p=p[positive]
    uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3]
    inside=(uv[:,0]>=2)&(uv[:,0]<target.shape[1]-3)&(uv[:,1]>=2)&(uv[:,1]<target.shape[0]-3)
    p=p[inside];uv=uv[inside]
    if not len(p):return {'projected':0,'smooth_valid':0,'errors_m':np.array([])}
    x=np.rint(uv[:,0]).astype(int);y=np.rint(uv[:,1]).astype(int)
    lo=cv2.erode(target,np.ones((3,3),np.uint8));hi=cv2.dilate(target,np.ones((3,3),np.uint8))
    sampled=cv2.remap(target,uv[:,0].astype(np.float32).reshape(-1,1),
        uv[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel()
    valid=(lo[y,x]>=config['min_depth_m'])&(hi[y,x]<config['max_depth_m'])
    smooth=valid & ((hi[y,x]-lo[y,x])<config['max_patch_spread_m'])
    delta=p[:,2]-sampled
    # Points behind a nearer observed surface are occluded, not contradictions.
    occluded=smooth & (delta>config['agreement_tolerance_m'])
    comparable=smooth & ~occluded
    return {'projected':len(p),'smooth_valid':int(smooth.sum()),'occluded':int(occluded.sum()),
        'comparable':int(comparable.sum()),'errors_m':delta[comparable],
        'all_smooth_errors_m':delta[smooth]}


def summarize(errors, tolerance_m=.5):
    errors=np.asarray(errors)
    if not len(errors):return {'count':0,'median_abs_m':None,'p95_abs_m':None,'agreement_fraction':0.0}
    a=np.abs(errors)
    return {'count':len(a),'median_abs_m':float(np.median(a)),
            'p95_abs_m':float(np.quantile(a,.95)),
            'agreement_fraction':float(np.mean(a<=tolerance_m))}
