"""Exact grid-plane segment traversal for nonuniform Z layers (CPU reference).

All boundary crossings are enumerated and sorted; midpoint classification visits
exactly positive-length voxel intersections, never fixed-distance stepping.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VoxelGrid:
    x: tuple=(-20.,60.)
    y: tuple=(-30.,30.)
    z: tuple=(-2.,-.5,1.,2.5,4.)
    resolution: float=.5

    def __post_init__(self):
        if self.resolution<=0 or any(not np.isfinite(v).all() for v in (self.x,self.y,self.z)):raise ValueError('Invalid grid')
        if any(np.any(np.diff(v)<=0) for v in (self.x,self.y,self.z)):raise ValueError('Nonincreasing edges')
        if any(not np.isclose((v[1]-v[0])/self.resolution,round((v[1]-v[0])/self.resolution)) for v in (self.x,self.y)):raise ValueError('Noninteger grid')

    @property
    def shape(self):return (len(self.z)-1,round((self.x[1]-self.x[0])/self.resolution),round((self.y[1]-self.y[0])/self.resolution))

    @property
    def edges(self):return (np.linspace(*self.x,self.shape[1]+1),np.linspace(*self.y,self.shape[2]+1),np.array(self.z))

    def locate(self,points):
        p=np.asarray(points);ix=np.floor((p[...,0]-self.x[0])/self.resolution).astype(np.int64);iy=np.floor((p[...,1]-self.y[0])/self.resolution).astype(np.int64);iz=np.searchsorted(self.z,p[...,2],side='right')-1
        valid=np.isfinite(p).all(-1)&(ix>=0)&(ix<self.shape[1])&(iy>=0)&(iy<self.shape[2])&(iz>=0)&(iz<self.shape[0])
        rank=(iz*self.shape[1]+ix)*self.shape[2]+iy
        return np.where(valid,rank,-1),valid


def box_interval(origin,direction,lower,upper):
    lo=-np.inf;hi=np.inf
    for o,d,a,b in zip(origin,direction,lower,upper):
        if abs(d)<1e-12:
            if not a<=o<=b:return None
        else:
            u,v=sorted(((a-o)/d,(b-o)/d));lo=max(lo,u);hi=min(hi,v)
    return (lo,hi) if hi>lo else None


def trace_segment(origin,endpoint,grid,*,start_distance=0.,end_margin=0.):
    o=np.asarray(origin,float);e=np.asarray(endpoint,float)
    if o.shape!=(3,) or e.shape!=(3,) or not np.isfinite([o,e]).all():return np.empty(0,np.int64)
    delta=e-o;length=np.linalg.norm(delta)
    if length<=start_distance+end_margin:return np.empty(0,np.int64)
    begin=start_distance/length;end=1-end_margin/length
    crossing=[begin,end]
    for axis,edges in enumerate(grid.edges):
        if abs(delta[axis])>1e-12:
            ts=(edges-o[axis])/delta[axis];crossing.extend(ts[(ts>begin)&(ts<end)].tolist())
    ts=np.unique(crossing);mid=(ts[:-1]+ts[1:])/2
    rank,valid=grid.locate(o[None]+mid[:,None]*delta)
    # Exactly on a face uses the half-open higher-side convention.
    return rank[valid]
