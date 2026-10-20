"""Precompiled conservative route corridor with exact-map boundary fallback.

This is a map approximation requiring disagreement audit; never silently called
an exact road mesh. Initialization cost is reported outside recurring latency.
"""
import time
import numpy as np


class RouteRoad:
    def __init__(self,cmap,route,z=0.,margin=.25):
        import carla
        start=time.perf_counter();self.cmap=cmap;self.z=z;self.cache={};self.api_calls=0
        route=np.asarray(route);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
        ss=np.arange(0,arc[-1],.25);xy=np.stack([np.interp(ss,arc,route[:,i]) for i in (0,1)],1)
        points=[];width=[]
        for x,y in xy:
            wp=cmap.get_waypoint(carla.Location(x=float(x),y=float(-y),z=z),project_to_road=False,lane_type=carla.LaneType.Driving)
            if wp is None:points.append([x,y]);width.append(0.)
            else:points.append([wp.transform.location.x,-wp.transform.location.y]);width.append(max(0,wp.lane_width/2-margin))
        self.points=np.asarray(points);self.width=np.minimum(width[:-1],width[1:]);self.delta=np.diff(self.points,axis=0)
        self.length2=np.maximum((self.delta**2).sum(-1),1e-12);self.init_ms=(time.perf_counter()-start)*1000

    def fast(self,points):
        points=np.atleast_2d(points);off=points[:,None,:]-self.points[None,:-1]
        frac=np.clip((off*self.delta).sum(-1)/self.length2,0,1)
        distance=np.linalg.norm(off-frac[:,:,None]*self.delta,axis=-1)
        return (distance<self.width).any(-1)

    def exact(self,x,y):
        import carla
        cell=(float(x),float(y))
        if cell not in self.cache:
            self.api_calls+=1
            self.cache[cell]=self.cmap.get_waypoint(carla.Location(x=float(x),y=float(-y),z=self.z),
                project_to_road=False,lane_type=carla.LaneType.Driving) is not None
        return self.cache[cell]

    def __call__(self,x,y):
        return bool(self.fast([[x,y]])[0]) or self.exact(x,y)

    def batch(self,points):
        accepted=self.fast(points)
        for i in np.flatnonzero(~accepted):accepted[i]=self.exact(*points[i])
        return accepted
