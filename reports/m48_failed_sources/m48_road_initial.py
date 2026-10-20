"""Exact-coordinate cache and analytically certified straight-lane interiors.

Only constant-width, zero-lane-offset OpenDRIVE line primitives are eligible.
Curves, junctions, changing width, unsupported geometry and borders fall back
onto CARLA's full map lookup. A certificate does not infer traffic direction.
"""
import math,time
import xml.etree.ElementTree as ET
import numpy as np

class CertifiedRoad:
    def __init__(self,cmap,z=0.,margin=.20):
        self.cmap=cmap;self.z=z;self.cache={};self.calls=0;self.fast_hits=0
        self.rectangles=[];start=time.perf_counter();root=ET.fromstring(cmap.to_opendrive())
        for road in root.findall('road'):
            if road.attrib.get('junction','-1')!='-1':continue
            laneoffs=road.findall('lanes/laneOffset')
            if any(any(abs(float(l.attrib.get(k,0)))>1e-12 for k in ['a','b','c','d']) for l in laneoffs):continue
            sections=road.findall('lanes/laneSection')
            for i,section in enumerate(sections):
                low=float(section.attrib['s']);high=float(sections[i+1].attrib['s']) if i+1<len(sections) else float(road.attrib['length'])
                for side,sign in [('left',1),('right',-1)]:
                    lanes=sorted(section.findall(side+'/lane'),key=lambda x:abs(int(x.attrib['id'])))
                    offset=0.
                    for lane in lanes:
                        widths=lane.findall('width')
                        if len(widths)!=1 or any(abs(float(widths[0].attrib.get(k,0)))>1e-12 for k in ['sOffset','b','c','d']):break
                        width=float(widths[0].attrib['a']);center=sign*(offset+width/2);offset+=width
                        if lane.attrib['type']!='driving' or width<=2*margin:continue
                        for geo in road.findall('planView/geometry'):
                            if geo.find('line') is None:continue
                            gs=float(geo.attrib['s']);gl=float(geo.attrib['length']);a=max(low,gs)+margin;b=min(high,gs+gl)-margin
                            if b<=a:continue
                            heading=float(geo.attrib['hdg']);direction=np.array([math.cos(heading),-math.sin(heading)]);left=np.array([-math.sin(heading),-math.cos(heading)])
                            origin=np.array([float(geo.attrib['x']),-float(geo.attrib['y'])])+direction*(a-gs)+left*center
                            self.rectangles.append((origin,direction,left,b-a,width/2-margin))
        if self.rectangles:
            self.origins=np.array([r[0] for r in self.rectangles]);self.dirs=np.array([r[1] for r in self.rectangles]);self.left=np.array([r[2] for r in self.rectangles]);self.length=np.array([r[3] for r in self.rectangles]);self.half=np.array([r[4] for r in self.rectangles])
        self.init_ms=(time.perf_counter()-start)*1000
    def certified(self,points):
        p=np.atleast_2d(points)
        if not self.rectangles:return np.zeros(len(p),bool)
        delta=p[:,None,:]-self.origins[None];along=(delta*self.dirs).sum(-1);across=abs((delta*self.left).sum(-1))
        return ((along>0)&(along<self.length)&(across<self.half)).any(-1)
    def exact(self,x,y):
        import carla
        key=(float(x),float(y))
        if key not in self.cache:
            self.calls+=1
            self.cache[key]=self.cmap.get_waypoint(carla.Location(x=float(x),y=-float(y),z=self.z),project_to_road=False,lane_type=carla.LaneType.Driving) is not None
        return self.cache[key]
    def __call__(self,x,y):
        # Certificate disabled in closed loop until multi-map disagreement audit.
        return self.exact(x,y)
    def batch(self,points,*,enable_certificates=False):
        accepted=self.certified(points) if enable_certificates else np.zeros(len(points),bool)
        self.fast_hits+=int(accepted.sum())
        for i in np.flatnonzero(~accepted):accepted[i]=self.exact(*points[i])
        return accepted
