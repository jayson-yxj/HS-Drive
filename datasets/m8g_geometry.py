"""Read-only evidence geometry; never used to override M8F placement acceptance."""
import math
import xml.etree.ElementTree as ET
import numpy as np


def inside_box(point, box, tolerance=.05):
    q=np.linalg.solve(np.asarray(box['T_world_from_box']),np.r_[point,1])[:3]
    return bool(np.all(np.abs(q)<=np.asarray(box['extent_m'])+tolerance))


def planned_box(result, clearance=.03):
    """Diagnostic reconstruction even if an overhead hit rejects surface selection.

    This is a hypothetical box, never an approved placement or fallback ground.
    """
    a=result['asset'];m=np.asarray(result['waypoint']['matrix'],float).copy()
    e=np.asarray(a['extent']);local=np.asarray(a['local_box_matrix'])
    # Footprint ray XY preserves the requested lateral offset.
    samples=np.array([r['start'][:2] for r in result['rays']])
    bm=m@local;m[:2,3]+=samples.mean(0)-bm[:2,3];bm=m@local
    corners=np.array([[x,y,z] for x in [-e[0],e[0]] for y in [-e[1],e[1]] for z in [-e[2],e[2]]])@bm[:3,:3].T+bm[:3,3]
    ground=[r['decision']['selected']['xyz'][2] for r in result['rays'] if r['decision'].get('selected')]
    if len(ground)!=len(samples):return None
    bm[2,3]+=float(np.median(ground)-corners[:,2].min()+clearance)
    corners=np.array([[x,y,z] for x in [-e[0],e[0]] for y in [-e[1],e[1]] for z in [-e[2],e[2]]])@bm[:3,:3].T+bm[:3,3]
    return dict(T_world_from_box=bm.tolist(),extent_m=e.tolist(),corners_world_m=corners.tolist(),
                bottom_z=float(corners[:,2].min()),top_z=float(corners[:,2].max()),
                status='HypotheticalSurfaceAlignedOBB_NotPlacementApproval')


def overhead_class(hit, box, sources, render_agrees, margin=.5):
    if len(sources)>1:return 'ConflictingGeometry'
    if not sources:return 'UnidentifiedRayHit'
    if inside_box(hit['xyz'],box,0) and render_agrees:return 'SourceQualifiedCollision'
    # A high hit alone does not prove clearance of the source's full geometry.
    if (min(p[2] for p in sources[0]['corners_world_m'])>box['top_z']+margin
            and hit['xyz'][2]>box['top_z']+margin and render_agrees):
        return 'VerifiedNonIntersectingOverhead'
    return 'UnidentifiedRayHit'


class OpenDriveGeometry:
    """Independent evaluation of line/arc reference lines and lane-width polynomials.

    Same authored OpenDRIVE as CARLA; not an independent physical mesh truth.
    Unsupported geometries/superelevation refuse proof.
    """
    def __init__(self, text):
        self.roads={int(r.attrib['id']):r for r in ET.fromstring(text).findall('road')}

    @staticmethod
    def polynomial(nodes,s,key='s'):
        eligible=[n for n in nodes if float(n.attrib.get(key,0))<=s+1e-8]
        if not eligible:return 0.
        n=max(eligible,key=lambda x:float(x.attrib.get(key,0)));d=s-float(n.attrib.get(key,0))
        return sum(float(n.attrib.get(k,0))*d**i for i,k in enumerate(['a','b','c','d']))

    def reference(self, road_id,s):
        r=self.roads[road_id];gs=r.findall('planView/geometry');g=max([g for g in gs if float(g.attrib['s'])<=s+1e-8],key=lambda x:float(x.attrib['s']))
        a=g.attrib;d=s-float(a['s']);x,y,h=map(float,[a['x'],a['y'],a['hdg']])
        if g.find('line') is not None:x+=d*math.cos(h);y+=d*math.sin(h)
        elif g.find('arc') is not None:
            k=float(g.find('arc').attrib['curvature']);x+=(math.sin(h+k*d)-math.sin(h))/k;y+=(-math.cos(h+k*d)+math.cos(h))/k;h+=k*d
        else:raise ValueError('UnsupportedOpenDrivePrimitive')
        if any(abs(float(n.attrib.get(k,0)))>1e-10 for n in r.findall('lateralProfile/superelevation') for k in ['a','b','c','d']):raise ValueError('UnsupportedSuperelevation')
        return np.array([x,y,self.polynomial(r.findall('elevationProfile/elevation'),s)]),h

    def bounds(self, road_id,lane_id,s):
        r=self.roads[road_id];sections=r.findall('lanes/laneSection');sec=max([n for n in sections if float(n.attrib['s'])<=s+1e-8],key=lambda n:float(n.attrib['s']))
        lanes={int(n.attrib['id']):n for n in sec.findall('left/lane')+sec.findall('right/lane')}
        if lane_id not in lanes:raise ValueError('MissingDeclaredLane')
        offset=self.polynomial(r.findall('lanes/laneOffset'),s);sign=1 if lane_id>0 else -1;t=offset
        for j in range(1,abs(lane_id)+1):
            n=lanes[sign*j];w=self.polynomial(n.findall('width'),s-float(sec.attrib['s']),'sOffset')
            if w<=0:raise ValueError('NonPositiveLaneWidth')
            if j==abs(lane_id):return sorted([t,t+sign*w]),sections.index(sec),n.attrib['type']
            t+=sign*w

    def lane_center(self,road_id,lane_id,s):
        ref,h=self.reference(road_id,s);bounds,section,kind=self.bounds(road_id,lane_id,s);t=sum(bounds)/2
        return np.array([ref[0]-math.sin(h)*t,-(ref[1]+math.cos(h)*t),ref[2]])

    def point(self,road_id,lane_id,s_hint,point,inset=.1):
        road=self.roads[road_id];lo=max(0,s_hint-12);hi=min(float(road.attrib['length']),s_hint+12)
        ss=np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)/.05))+1));p=np.array([point[0],-point[1]])
        refs=np.array([self.reference(road_id,s)[0][:2] for s in ss]);i=int(np.argmin(np.linalg.norm(refs-p,axis=1)));s=float(ss[i])
        # One bounded tangent projection refines the .05m sample, on line/arc.
        ref,h=self.reference(road_id,s);s=float(np.clip(s+np.dot(p-ref[:2],[math.cos(h),math.sin(h)]),lo,hi))
        ref,h=self.reference(road_id,s);t=float(np.dot(p-ref[:2],[-math.sin(h),math.cos(h)]));bounds,section,kind=self.bounds(road_id,lane_id,s)
        margin=min(t-bounds[0],bounds[1]-t);zdelta=abs(float(point[2])-float(ref[2]))
        longitudinal_residual=float(np.dot(p-ref[:2],[math.cos(h),math.sin(h)]))
        return dict(s=s,t=t,bounds=bounds,margin_m=margin,lane_width_m=bounds[1]-bounds[0],section_id=section,lane_type=kind,z_delta_m=zdelta,longitudinal_residual_m=longitudinal_residual,
                    within=bool(margin>=inset and zdelta<=.5 and abs(longitudinal_residual)<=.02),center=self.lane_center(road_id,lane_id,s).tolist(),reference='IndependentOpenDRIVEArithmetic_NotIndependentMesh')
