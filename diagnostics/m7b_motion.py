"""Opt-in arc geometry and finite snapshot-only motion hypotheses; no control API."""
from dataclasses import dataclass
import numpy as np
from simulator.motion_oracle import wrap
from simulator.m47_motion import continuous_rectangles
from diagnostics.m7a_planning import center


class ArcRoute:
    """Piecewise linear XY, unwrapped waypoint heading, explicit finite support."""
    def __init__(self, points, lane_ids=None):
        self.points=np.asarray(points,float)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or len(self.points)<2 or not np.isfinite(self.points).all():
            raise ValueError('invalid_route')
        self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.points[:,:2],axis=0),axis=1))]
        if np.any(np.diff(self.arc)<1e-6):raise ValueError('duplicate_route_point')
        self.headings=np.unwrap(self.points[:,2])
        self.lanes=np.asarray(lane_ids if lane_ids is not None else [0]*len(self.points),int)
        self.length=float(self.arc[-1])
    def at(self,s):
        q=np.asarray(s,float)
        if not np.isfinite(q).all() or np.any(q< -1e-8) or np.any(q>self.length+1e-8):raise ValueError('PredictionOutOfSupport_route_end')
        return np.stack([np.interp(q,self.arc,self.points[:,0]),np.interp(q,self.arc,self.points[:,1]),np.interp(q,self.arc,self.headings)],axis=-1)
    def tangent(self,s):
        yaw=self.at(s)[...,2];return np.stack([np.cos(yaw),np.sin(yaw)],axis=-1)
    def lane_id(self,s):
        self.at(s);return self.lanes[np.minimum(np.searchsorted(self.arc,s,side='right')-1,len(self.lanes)-1)]
    def project(self,xy):
        d=np.diff(self.points[:,:2],axis=0);f=np.clip(np.sum((xy-self.points[:-1,:2])*d,axis=1)/np.sum(d*d,axis=1),0,1)
        p=self.points[:-1,:2]+f[:,None]*d;i=int(np.argmin(np.linalg.norm(p-xy,axis=1)))
        return float(self.arc[i]+f[i]*(self.arc[i+1]-self.arc[i])),float(np.linalg.norm(p[i]-xy))


def longitudinal(t,v0,a,onset=0.,vmax=8.):
    """Analytic displacement/speed: delayed constant acceleration, bounded speed."""
    t=np.asarray(t,float);u=np.maximum(t-onset,0.)
    limit=0. if a<0 else vmax
    active=np.minimum(u,abs((limit-v0)/a)) if a else u
    v=np.clip(v0+a*active,0,vmax)
    s=v0*np.minimum(t,onset)+v0*active+.5*a*active**2+v*(u-active)
    return s,v


@dataclass(frozen=True)
class ObservedMotion:
    object_id:str
    route:ArcRoute
    s0:float
    mode:str='road_vehicle'  # static geometry or road vehicle only; crossing linear corridor supported explicitly
    source:str='Privileged_current_snapshot_and_map_no_future'


@dataclass(frozen=True)
class MotionHypothesis:
    object_id:str
    name:str
    times:tuple
    poses:tuple  # x y yaw speed
    acceleration:float
    onset_s:float
    source:str='Privileged_snapshot_finite_motion_assumption'


def build_hypotheses(scene,observations,cfg):
    """No evaluator references accepted. Unsupported actors invalidate robust claims."""
    observations={o.object_id:o for o in observations};out=[];unknown=[]
    t=np.arange(0,cfg['horizon_s']+1e-8,cfg['motion_dt_s'])
    for box in scene.boxes:
        o=observations.get(box.object_id)
        if o is None:unknown.append(dict(object_id=box.object_id,reason='missing_motion_observation'));continue
        v0=float(np.linalg.norm(box.velocity))
        if not cfg['actor_speed_support'][0]<=v0<=cfg['actor_speed_support'][1]:
            unknown.append(dict(object_id=box.object_id,reason='initial_speed_out_of_support'));continue
        pose=o.route.at(o.s0)
        if np.linalg.norm(pose[:2]-box.xy)>cfg['path_match_tolerance_m'] or abs(wrap(pose[2]-box.yaw))>cfg['path_heading_tolerance_rad'] or (v0>1e-6 and np.linalg.norm(np.array(box.velocity)-o.route.tangent(o.s0)*v0)>.12):
            unknown.append(dict(object_id=box.object_id,reason='snapshot_path_mismatch'));continue
        specs=[(0.,0.)] if o.mode=='static_geometry' else [(a,onset) for a in cfg['hypothesis_accelerations'] for onset in (cfg['hypothesis_onsets_s'] if a else [0.])]
        seen=set()
        for a,onset in specs:
            s,v=longitudinal(t,v0,a,onset,cfg['actor_speed_support'][1])
            try:p=o.route.at(o.s0+s)
            except ValueError:
                unknown.append(dict(object_id=box.object_id,reason='route_end',acceleration=a,onset=onset));continue
            points=np.c_[p,v];key=points.tobytes()
            if key in seen:continue
            seen.add(key)
            name='constant_speed_on_route' if a==0 else ('braking' if a<0 else 'acceleration')+f'_{a}_delay{onset}'
            out.append(MotionHypothesis(box.object_id,name,tuple(t),tuple(map(tuple,points)),a,onset))
    return out,unknown


def interpolate(points,times,q):
    p=np.array(points);return np.stack([np.interp(q,times,np.unwrap(p[:,i]) if i==2 else p[:,i]) for i in range(p.shape[1])],axis=-1)


def evaluate_dynamic_hypotheses(candidate,observed_scene,hypotheses,unknown=()):
    """Conservative SAT for each piecewise linear trajectory + rotation bounds.

    Does not certify unenumerated accelerations, intent changes, or continuous meshes.
    Unknown never becomes free. No evaluator future is an argument.
    """
    try:
        observed_scene.validate(observed_scene.stamp.timestamp_s)
        pcheck=np.array(candidate.points,float);tcheck=np.array(candidate.times_s,float)
        if candidate.stamp!=observed_scene.stamp or pcheck.shape!=(len(tcheck),4) or len(tcheck)<3 or not np.isfinite(pcheck).all() or not np.isfinite(tcheck).all() or tcheck[0]!=0 or abs(tcheck[-1]-6)>1e-8 or np.any(np.diff(tcheck)<=0):
            raise ValueError('invalid_candidate_time_or_values')
    except (ValueError,TypeError,AttributeError):
        return dict(status='InvalidInput',robust=False,collisions=[],stop_margin_failures=[],unknown=[dict(reason='invalid_input')],hypotheses_checked=0)
    collisions=[];gaps=[];boxes={o.object_id:o for o in observed_scene.boxes}
    if set(boxes)-{h.object_id for h in hypotheses}:
        unknown=tuple(unknown)+(dict(reason='missing_actor_hypotheses'),)
    from shapely.geometry import Polygon
    from simulator.oracle_driver import footprint
    stop=candidate.name=='D_stop' or candidate.name.startswith('E_stop')
    for h in hypotheses:
        box=boxes[h.object_id];t=np.array(h.times);p=np.array(h.poses)
        ego=interpolate(candidate.points,candidate.times_s,t);half=np.array(observed_scene.body_half_extent)+[.2,.15]
        offset=observed_scene.body_center_offset;first=None;mingap=float('inf')
        # Numerical vectorization preserves all SAT interval and rotation bounds.
        ecenter=np.array([center(pose,offset) for pose in ego])
        ea=wrap(np.diff(ego[:,2]));oa=wrap(np.diff(p[:,2]))
        er=(np.linalg.norm(half)+np.linalg.norm(offset))*abs(ea)/2
        obr=np.hypot(box.half_length,box.half_width)*abs(oa)/2
        hits=sat_intervals_vectorized(ecenter[:-1],ecenter[1:],p[:-1,:2],p[1:,:2],
            np.broadcast_to(half,(len(t)-1,2))+er[:,None],
            np.broadcast_to([box.half_length,box.half_width],(len(t)-1,2))+obr[:,None],
            ego[:-1,2]+ea/2,p[:-1,2]+oa/2)
        if hits.any():first=float(t[np.flatnonzero(hits)[0]])
        if stop:
            import shapely
            ep=shapely.polygons(np.array([footprint(*xy,pose[2],*observed_scene.body_half_extent) for xy,pose in zip(ecenter,ego)]))
            op=shapely.polygons(np.array([footprint(*pose[:2],pose[2],box.half_length,box.half_width) for pose in p]))
            mingap=float(shapely.distance(ep,op).min())
            if mingap<3.-1e-8:gaps.append(dict(object_id=h.object_id,hypothesis=h.name,gap_m=mingap))
        if first is not None:collisions.append(dict(object_id=h.object_id,hypothesis=h.name,first_interval_start_s=first))
    status='PredictionOutOfSupport' if unknown else 'CollisionUnderHypothesis' if collisions else 'StopMarginUnderHypothesis' if gaps else 'RobustWithinHypotheses'
    return dict(status=status,robust=status=='RobustWithinHypotheses',collisions=collisions,stop_margin_failures=gaps,unknown=list(unknown),hypotheses_checked=len(hypotheses))


def stop_witness(request,duration):
    """Analytic bounded-jerk cosine-speed stop; offline witness, no actuation."""
    from datasets.m7a_contracts import CandidateTrajectory
    v0=request.scene.ego_xy_yaw_speed[3];t=np.arange(61)*.1;u=np.minimum(t,duration)
    v=v0/2*(1+np.cos(np.pi*u/duration));v[t>=duration]=0
    distance=v0/2*(u+duration/np.pi*np.sin(np.pi*u/duration))
    route=ArcRoute(request.route);s0,_=route.project(np.array(request.scene.ego_xy_yaw_speed[:2]));p=route.at(s0+distance)
    return CandidateTrajectory(request.scene.stamp,f'E_stop_T{duration}',tuple(map(tuple,np.c_[p,v])),tuple(t),request.route)


def sat_intervals_vectorized(e0,e1,o0,o1,half,other_half,eyaw,oyaw):
    """Batch exact constant-orientation translation SAT on normalized time[0,1]."""
    ec=np.stack([np.cos(eyaw),np.sin(eyaw)],-1);oc=np.stack([np.cos(oyaw),np.sin(oyaw)],-1)
    eb=np.stack([ec,np.stack([-ec[:,1],ec[:,0]],-1)],1)
    ob=np.stack([oc,np.stack([-oc[:,1],oc[:,0]],-1)],1)
    axes=np.concatenate([eb,ob],axis=1)
    rad=np.sum(abs(np.einsum('naj,nbj->nab',axes,eb))*half[:,None,:],-1)+np.sum(abs(np.einsum('naj,nbj->nab',axes,ob))*other_half[:,None,:],-1)
    pos=np.einsum('naj,nj->na',axes,o0-e0);vel=np.einsum('naj,nj->na',axes,(o1-o0)-(e1-e0))
    parallel=abs(vel)<1e-10;den=np.where(parallel,1.,vel)
    aa=(-rad-pos)/den;bb=(rad-pos)/den
    low=np.where(parallel,-np.inf,np.minimum(aa,bb));high=np.where(parallel,np.inf,np.maximum(aa,bb))
    possible=~np.any(parallel&(abs(pos)>rad),axis=1)
    return possible & (np.maximum(low.max(axis=1),0)<=np.minimum(high.min(axis=1),1))
