"""Opt-in exact-input geometry reuse. No dynamic or age decision is cached."""
from dataclasses import dataclass,asdict
from collections import OrderedDict
import hashlib,json,time
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from datasets.m7a_contracts import CandidateTrajectory,PlanningEvaluation,Stamp
from simulator.motion_oracle import route_projection,wrap
from simulator.oracle_driver import footprint
from simulator.m47_motion import continuous_rectangles
from diagnostics.m7a_planning import center,swept_polygon


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class PreparedCandidateGeometry:
    candidate_id:str
    trajectory_sha256:str
    original_stamp:dict
    body_half_extent:tuple
    body_center_offset:tuple
    map_version:str
    allowed_lanes:tuple
    segment_sweeps:tuple
    guide_sweeps:tuple
    curvature:tuple
    acceleration:tuple
    jerk:tuple
    static_checks:dict  # each snapshot-specific key retains exact scene/road identity


class GeometryCache:
    def __init__(self,map_sha,capacity=50000):
        self.map_sha=map_sha;self.capacity=capacity
        self.sweeps=OrderedDict();self.points=OrderedDict();self.regions={};self.prepared=OrderedDict()
        self.hits=0;self.misses=0
    def _memo(self,store,key,fn):
        if key in store:
            self.hits+=1;store.move_to_end(key);return store[key]
        self.misses+=1;v=fn();store[key]=v
        if len(store)>self.capacity:store.popitem(last=False)
        return v
    def sweep(self,a,b,half,offset):
        # Exact doubles, complete body and map identity. No spatial quantization.
        key=(self.map_sha,tuple(a[:3]),tuple(b[:3]),tuple(half),tuple(offset))
        return self._memo(self.sweeps,key,lambda:swept_polygon(a,b,half,offset))
    def region(self,polygons,allowed,inset=.03):
        key=(self.map_sha,tuple(sorted(allowed)),float(inset),tuple((str(k),polygons[k].wkb) for k in sorted(allowed)))
        if key not in self.regions:self.regions[key]=unary_union([polygons[k] for k in sorted(allowed)]).buffer(-inset)
        return self.regions[key]
    def exact(self,allowed,z,query):
        # Caller scope includes road/lane pairs and map hash; query is immutable map only.
        scope=(self.map_sha,tuple(sorted(allowed)),float(z))
        return lambda x,y:self._memo(self.points,scope+(float(x),float(y)),lambda:query(x,y))
    def prepare(self,request,candidate,allowed):
        digest=fingerprint(asdict(candidate));scene=request.scene;key=(self.map_sha,digest,tuple(allowed),tuple(scene.body_half_extent),tuple(scene.body_center_offset))
        def build():
            p=np.array(candidate.points);t=np.array(candidate.times_s);dt=np.diff(t);ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1);a=np.diff(p[:,3])/dt
            half=np.array(scene.body_half_extent)+[.2,.15];offset=scene.body_center_offset
            return PreparedCandidateGeometry(candidate.name,digest,asdict(candidate.stamp),tuple(scene.body_half_extent),tuple(offset),self.map_sha,tuple(allowed),
                tuple(self.sweep(x,y,half,offset) for x,y in zip(p[:-1],p[1:])),
                tuple(self.sweep(x,y,half,offset) for x,y in zip(candidate.guide_path[:-1],candidate.guide_path[1:])),
                tuple(abs(wrap(np.diff(p[:,2])))/np.maximum(ds,1e-6)),tuple(a),tuple(np.diff(a)/((dt[:-1]+dt[1:])/2)),{})
        return self._memo(self.prepared,key,build)


def evaluate_cached(request,candidate,road_polygon,now,*,cache,exact_road=None, future_reference=None):
    """Future reference only in evaluator; never supplied to generate()."""
    start=time.perf_counter();reasons=[];metrics={}
    try:
        request.validate(now)
        if type(candidate) is not CandidateTrajectory:
            raise TypeError('missing_or_wrong_candidate_type')
        candidate.stamp.validate(now)
        if candidate.stamp != request.scene.stamp:
            raise ValueError('snapshot_stamp_mismatch')
    except (ValueError,TypeError) as e:
        return PlanningEvaluation(getattr(candidate,'name','Missing'),False,(str(e),),())
    p=np.asarray(candidate.points,float);t=np.asarray(candidate.times_s,float)
    if p.ndim!=2 or p.shape[1]!=4 or len(p)!=len(t) or len(t)<3 or not np.isfinite(p).all() or not np.isfinite(t).all() or t[0]!=0 or np.any(np.diff(t)<=0):
        return PlanningEvaluation(candidate.name,False,('invalid_trajectory',),())
    prepared=cache.prepare(request,candidate,(hashlib.sha256(road_polygon.wkb).hexdigest(),))
    if abs(t[-1]-6.)>1e-6:reasons.append('six_second_horizon_required')
    if not np.allclose(p[0],request.scene.ego_xy_yaw_speed,atol=1e-6):reasons.append('initial_state_mismatch')
    dt=np.diff(t);ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
    integral=(p[:-1,3]+p[1:,3])*dt/2
    a=np.diff(p[:,3])/dt;j=np.diff(a)/((dt[:-1]+dt[1:])/2)
    dyaw=wrap(np.diff(p[:,2]));curv=abs(dyaw)/np.maximum(ds,1e-6)
    moving=ds>1e-3
    tangent=np.arctan2(np.diff(p[:,1]),np.diff(p[:,0]))
    midyaw=p[:-1,2]+dyaw/2
    if np.max(abs(ds-integral))>.025:reasons.append('position_velocity_inconsistent')
    if np.any(p[:,3]<0) or np.max(p[:,3])>5.6:reasons.append('speed_limit')
    if a.max()>1.51 or a.min()<-1.81:reasons.append('acceleration_limit')
    if np.max(abs(j))>2.51:reasons.append('internal_jerk_limit')
    if curv.max()>.10:reasons.append('curvature_limit')
    if np.any(abs(wrap(tangent[moving]-midyaw[moving]))>.15):reasons.append('heading_path_inconsistent')
    if not candidate.lane_change_permitted:reasons.append('lane_change_prohibited')
    half=np.array(request.scene.body_half_extent)+[.2,.15];offset=request.scene.body_center_offset
    collision_ids=set();road_fail=False;exact_fail=False;min_gap=float('inf')
    # Full body swept road check, including road holes, not just four corners.
    for i,(aa,bb) in enumerate(zip(p[:-1],p[1:])):
        sweep=prepared.segment_sweeps[i]
        if not road_polygon.covers(sweep):road_fail=True
        if exact_road is not None:
            for xy in np.asarray(sweep.exterior.coords):
                if not exact_road(*xy):exact_fail=True
        e0=center(aa,offset);e1=center(bb,offset)
        er=(np.linalg.norm(half)+np.linalg.norm(offset))*abs(wrap(bb[2]-aa[2]))/2
        emid=aa[2]+wrap(bb[2]-aa[2])/2
        for obj in request.scene.boxes:
            v=np.array(obj.velocity);o0=np.array(obj.xy)+t[i]*v
            omid=obj.yaw+obj.yaw_rate*(t[i]+dt[i]/2)
            obr=np.hypot(obj.half_length,obj.half_width)*abs(obj.yaw_rate)*dt[i]/2
            if continuous_rectangles(e0,o0,(e1-e0)/dt[i],v,half+er,
                [obj.half_length+obr,obj.half_width+obr],emid,omid,dt[i]):collision_ids.add(obj.object_id)
    if road_fail or exact_fail:reasons.append('body_outside_allowed_road')
    if collision_ids:reasons.append('predicted_collision')
    # Audit static route intent beyond horizon too; D_stop never promises bypass.
    guide=np.asarray(candidate.guide_path)
    if candidate.name!='D_stop':
        for sweep in prepared.guide_sweeps:
            if not road_polygon.covers(sweep):
                reasons.append('guide_path_road_violation');break
        static=[o for o in request.scene.boxes if np.linalg.norm(o.velocity)<1e-8 and abs(o.yaw_rate)<1e-8]
        for o in static:
            other=Polygon(footprint(*o.xy,o.yaw,o.half_length,o.half_width))
            if any(sweep.intersects(other) for sweep in prepared.guide_sweeps):
                reasons.append('guide_path_static_collision');break
    else:
        if p[-1,3]>.08:reasons.append('stop_not_reached_in_horizon')
        for o in request.scene.boxes:
            for pose,ti in zip(p,t):
                other=Polygon(footprint(*(np.array(o.xy)+ti*np.array(o.velocity)),o.yaw+ti*o.yaw_rate,o.half_length,o.half_width))
                gap=Polygon(footprint(*center(pose,offset),pose[2],*request.scene.body_half_extent)).distance(other)
                min_gap=min(min_gap,gap)
        if min_gap<3.0:reasons.append('standstill_margin_not_preserved')
    progress0,_,_,_=route_projection(request.route,p[0,:2]);progress1,_,_,_=route_projection(request.route,p[-1,:2])
    errors=[]
    if future_reference is not None:
        for o in request.scene.boxes:
            for ti,xy in future_reference.get(o.object_id,()):
                errors.append(float(np.linalg.norm(np.array(o.xy)+ti*np.array(o.velocity)-xy)))
    metrics.update(horizon_s=float(t[-1]),age_s=now-candidate.stamp.timestamp_s,
        original_timestamp_s=candidate.stamp.timestamp_s,valid_until_s=candidate.stamp.valid_until_s,
        max_acceleration=float(a.max()),max_deceleration=float(-a.min()),max_internal_jerk=float(abs(j).max()),
        max_curvature=float(curv.max()),max_position_integral_error_m=float(abs(ds-integral).max()),
        route_progress_m=progress1-progress0,endpoint_goal_distance_m=float(np.linalg.norm(p[-1,:2]-request.goal_xy_yaw[:2])),
        endpoint_heading_error_rad=float(abs(wrap(p[-1,2]-request.goal_xy_yaw[2]))),
        prediction_collision_ids=sorted(collision_ids),stop_gap_m=None if min_gap==float('inf') else min_gap,
        future_prediction_errors_m=errors,road_full_body_sweep_rejected=road_fail,exact_map_rejected=exact_fail,
        latency_ms=(time.perf_counter()-start)*1000,
        road_reference='OpenDRIVE sampled lane polygon, not curb/mesh Gold',
        collision_reference='continuous_translation_SAT_with_conservative_rotation_bound_under_CV_yawrate')
    static_key=fingerprint(dict(boxes=[asdict(b) for b in request.scene.boxes],stamp=asdict(request.scene.stamp),now=now,road_sha=hashlib.sha256(road_polygon.wkb).hexdigest()))
    prepared.static_checks[static_key]=dict(road_full_body=not road_fail,exact_map=not exact_fail,static_collisions=sorted(collision_ids))
    return PlanningEvaluation(candidate.name,not reasons,tuple(dict.fromkeys(reasons)),tuple(metrics.items()))
