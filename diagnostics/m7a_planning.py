"""Independent candidate evaluator; reuses frozen motion planner and SAT."""
import time
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from datasets.m7a_contracts import CandidateTrajectory, PlanningEvaluation, Stamp
from simulator.motion_oracle import MotionPlanner, OracleState, route_projection, wrap
from simulator.m6b_bypass import offset_path
from simulator.oracle_driver import footprint
from simulator.m47_motion import continuous_rectangles


def center(pose, offset):
    x, y, yaw = pose[:3]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x+c*offset[0]-s*offset[1], y+s*offset[0]+c*offset[1]])


def swept_polygon(a, b, half, offset=(0., 0.)):
    """Conservative bound for linear center motion and shortest-arc rotation.

    Includes the midpoint-rotation error bound. Not an exact curved mesh sweep.
    """
    angle = abs(wrap(b[2]-a[2]))
    pad = (np.linalg.norm(half)+np.linalg.norm(offset))*angle/2
    hull = unary_union([Polygon(footprint(*center(p, offset), p[2], *half)) for p in (a,b)]).convex_hull
    # GEOS round buffers approximate a disk by an inscribed polygon; inflate
    # by 1/cos(pi/16) so the polygon contains the desired radius.
    return hull.buffer(pad/np.cos(np.pi/16), quad_segs=4) if pad else hull


def generate(request, *, left_offset=None, right_offset=None, lane_permitted=True):
    request.validate(request.scene.stamp.timestamp_s)
    scene=request.scene
    state=OracleState(*scene.ego_xy_yaw_speed,scene.stamp.timestamp_s)
    routes=[('A_keep_route',np.array(request.route),True)]
    for name,offset in [('B_left',left_offset),('C_right',right_offset)]:
        if offset is not None:
            routes.append((name,offset_path(request.route,offset),lane_permitted))
    routes.append(('D_stop',np.array(request.route),True))
    out=[]
    for name,path,legal in routes:
        trajectory,_=MotionPlanner().plan(path,state,(),max_speed=0. if name=='D_stop' else 2.5)
        p=trajectory.points
        c,s=np.cos(state.yaw),np.sin(state.yaw)
        xy=p[:,:2]@np.array([[c,s],[-s,c]])+[state.x,state.y]
        points=np.vstack((scene.ego_xy_yaw_speed,np.c_[xy,p[:,2]+state.yaw,p[:,3]]))
        out.append(CandidateTrajectory(scene.stamp,name,tuple(map(tuple,points)),
            tuple(np.arange(len(points))*.1),tuple(map(tuple,path)),legal))
    return out


def evaluate(request,candidate,road_polygon,now,*,exact_road=None, future_reference=None):
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
        sweep=swept_polygon(aa,bb,half,offset)
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
        for aa,bb in zip(guide[:-1],guide[1:]):
            sweep=swept_polygon(aa,bb,half,offset)
            if not road_polygon.covers(sweep):
                reasons.append('guide_path_road_violation');break
        static=[o for o in request.scene.boxes if np.linalg.norm(o.velocity)<1e-8 and abs(o.yaw_rate)<1e-8]
        for o in static:
            other=Polygon(footprint(*o.xy,o.yaw,o.half_length,o.half_width))
            if any(swept_polygon(aa,bb,half,offset).intersects(other) for aa,bb in zip(guide[:-1],guide[1:])):
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
    return PlanningEvaluation(candidate.name,not reasons,tuple(dict.fromkeys(reasons)),tuple(metrics.items()))


def select(evaluations):
    for name in ['A_keep_route','B_left','C_right','D_stop']:
        if any(e.candidate==name and e.planning_feasible for e in evaluations):
            return dict(selected=name,decision='StopAndDegrade' if name=='D_stop' else 'PlanningFeasible',control_authorized=False)
    return dict(selected=None,decision='RejectNoFeasibleCandidate_DegradationRequired',control_authorized=False)
