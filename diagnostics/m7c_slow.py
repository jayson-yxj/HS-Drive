"""One opt-in analytic slow-follow family, no inference/control or GT input."""
from dataclasses import replace
import numpy as np
from shapely.geometry import Polygon
from diagnostics.m7b_motion import ArcRoute,interpolate
from diagnostics.m7a_planning import center
from datasets.m7a_contracts import CandidateTrajectory
from simulator.oracle_driver import footprint


def slow_candidate(request,target=.5,duration=4.5):
    v0=request.scene.ego_xy_yaw_speed[3];t=np.arange(61)*.1;u=np.minimum(t,duration)
    v=target+(v0-target)/2*(1+np.cos(np.pi*u/duration));v[t>=duration]=target
    distance=target*t+(v0-target)/2*(u+duration/np.pi*np.sin(np.pi*u/duration))
    route=ArcRoute(request.route);s0,_=route.project(np.array(request.scene.ego_xy_yaw_speed[:2]));p=route.at(s0+distance)
    return CandidateTrajectory(request.scene.stamp,'E_slow_follow',tuple(map(tuple,np.c_[p,v])),tuple(t),request.route)


def stopping_reserve(candidate,scene,hypotheses,route):
    """Extra conservative reserve for slow candidate, no relaxed collision rule.

    Requires >=3m plus jerk-ramp/decel stopping allowance to every actor at every
    sample. This can reject harmless off-lane actors. Tail available is separately
    checked; does not prove braking response of the actual CARLA vehicle.
    """
    boxes={o.object_id:o for o in scene.boxes};failures=[];minimum=None
    for h in hypotheses:
        p=np.array(h.poses);ego=interpolate(candidate.points,candidate.times_s,h.times);b=boxes[h.object_id]
        for ep,op,t in zip(ego,p,h.times):
            body=Polygon(footprint(*center(ep,scene.body_center_offset),ep[2],*scene.body_half_extent));ob=Polygon(footprint(*op[:2],op[2],b.half_length,b.half_width))
            margin=body.distance(ob)-(3.+ep[3]**2/(2*1.8)+ep[3]*1.8/2.5)
            minimum=margin if minimum is None else min(minimum,margin)
            if margin< -1e-8:
                failures.append(dict(actor=h.object_id,hypothesis=h.name,time_s=float(t),reserve_deficit_m=float(-margin)));break
    endpoint=np.array(candidate.points[-1]);s,_=route.project(endpoint[:2]);remaining=route.length-s-scene.body_half_extent[0]-scene.body_center_offset[0]
    required=3+endpoint[3]**2/(2*1.8)+endpoint[3]*1.8/2.5
    if remaining<required:failures.append(dict(reason='route_tail_stop_reserve',remaining=remaining,required=required))
    return dict(passed=not failures,failures=failures,minimum_actor_reserve_m=minimum,route_tail_m=remaining,stop_distance_required_m=required,meaning='Conservative finite-hypothesis planning reserve; FeasibleToExecute=NotValidated')
