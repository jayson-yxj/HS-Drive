"""Finite forward waypoint continuation; ambiguity stays unknown."""
from dataclasses import dataclass
import numpy as np
from diagnostics.m7b_motion import ArcRoute
from simulator.motion_oracle import wrap
import math

def pose(w):
    t=w.transform
    return (t.location.x,-t.location.y,-math.radians(t.rotation.yaw))


@dataclass
class ConnectedArcRoute:
    route:ArcRoute
    nodes:list
    transitions:list
    status:str
    stop_reason:str
    length_requested:float
    source:str='Privileged_OpenDRIVE_WaypointSuccessors'
    def at(self,s):return self.route.at(s)
    def tangent(self,s):return self.route.tangent(s)
    def project(self,xy):return self.route.project(xy)
    @property
    def length(self):return self.route.length
    @property
    def points(self):return self.route.points


def node(w):
    return dict(road=int(w.road_id),lane=int(w.lane_id),section=int(w.section_id),s=float(w.s),width=float(w.lane_width),junction=bool(w.is_junction),pose=list(pose(w)))


def continue_waypoints(start,required_length=100.,step=.5,max_extra=100.,intent=None):
    """No intent => exactly one driving successor; never choose arbitrary branch.

    intent maps current (road,lane) to an explicitly authorized next pair.
    Caller must supply legal route intent, not evaluator future motion.
    """
    intent={} if intent is None else intent;nodes=[node(start)];wps=[start];trans=[];length=0.;reason='BudgetLimit';status='PredictionOutOfSupport'
    for _ in range(int(max_extra/step)+1):
        if length>=required_length:status='Supported';reason='RequestedLengthReached';break
        w=wps[-1];raw=w.next(step)
        good=[x for x in raw if str(x.lane_type).split('.')[-1]=='Driving' and x.lane_width>0]
        choices={(int(x.road_id),int(x.lane_id),int(x.section_id)):x for x in good}
        if not choices:reason='NoLegalSuccessor';break
        intended=intent.get((int(w.road_id),int(w.lane_id)))
        if len(choices)>1:
            if intended is None:reason='AmbiguousSuccessors';trans.append(dict(from_node=node(w),allowed_successors=[node(x) for x in choices.values()],selected=None,reason=reason));break
            choices={k:x for k,x in choices.items() if k[:2]==tuple(intended)}
            if len(choices)!=1:reason='IllegalOrAmbiguousIntent';break
        nxt=next(iter(choices.values()));a=np.array(pose(w));b=np.array(pose(nxt));distance=float(np.linalg.norm(b[:2]-a[:2]));yaw=float(abs(wrap(b[2]-a[2])))
        forward=float(np.dot(b[:2]-a[:2],[np.cos(a[2]),np.sin(a[2])]))
        if distance<1e-5 or distance>.75 or yaw>.15 or forward<=0:reason='ConnectionDiscontinuityOrReverse';trans.append(dict(from_node=node(w),to_node=node(nxt),distance=distance,yaw=yaw,forward=forward,accepted=False));break
        if (w.road_id,w.lane_id,w.section_id)!=(nxt.road_id,nxt.lane_id,nxt.section_id):
            trans.append(dict(from_node=node(w),to_node=node(nxt),allowed_successors=[node(x) for x in good],selected=(int(nxt.road_id),int(nxt.lane_id)),selection='UniqueLegalSuccessor' if len(good)==1 else 'ExplicitIntent',distance=distance,yaw=yaw,forward=forward,accepted=True))
        length+=distance;wps.append(nxt);nodes.append(node(nxt))
    if len(nodes)<2:raise ValueError('NoPositiveForwardSupport:'+reason)
    return ConnectedArcRoute(ArcRoute([n['pose'] for n in nodes],[n['lane'] for n in nodes]),nodes,trans,status,reason,required_length)


def extend_existing(cmap,points,z,extra=60.):
    """Preserve original prefix bytes, extend actual final waypoint forward."""
    import carla
    p=np.array(points);last=p[-1];wp=cmap.get_waypoint(carla.Location(x=float(last[0]),y=-float(last[1]),z=z),project_to_road=False,lane_type=carla.LaneType.Driving)
    if wp is None:raise ValueError('LastPointNotDriving')
    # CARLA nearest waypoint lookup can return a cached ~0.5m neighbor.
    # Refine longitudinal s by the exact signed offset using get_waypoint_xodr.
    anchor=np.array(pose(wp));signed=float(np.dot(last[:2]-anchor[:2],[np.cos(anchor[2]),np.sin(anchor[2])]))
    refined=cmap.get_waypoint_xodr(wp.road_id,wp.lane_id,wp.s+signed*(-1 if wp.lane_id>0 else 1))
    if refined is not None:wp=refined
    anchor=np.array(pose(wp))
    if np.linalg.norm(anchor[:2]-last[:2])>.15 or abs(wrap(anchor[2]-last[2]))>.08:raise ValueError('LastPointAnchorMismatch')
    continuation=continue_waypoints(wp,required_length=extra,max_extra=extra)
    # Preserve prefix; verify join from actual stored last sample to first successor.
    tail=continuation.points[1:];join=float(np.linalg.norm(tail[0,:2]-last[:2]))
    if not 1e-6<join<=.75:raise ValueError('JoinDiscontinuity')
    combined=ArcRoute(np.vstack([p,tail]));return combined,continuation,join
