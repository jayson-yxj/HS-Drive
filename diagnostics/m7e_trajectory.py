"""Opt-in single state-anchored quintic geometry, no actuator interface."""
import numpy as np
from numpy.polynomial.polynomial import polyval
from diagnostics.m7b_motion import ArcRoute
from datasets.m7a_contracts import CandidateTrajectory
from simulator.motion_oracle import wrap


class InitialStateConnectionInfeasible(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__('InitialStateConnectionInfeasible:' + reason)


def initial_state(request):
    route = ArcRoute(request.route)
    x,y,yaw,v = request.scene.ego_xy_yaw_speed
    s,distance = route.project(np.array([x,y]));p=route.at(s)
    lateral=float(np.dot(np.array([x,y])-p[:2],[-np.sin(p[2]),np.cos(p[2])]))
    return dict(ego_x=x,ego_y=y,ego_yaw=yaw,v_forward_mps=v,route_progress_s=s,
                lateral_error_m=lateral,route_distance_m=distance,heading_error_rad=float(wrap(yaw-p[2])),
                timestamp_s=request.scene.stamp.timestamp_s)


def state_anchored_slow(request,cfg):
    request.validate(request.scene.stamp.timestamp_s)
    state=initial_state(request);route=ArcRoute(request.route)
    x,y,yaw,v0=request.scene.ego_xy_yaw_speed;length=cfg['connection_length_m'];s0=state['route_progress_s']
    def reject(reason):raise InitialStateConnectionInfeasible(reason)
    if not 0<=v0<=5.6:reject('unsupported_forward_speed')
    if state['route_distance_m']>cfg['max_lateral_error_m'] or abs(state['heading_error_rad'])>cfg['max_heading_error_rad']:reject('initial_error_outside_declared_support')
    if s0+length>route.length:reject('insufficient_forward_route')
    # Quintic Hermite vector curve, derivatives with respect to normalized u.
    # Initial curvature=0 is an explicit unobserved-state assumption, not GT.
    end=route.at(s0+length);d0=length*np.array([np.cos(yaw),np.sin(yaw)])
    d1=length*np.array([np.cos(end[2]),np.sin(end[2])])
    lo=max(0.,s0+length-.25);hi=min(route.length,s0+length+.25)
    k1=float(wrap(route.at(hi)[2]-route.at(lo)[2])/(hi-lo))
    dd1=length**2*k1*np.array([-np.sin(end[2]),np.cos(end[2])])
    coeff=np.zeros((6,2));coeff[0]=[x,y];coeff[1]=d0
    rhs=np.array([end[:2]-coeff[0]-d0,d1-d0,dd1])
    coeff[3:]=np.linalg.solve(np.array([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]]),rhs)
    u=np.linspace(0.,1.,int(np.ceil(length/cfg['geometry_step_m']))+1)
    xy=polyval(u,coeff).T
    deriv=polyval(u,coeff[1:]*np.arange(1,6)[:,None]).T
    dd=polyval(u,coeff[2:]*(np.arange(2,6)*np.arange(1,5))[:,None]).T
    norm=np.linalg.norm(deriv,axis=1)
    if np.any(norm<1e-6):reject('singular_connection')
    curvature=(deriv[:,0]*dd[:,1]-deriv[:,1]*dd[:,0])/norm**3
    if np.max(abs(curvature))>.10:reject('connection_curvature_limit')
    angles=np.unwrap(np.arctan2(deriv[:,1],deriv[:,0]));angles+=yaw-angles[0]
    route_heading=route.at(s0+u*length)[:,2]
    if np.any(abs(wrap(angles-route_heading))>np.pi/2):reject('wrong_road_direction')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    t=np.arange(61)*.1;duration=cfg['speed_transition_s'];target=cfg['target_speed_mps'];tt=np.minimum(t,duration)
    v=target+(v0-target)/2*(1+np.cos(np.pi*tt/duration));v[t>=duration]=target
    distance=target*t+(v0-target)/2*(tt+duration/np.pi*np.sin(np.pi*tt/duration))
    q=np.minimum(distance,arc[-1]);points=np.c_[np.interp(q,arc,xy[:,0]),np.interp(q,arc,xy[:,1]),np.interp(q,arc,angles),v]
    tail=distance>arc[-1]
    if np.any(tail):
        try:points[tail,:3]=route.at(s0+length+distance[tail]-arc[-1])
        except ValueError:reject('six_second_route_end')
        points[tail,2]+=2*np.pi*round((angles[-1]-end[2])/(2*np.pi))
    # Endpoint formulas already anchor the curve; this verifies, not overwrites, t0.
    if not np.allclose(points[0],request.scene.ego_xy_yaw_speed,rtol=0,atol=1e-12):reject('analytic_anchor_error')
    a=np.diff(v)/.1;j=np.diff(a)/.1;ds=np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)
    if a.max()>1.51 or a.min()< -1.81 or max(abs(j))>2.51:reject('speed_profile_constraints')
    if np.max(abs(ds-(v[:-1]+v[1:])*.05))>.025:reject('integration_constraints')
    if np.max(abs(wrap(np.diff(points[:,2])))/np.maximum(ds,1e-6))>.10:reject('sampled_curvature_limit')
    # Guide starts on the connection too; never audit an imaginary snapped guide.
    guide=np.c_[xy,angles]
    tail_s=np.arange(s0+length+.5,route.length,.5)
    if len(tail_s):guide=np.vstack([guide,route.at(tail_s)])
    guide=np.vstack([guide,route.at(route.length)]) if np.linalg.norm(guide[-1,:2]-route.at(route.length)[:2])>1e-6 else guide
    candidate=CandidateTrajectory(request.scene.stamp,'F_state_anchored_slow',tuple(map(tuple,points)),tuple(t),tuple(map(tuple,guide)))
    return candidate,dict(initial=state,connection_length_m=length,connection_arc_length_m=float(arc[-1]),max_connection_curvature=float(max(abs(curvature))),
        initial_acceleration_assumption=0.,initial_curvature_assumption=0.,initial_acceleration_observed=False,initial_yaw_rate_observed=False,
        connection_completed_in_horizon=bool(distance[-1]>=arc[-1]),position_endpoint_residual_m=float(np.linalg.norm(xy[-1]-end[:2])))
