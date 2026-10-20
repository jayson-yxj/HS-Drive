"""M4.6 time-parametrized oracle trajectories with a shared clearance policy."""
from dataclasses import dataclass
import math
import numpy as np
from simulator.oracle_driver import OracleState,Trajectory,footprint,polygons_intersect,wrap


def cross_track(points,route):
    start=route[:-1,:2];delta=np.diff(route[:,:2],axis=0);offset=points[:,None,:]-start[None]
    t=np.clip((offset*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-10),0,1)
    return np.linalg.norm(offset-t[:,:,None]*delta,axis=-1).min(-1)


@dataclass(frozen=True)
class MotionPolicy:
    standstill_gap: float=3.0
    footprint_margin: float=.2
    lateral_margin: float=.15
    comfortable_decel: float=1.8
    acceleration: float=1.5
    jerk: float=2.5
    emergency_decel: float=5.
    response_time: float=.25
    dt: float=.1
    horizon: int=60
    safety_step: float=.1


def route_projection(route,xy):
    r=np.asarray(route,float);delta=np.diff(r[:,:2],axis=0);length=np.linalg.norm(delta,axis=1)
    if len(r)<2 or (length<1e-8).any():raise ValueError('Route must have distinct consecutive points')
    fraction=np.clip(((np.asarray(xy)-r[:-1,:2])*delta).sum(-1)/length**2,0,1)
    points=r[:-1,:2]+fraction[:,None]*delta;index=int(np.linalg.norm(points-xy,axis=1).argmin())
    arc=np.r_[0,np.cumsum(length)]
    return float(arc[index]+fraction[index]*length[index]),arc,index,float(np.linalg.norm(points[index]-xy))


def clearance(route,state,obstacles,half_length=2.4,half_width=1.,policy=MotionPolicy()):
    """Route arc bumper gap; oriented footprints project onto local tangent.

    Non-route actors still enter Safety; crossing prediction is handled there.
    """
    s,arc,index,_=route_projection(route,[state.x,state.y]);r=np.asarray(route);candidates=[]
    for obj in obstacles:
        so,_,oi,lateral=route_projection(r,obj['xy']);yaw=float(r[oi,2]);tangent=np.array([np.cos(yaw),np.sin(yaw)])
        delta=wrap(obj['yaw']-yaw);long_extent=abs(np.cos(delta))*obj['half_length']+abs(np.sin(delta))*obj['half_width']
        lat_extent=abs(np.sin(delta))*obj['half_length']+abs(np.cos(delta))*obj['half_width']
        if so>s and lateral<=half_width+lat_extent+policy.lateral_margin:
            gap=so-s-half_length-long_extent;speed=max(0,float(np.dot(obj.get('velocity',[0.,0.]),tangent)))
            candidates.append(dict(gap_m=float(gap),speed_mps=speed,actor_id=obj.get('actor_id'),stop_arc_m=float(so-half_length-long_extent-policy.standstill_gap)))
    return min(candidates,key=lambda c:c['gap_m']) if candidates else None


def validate_kinematics(trajectory,state,policy=MotionPolicy()):
    p=np.asarray(trajectory.points)
    if p.ndim!=2 or p.shape[1]!=4 or len(p)<2 or not np.isfinite(p).all() or not np.isfinite(trajectory.dt_s) or trajectory.dt_s<=0:return 'invalid_trajectory'
    if np.any(p[:,3]<-1e-6):return 'negative_velocity'
    # Straight segment chord <= path length for curves; metadata is not needed
    # for this fail-closed upper bound. Start pose must be the current origin.
    length=np.linalg.norm(np.diff(np.vstack(([0.,0.],p[:,:2])),axis=0),axis=1)
    velocity=np.r_[state.speed_mps,p[:,3]];integral=(velocity[:-1]+velocity[1:])*trajectory.dt_s/2
    if np.any(length>integral+.025):return 'position_velocity_inconsistent'
    acceleration=np.diff(velocity)/trajectory.dt_s
    if acceleration.max()>policy.acceleration+.01 or acceleration.min()<-policy.comfortable_decel-.01:return 'acceleration_limit'
    return None


class MotionPlanner:
    def __init__(self,policy=MotionPolicy()):
        self.policy=policy;self.previous_acceleration=0.;self.timestamp=None

    def plan(self,route,state,obstacles=(),*,max_speed=4.,half_length=2.4,half_width=1.):
        p=self.policy;r=np.asarray(route,float);s0,arc,index,lateral=route_projection(r,[state.x,state.y])
        ahead=clearance(r,state,obstacles,half_length,half_width,p)
        # Moving lead: conservative stationary stop target at current rear bumper.
        # Replanning naturally advances this target; same gap policy used by Safety.
        available=arc[-1]-s0-1.
        if ahead is not None:available=min(available,ahead['gap_m']-p.standstill_gap)
        headings=np.unwrap(r[:,2]);curvature=np.abs(np.diff(headings))/np.maximum(np.diff(arc),.01)
        local_curv=float(curvature[index:min(len(curvature),index+15)].max(initial=0))
        cruise=min(max_speed,math.sqrt(1.5/max(local_curv,1e-5)))
        dt=p.dt;v=max(0,state.speed_mps);distance=0.;acc=self.previous_acceleration;points=[];speeds=[];accs=[];distances=[]
        if self.timestamp is None or not 0<state.timestamp_s-self.timestamp<=.25:acc=0.
        for k in range(p.horizon):
            remaining=max(0,available-distance)
            # Stopping-distance bound includes a smooth acceleration-release buffer.
            buffer=v*(max(acc,0)/p.jerk + p.comfortable_decel/(2*p.jerk)+.25)
            vcap=min(cruise,math.sqrt(max(0,2*p.comfortable_decel*(remaining-buffer))))
            if remaining<12.:
                # Low-speed coast envelope: this CARLA vehicle's service brake
                # has abrupt low-speed stiction. Reserve distance to coast before
                # applying stationary hold; still retain independent hard safety.
                vcap=min(vcap,math.sqrt(max(0,2*.20*(remaining-.5))))
            desired=float(np.clip((vcap-v)/.7,-p.comfortable_decel,p.acceleration))
            if remaining<.1 and v<.15:desired=-min(v/dt,p.comfortable_decel)
            next_acc=float(np.clip(desired,acc-p.jerk*dt,acc+p.jerk*dt))
            # Keep enough velocity to release negative acceleration at bounded
            # jerk; prevents clipping velocity to zero with a discontinuous a.
            release_bound=p.jerk*dt-math.sqrt((p.jerk*dt)**2+2*p.jerk*v)
            next_acc=max(next_acc,release_bound)
            next_v=max(0,v+next_acc*dt)
            effective_acc=(next_v-v)/dt
            step=(v+next_v)*dt/2
            distance+=step;distances.append(distance);speeds.append(next_v);accs.append(effective_acc)
            v=next_v;acc=effective_acc
        query=np.minimum(s0+np.array(distances),arc[-1])
        xy=np.stack([np.interp(query,arc,r[:,i]) for i in (0,1)],1);yaw=np.interp(query,arc,headings)
        # Start path at current actual location with a gradual lateral correction.
        closest=np.array([np.interp(s0,arc,r[:,i]) for i in (0,1)])
        offset=np.array([state.x,state.y])-closest
        xy+=np.exp(-np.array(distances)/max(10.,state.speed_mps*2))[:,None]*offset
        c,si=np.cos(state.yaw),np.sin(state.yaw);local=(xy-[state.x,state.y])@np.array([[c,-si],[si,c]])
        points=np.c_[local,wrap(yaw-state.yaw),speeds]
        trajectory=Trajectory(points,state.timestamp_s,dt,source='carla_oracle_m46')
        # Advance planner acceleration only one controller tick, not a full horizon.
        control_dt=.05 if self.timestamp is None else np.clip(state.timestamp_s-self.timestamp,.001,.1)
        self.previous_acceleration=float(np.clip(accs[0],self.previous_acceleration-p.jerk*control_dt,self.previous_acceleration+p.jerk*control_dt))
        self.timestamp=state.timestamp_s
        return trajectory,dict(route_index=index,route_arc=s0,remaining_m=float(available),obstacle=ahead,
            acceleration=np.array(accs),path_distance=np.array(distances),kinematic_failure=validate_kinematics(trajectory,state,p))


def safety_check(trajectory,state,obstacles,*,half_length=2.4,half_width=1.,policy=MotionPolicy(),road_predicate=None):
    if trajectory is None:return 'no_trajectory'
    if not np.isfinite(trajectory.timestamp_s):return 'invalid_trajectory'
    age=state.timestamp_s-trajectory.timestamp_s
    if age<0 or age>.2:return 'trajectory_timeout'
    bad=validate_kinematics(trajectory,state,policy)
    if bad:return bad
    p=trajectory.points
    if (p[:,3]>5.6).any():return 'speed_limit'
    c,s=np.cos(state.yaw),np.sin(state.yaw);xy=p[:,:2]@np.array([[c,s],[-s,c]])+[state.x,state.y]
    last=np.array([state.x,state.y,state.yaw]);elapsed=0
    for point,position in zip(p,xy):
        cur=np.r_[position,state.yaw+point[2]]
        # Limit spatial and temporal step, including stationary ego / crossing actor.
        steps=max(1,int(np.ceil(np.linalg.norm(cur[:2]-last[:2])/policy.safety_step)),int(np.ceil(trajectory.dt_s/.05)))
        for f in np.linspace(0,1,steps+1):
            t=elapsed+f*trajectory.dt_s;pos=(1-f)*last[:2]+f*cur[:2];yaw=last[2]+f*wrap(cur[2]-last[2])
            body=footprint(*pos,yaw,half_length+policy.footprint_margin,half_width+policy.lateral_margin)
            if road_predicate and not all(road_predicate(x,y) for x,y in body):return 'road_boundary'
            for o in obstacles:
                other=footprint(*(np.array(o['xy'])+t*np.array(o.get('velocity',[0.,0.]))),o['yaw'],o['half_length'],o['half_width'])
                if polygons_intersect(body,other):return 'obstacle_footprint'
        last=cur;elapsed+=trajectory.dt_s
    return None


class MotionController:
    """Acceleration feedforward plus velocity feedback, bounded actuator rates."""
    def __init__(self,policy=MotionPolicy()):
        self.policy=policy;self.last_time=None;self.steer=0.;self.effort=0.;self.integral=0.;self.filtered_speed=None;self.holding=False

    def command(self,trajectory,state,reason=None,dt=.05,clearance_m=None):
        if self.last_time is not None and not 0<state.timestamp_s-self.last_time<=.25:reason=reason or 'controller_timeout'
        self.last_time=state.timestamp_s
        if reason or trajectory is None:
            self.effort=-1.;self.integral=0.;return dict(steer=self.steer,throttle=0.,brake=1.,reason=reason or 'no_trajectory')
        p=trajectory.points
        look=max(3.,state.speed_mps*.8);index=int(np.argmin(abs(np.linalg.norm(p[:,:2],axis=1)-look)));x,y=p[index,:2]
        steer=-np.clip(math.atan2(5.6*y,max(x*x+y*y,.1))/.6,-1,1);self.steer=float(np.clip(steer,self.steer-1.5*dt,self.steer+1.5*dt))
        acceleration=(p[0,3]-state.speed_mps)/trajectory.dt_s
        self.filtered_speed=state.speed_mps if self.filtered_speed is None else .3*state.speed_mps+.7*self.filtered_speed
        target=float(p[min(4,len(p)-1),3]);error=target-self.filtered_speed
        self.integral=float(np.clip(self.integral+error*dt,-.5,.5))
        demand=acceleration+.5*error+.05*self.integral
        # Empirical feedforward scaling plus the velocity error above. These
        # actuator limits do not guarantee physical acceleration/jerk limits.
        effort=demand/3. if demand>=0 else demand/8.
        at_stop=clearance_m is not None and clearance_m<=self.policy.standstill_gap+.25
        if at_stop and state.speed_mps<1.5:self.holding=True
        elif state.speed_mps<.3 and np.linalg.norm(p[-1,:2])<.8 and p[-1,3]<.08:self.holding=True
        if not at_stop and np.linalg.norm(p[-1,:2])>1.2:self.holding=False
        if self.holding:effort=-.2;self.integral=0.
        elif state.speed_mps<1.5:effort=max(0,effort)
        rate=.4 if self.holding else .8
        self.effort=float(np.clip(effort,self.effort-rate*dt,self.effort+rate*dt));self.effort=float(np.clip(self.effort,-1,.65))
        return dict(steer=self.steer,throttle=max(0.,self.effort),brake=max(0.,-self.effort),reason=None)
