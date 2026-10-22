"""Opt-in explicit stop pose and low-speed control. No learned perception inputs."""
from dataclasses import dataclass
import math
import numpy as np
from simulator.motion_oracle import MotionPlanner, route_projection, clearance
from simulator.oracle_driver import wrap
from simulator.m48_motion import RouteController


@dataclass(frozen=True)
class StopGoal:
    target_position_xy: tuple
    target_yaw: float
    position_tolerance_m: float = .5
    heading_tolerance_rad: float = math.radians(5)
    terminal_speed_mps: float = .08
    source: str = 'Privileged'
    timestamp: float = 0.
    coordinate_frame: str = 'hs_world_x_y_left_z_up_m'

    def validate(self):
        if self.source != 'Privileged' or self.coordinate_frame != 'hs_world_x_y_left_z_up_m':
            raise ValueError('Privileged HS world stop pose required')
        if len(self.target_position_xy)!=2 or not np.isfinite([*self.target_position_xy,self.target_yaw,self.timestamp,self.position_tolerance_m,self.heading_tolerance_rad,self.terminal_speed_mps]).all():
            raise ValueError('invalid stop pose')
        if min(self.position_tolerance_m,self.heading_tolerance_rad,self.terminal_speed_mps)<=0:
            raise ValueError('invalid tolerances')


def goal_route(route, goal):
    """Explicit adapter to frozen planner's 1m navigation reserve.

    The auxiliary point is NOT a new stop goal. It cancels the implicit planner
    reserve so integration ends at the audited goal; obstacle margins unchanged.
    """
    goal.validate();r=np.asarray(route,float)
    if np.linalg.norm(r[-1,:2]-goal.target_position_xy)>1e-3:
        raise ValueError('route must end at audited stop pose')
    tangent=np.array([math.cos(goal.target_yaw), math.sin(goal.target_yaw)])
    return np.vstack((r,np.r_[r[-1,:2]+tangent,goal.target_yaw]))


def remaining_distance(route,state,obstacles,hl,hw):
    s,arc,_,_=route_projection(route,[state.x,state.y])
    remaining=arc[-1]-s
    ahead=clearance(route,state,obstacles,hl,hw)
    if ahead:remaining=min(remaining,ahead['gap_m']-3.0)
    # Signed projection beyond goal prevents falsely claiming positive distance.
    end=np.asarray(route)[-1]
    overshoot=float(np.dot(np.array([state.x,state.y])-end[:2],[np.cos(end[2]),np.sin(end[2])]))
    if overshoot>0:remaining=min(remaining,-overshoot)
    return float(remaining)


class LowSpeedController(RouteController):
    """Measured coast/feedforward candidate; service brake only after settling.

    Emergency remains immediate brake=1 and bypasses actuator rate limits.
    HOLD is latched only near target and measured speed<=0.02; moving targets
    can release it. Remaining distance is current actual projection, not age*v.
    """
    def __init__(self):
        super().__init__();self.mode='APPROACH';self.previous_throttle=0.;self.transitions=[]

    def command(self,trajectory,state,reason=None,dt=.05,remaining_m=None,**kw):
        if self.last_time is not None and not 0<state.timestamp_s-self.last_time<=.25:
            reason=reason or 'controller_timeout'
        self.last_time=state.timestamp_s
        old=self.mode;v=state.speed_mps;remaining=1000 if remaining_m is None else remaining_m
        why=None
        if reason or trajectory is None:
            self.mode='EMERGENCY';self.previous_throttle=0.;why=reason or 'no_trajectory'
            u=dict(throttle=0.,brake=1.,steer=self.steer,reason=why)
        else:
            p=trajectory.points;look=max(2.,v*.65);idx=int(np.argmin(abs(np.linalg.norm(p[:,:2],axis=1)-look)))
            x,y=p[idx,:2];wanted=-np.clip(math.atan2(5.6*y,max(x*x+y*y,.1))/.6,-1,1)
            self.steer=float(np.clip(wanted,self.steer-1.5*dt,self.steer+1.5*dt))
            if self.mode=='EMERGENCY':self.mode='APPROACH';why='valid_plan_recovered'
            if self.mode=='HOLD' and remaining>1.0:self.mode='CREEP';why='target_advanced'
            if self.mode!='HOLD':
                if remaining<=.35 and v<=.02:self.mode='HOLD';why='near_goal_and_measured_settled'
                elif remaining <= max(.35,self.coast_distance(v)+.12):self.mode='SETTLE';why='coast_distance_reaches_target'
                elif remaining<3:self.mode='CREEP';why='terminal_approach'
                else:self.mode='APPROACH';why='tracking'
            target=float(p[min(4,len(p)-1),3])
            # Never command creep beyond the safe trajectory's speed envelope.
            if self.mode=='CREEP':target=min(target,.45)
            demand=float(np.clip((target-v)*1.2,-1.2,.6))
            bias=-.5750421118-.2823521898*v
            throttle=float(np.clip((demand-bias)/7.162067586,0,.25))
            if target<v-.04:throttle=0.
            if self.mode in ('SETTLE','HOLD'):throttle=0.
            # Downward throttle release is immediate; residual effort must not
            # keep pushing during settle. Upward is slew-limited.
            throttle=min(throttle,self.previous_throttle+.15*dt)
            brake=.12 if self.mode=='HOLD' else 0.
            # Comfortable service brake above nonlinear low-speed band only.
            if v>2.2 and demand<bias:brake=min(.12,(bias-demand)/5.219885642)
            self.previous_throttle=throttle
            u=dict(throttle=throttle,brake=brake,steer=self.steer,reason=None,target_speed=target,requested_acceleration=demand)
        if old!=self.mode:self.transitions.append(dict(timestamp=state.timestamp_s,old=old,new=self.mode,reason=why,speed=v,remaining_m=remaining))
        u.update(mode=self.mode,transition_reason=why if old!=self.mode else None)
        return u

    @staticmethod
    def coast_distance(v):
        # M47 high-speed coast model, lower-speed bounded approximation; new
        # probe audit is reported separately and does not claim exact physics.
        a=.20;b=.40;v=max(0,v)
        return v/b-a/(b*b)*math.log1p(b*v/a)
