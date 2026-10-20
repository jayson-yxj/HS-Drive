"""Privileged low-speed planning/control contracts. Never a learned input adapter."""
from dataclasses import dataclass
import math
import numpy as np


def wrap(angle):return (angle+np.pi)%(2*np.pi)-np.pi


def footprint(x,y,yaw,half_length,half_width):
    points=np.array([[half_length,half_width],[half_length,-half_width],[-half_length,-half_width],[-half_length,half_width]])
    c,s=np.cos(yaw),np.sin(yaw);return points@np.array([[c,s],[-s,c]])+np.array([x,y])


def polygons_intersect(a,b):
    for poly in (a,b):
        edges=np.roll(poly,-1,axis=0)-poly
        for edge in edges:
            normal=np.array([-edge[1],edge[0]]);aa=a@normal;bb=b@normal
            if aa.max()<bb.min() or bb.max()<aa.min():return False
    return True


@dataclass
class OracleState:
    x: float
    y: float
    yaw: float
    speed_mps: float
    timestamp_s: float


@dataclass
class Trajectory:
    points: np.ndarray # [T,4] HS ego x forward/y left/yaw left/speed mps
    timestamp_s: float
    dt_s: float=.5
    source: str='carla_oracle'


def local_trajectory(route,state,*,max_speed=4.,horizon=8,dt=.5,stop_distance=None):
    """Route world is HS right-handed; fixed dt trajectory with curvature braking."""
    r=np.asarray(route,float);pos=np.array([state.x,state.y]);distance=np.linalg.norm(r[:,:2]-pos,axis=1);nearest=int(distance.argmin())
    rr=r[nearest:];segments=np.linalg.norm(np.diff(rr[:,:2],axis=0),axis=1);arc=np.r_[0,np.cumsum(segments)]
    if len(rr)<2:return Trajectory(np.tile([0,0,0,0],(horizon,1)),state.timestamp_s,dt),nearest
    yaw=np.unwrap(rr[:,2]);curvature=np.abs(np.diff(yaw))/np.maximum(segments,.1)
    curve=float(curvature[:min(12,len(curvature))].max(initial=0));speed=min(max_speed,math.sqrt(1.5/max(curve,1e-4)),math.sqrt(max(0,2*1.5*(arc[-1]-1.))))
    if stop_distance is not None:speed=min(speed,math.sqrt(max(0,2*1.8*(stop_distance-4.))))
    s=np.minimum(np.arange(1,horizon+1)*dt*max(speed,.5),arc[-1]);xy=np.stack([np.interp(s,arc,rr[:,j]) for j in (0,1)],1);heading=np.interp(s,arc,yaw)
    c,si=np.cos(state.yaw),np.sin(state.yaw);local=(xy-pos)@np.array([[c,-si],[si,c]])
    points=np.c_[local,wrap(heading-state.yaw),np.full(horizon,speed)]
    return Trajectory(points,state.timestamp_s,dt),nearest


def safety_check(trajectory,state,obstacles,*,half_length=2.4,half_width=1.,max_speed=5.6,max_age=.2,road_predicate=None):
    if trajectory is None:return 'no_trajectory'
    p=np.asarray(trajectory.points)
    if p.ndim!=2 or p.shape[1]!=4 or len(p)<2 or not np.isfinite(p).all() or not np.isfinite(trajectory.timestamp_s) or not np.isfinite(trajectory.dt_s) or trajectory.dt_s<=0:return 'invalid_trajectory'
    age=state.timestamp_s-trajectory.timestamp_s
    if age<0 or age>max_age:return 'trajectory_timeout'
    if np.any(p[:,3]<0) or np.any(p[:,3]>max_speed):return 'speed_limit'
    if np.any(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)>max_speed*trajectory.dt_s+1):return 'trajectory_jump'
    c,s=np.cos(state.yaw),np.sin(state.yaw)
    world=p[:,:2]@np.array([[c,s],[-s,c]])+np.array([state.x,state.y])
    # Include current pose and densify at <=0.25m: discrete footprints, NOT exact swept volume.
    last=np.array([state.x,state.y,state.yaw]);elapsed=0.
    for i,(xy,point) in enumerate(zip(world,p)):
        cur=np.r_[xy,state.yaw+point[2]];steps=max(1,int(np.ceil(np.linalg.norm(cur[:2]-last[:2])/.25)))
        for f in np.linspace(0,1,steps+1):
            xy2=last[:2]*(1-f)+cur[:2]*f;yaw=last[2]+f*wrap(cur[2]-last[2]);body=footprint(*xy2,yaw,half_length+.2,half_width+.1)
            if road_predicate is not None and not all(road_predicate(x,y) for x,y in body):return 'road_boundary'
            time_s=elapsed+f*trajectory.dt_s
            for obstacle in obstacles:
                center=np.asarray(obstacle['xy'])+time_s*np.asarray(obstacle.get('velocity',[0.,0.]));other=footprint(*center,obstacle['yaw'],obstacle['half_length'],obstacle['half_width'])
                if polygons_intersect(body,other):return 'obstacle_footprint'
        last=cur;elapsed+=trajectory.dt_s
    return None


class PurePursuitPID:
    def __init__(self,wheelbase=2.8,max_steer_rad=.6):
        self.wheelbase=wheelbase;self.max_steer_rad=max_steer_rad;self.integral=0.;self.last_error=0.;self.last_steer=0.;self.last_throttle=0.;self.last_time=None

    def command(self,trajectory,state,reason=None,dt=.05):
        if self.last_time is not None and state.timestamp_s-self.last_time>.25:reason=reason or 'controller_timeout'
        self.last_time=state.timestamp_s
        if reason or trajectory is None:
            self.integral=0.;self.last_throttle=0.;return {'steer':self.last_steer,'throttle':0.,'brake':1.,'reason':reason or 'no_trajectory'}
        points=trajectory.points;lookahead=max(3.,state.speed_mps*.8);idx=int(np.argmin(np.abs(np.linalg.norm(points[:,:2],axis=1)-lookahead)));x,y=points[idx,:2]
        angle=math.atan2(2*self.wheelbase*y,max(x*x+y*y,.1));hs_steer=np.clip(angle/self.max_steer_rad,-1,1)
        # CARLA positive steer points right, HS positive yaw points left.
        steer=float(np.clip(-hs_steer,self.last_steer-1.5*dt,self.last_steer+1.5*dt));self.last_steer=steer
        target=float(points[0,3]);error=target-state.speed_mps;self.integral=float(np.clip(self.integral+error*dt,-2,2));derivative=(error-self.last_error)/dt;self.last_error=error
        effort=.45*error+.08*self.integral+.02*derivative
        throttle=float(np.clip(effort,0,.65));throttle=float(min(throttle,self.last_throttle+dt));self.last_throttle=throttle
        brake=float(np.clip(-effort,0,1))
        if target<.15:throttle=0.;brake=max(brake,.6)
        if brake>0:throttle=0.
        return {'steer':steer,'throttle':throttle,'brake':brake,'reason':None}
