"""5 Hz full-horizon planner / 20 Hz controller: immutable generation time."""
import numpy as np
from simulator.oracle_driver import Trajectory,wrap

class TimedPlan:
    def __init__(self,period=.2,max_age=.25):
        self.period=period;self.max_age=max_age;self.plan=None;self.state=None
    def due(self,timestamp):
        return self.plan is None or timestamp-self.plan.timestamp_s>=self.period-1e-6
    def store(self,plan,state):
        self.plan=plan;self.state=state
    def sample(self,state):
        if self.plan is None:return None,'no_trajectory'
        age=state.timestamp_s-self.plan.timestamp_s
        if not np.isfinite(age) or age< -1e-6 or age>self.max_age+1e-6:return None,'trajectory_timeout'
        p=self.plan.points;dt=self.plan.dt_s;initial=self.state
        if not np.isfinite(p).all():return None,'invalid_trajectory'
        t=np.arange(len(p)+1)*dt;query=np.arange(1,len(p)+1)*dt+max(0,age)
        if query[0]>t[-1]:return None,'trajectory_timeout'
        query=query[query<=t[-1]+1e-6]
        c,s=np.cos(initial.yaw),np.sin(initial.yaw)
        xy=p[:,:2]@np.array([[c,s],[-s,c]])+[initial.x,initial.y]
        xy=np.vstack(([initial.x,initial.y],xy));yaw=np.unwrap(np.r_[initial.yaw,initial.yaw+p[:,2]])
        world=np.stack([np.interp(query,t,xy[:,i]) for i in (0,1)],1)
        c,s=np.cos(state.yaw),np.sin(state.yaw);local=(world-[state.x,state.y])@np.array([[c,-s],[s,c]])
        points=np.c_[local,wrap(np.interp(query,t,yaw)-state.yaw),np.interp(query,t,np.r_[initial.speed_mps,p[:,3]])]
        # Timestamp marks this resampling only; the immutable self.plan timestamp
        # is authoritative for expiration and is recorded separately by caller.
        return Trajectory(points,state.timestamp_s,dt,source='m48_resampled'),None
