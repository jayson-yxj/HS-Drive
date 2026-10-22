"""M48 route controller candidate, safety and physical dimensions unchanged."""
import math
import numpy as np
from simulator.motion_oracle import MotionController
from simulator.m47_motion import safety_check

class RouteController(MotionController):
    def command(self,trajectory,state,reason=None,dt=.05,**kwargs):
        before=self.steer
        result=super().command(trajectory,state,reason,dt,**kwargs)
        if reason or trajectory is None:return result
        p=trajectory.points;look=max(2.,state.speed_mps*.65)
        idx=int(np.argmin(abs(np.linalg.norm(p[:,:2],axis=1)-look)))
        x,y=p[idx,:2];desired=-np.clip(math.atan2(5.6*y,max(x*x+y*y,.1))/.6,-1,1)
        self.steer=float(np.clip(desired,before-1.5*dt,before+1.5*dt));result['steer']=self.steer
        return result
