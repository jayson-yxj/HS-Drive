"""Signed XY observation alongside, never replacing, nonnegative OracleState.

HS world x forward, y left, z up. Raw CARLA velocity has y right.
Applied control is separately queried at the snapshot boundary, not supplied by
CARLA ActorSnapshot. No direction inferred from gear or commanded reverse.
"""
from dataclasses import dataclass
import math
import numpy as np
from simulator.motion_oracle import route_projection

FRAME='hs_world_x_y_left_z_up_m'

@dataclass(frozen=True)
class SignedMotionState:
    v_forward_mps: float
    v_lateral_mps: float
    speed_norm_mps: float
    route_progress_rate_mps: float
    signed_goal_distance_m: float
    yaw_rad: float
    gear: int
    reverse_commanded: bool
    hand_brake_applied: bool
    brake_applied: float
    slope: float
    timestamp_s: float
    frame_id: int
    coordinate_frame: str=FRAME
    source: str='Privileged'

    def validate(self,now_s=None):
        values=[self.v_forward_mps,self.v_lateral_mps,self.speed_norm_mps,self.route_progress_rate_mps,self.signed_goal_distance_m,self.yaw_rad,self.brake_applied,self.slope,self.timestamp_s]
        if not all(math.isfinite(x) for x in values):raise ValueError('NonfiniteSignedState')
        if self.coordinate_frame!=FRAME or self.source!='Privileged':raise ValueError('WrongSourceOrFrame')
        if self.speed_norm_mps<0 or not 0<=self.brake_applied<=1 or self.frame_id<0:raise ValueError('InvalidSignedState')
        if math.hypot(self.v_forward_mps,self.v_lateral_mps)>self.speed_norm_mps+1e-5:raise ValueError('InconsistentVelocity')
        if now_s is not None and not -1e-6<=now_s-self.timestamp_s<=.10+1e-6:raise ValueError('StaleSignedState')


def signed_from_measurement(measurement,route,goal_signed_distance_m,slope):
    """Same immutable snapshot measurement as OracleState; no extra actor pose RPC."""
    vx,vy,vz=measurement['velocity'];velocity=np.array([vx,-vy]);yaw=-math.radians(measurement['rotation'][2])
    forward=np.array([math.cos(yaw),math.sin(yaw)]);left=np.array([-forward[1],forward[0]])
    r=np.asarray(route,dtype=float)
    if r.ndim!=2 or len(r)<2 or not np.isfinite(r).all():raise ValueError('InvalidRoute')
    position=np.array(measurement['position'][:2])*[1,-1]
    _,_,index,_=route_projection(r,position)
    tangent=r[min(index+1,len(r)-1),:2]-r[index,:2]
    if np.linalg.norm(tangent)<1e-8:raise ValueError('DegenerateRouteTangent')
    tangent/=np.linalg.norm(tangent)
    u=measurement['applied']
    state=SignedMotionState(float(velocity@forward),float(velocity@left),float(np.linalg.norm([vx,vy,vz])),float(velocity@tangent),float(goal_signed_distance_m),yaw,
        int(u['gear']),bool(u['reverse']),bool(u['hand_brake']),float(u['brake']),float(slope),float(measurement['t']),int(measurement['frame']))
    state.validate();return state
