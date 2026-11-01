"""Pure J13 safety decisions. No model, CARLA, environment truth or actuation."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PlanStamp:
    sequence: str
    input_frame: int
    input_sim_s: float
    input_wall_s: float
    inference_start_wall_s: float
    output_wall_s: float
    model_sha256: str


def freshness(stamp, sequence, frame, sim_s, wall_s, max_sim=.3, max_wall=.3):
    """Age starts at capture; completion never extends validity."""
    values=[stamp.input_sim_s,stamp.input_wall_s,stamp.inference_start_wall_s,
            stamp.output_wall_s,sim_s,wall_s]
    if not all(math.isfinite(v) for v in values):return 'InvalidPlanClock'
    if stamp.sequence!=sequence:return 'CrossSequencePlan'
    if stamp.input_frame>frame or stamp.input_sim_s>sim_s+1e-6:return 'FuturePlan'
    if not stamp.input_wall_s<=stamp.inference_start_wall_s<=stamp.output_wall_s<=wall_s+1e-6:return 'InvalidPlanClock'
    if sim_s-stamp.input_sim_s>max_sim+1e-6:return 'PlanSimulationExpired'
    if wall_s-stamp.input_wall_s>max_wall+1e-6:return 'PlanWallExpired'
    return None


def validate_trajectory(points, initial_speed, limits):
    """Check native six t+.5..3s samples; t0 anchor is diagnostics only.

    Window-average native speed is not instantaneous ego speed. Derivative
    diagnostics explicitly use adjacent window averages and segment headings.
    The unmodified returned trajectory never receives an invented t0/tail.
    """
    p=np.asarray(points,dtype=float)
    if p.shape!=(limits['points'],4):return 'TrajectoryShape',{}
    if not np.isfinite(p).all() or not math.isfinite(initial_speed):return 'NonfiniteTrajectory',{}
    dt=limits['dt_s'];xy=np.vstack([np.zeros((1,2)),p[:,:2]])
    segments=np.diff(xy,axis=0);length=np.linalg.norm(segments,axis=1)
    acceleration=np.diff(np.r_[initial_speed,p[:,3]])/dt
    angle=np.arctan2(segments[:,1],segments[:,0])
    delta=(np.diff(angle)+np.pi)%(2*np.pi)-np.pi
    moving=(length[:-1]>.1)&(length[1:]>.1)
    curvature=np.abs(delta)/np.maximum((length[:-1]+length[1:])/2,.1)
    curvature=np.where(moving,curvature,0.)
    lateral_accel=curvature*np.maximum(p[:-1,3],p[1:,3])**2
    discrepancy=np.abs(length/dt-p[:,3])
    diagnostics=dict(first_point_offset_s=.5,horizon_s=3.,initial_anchor=[0.,0.],
        segment_lengths_m=length.tolist(),window_speed_change_mps2=acceleration.tolist(),
        segment_curvature_per_m=curvature.tolist(),estimated_lateral_accel_mps2=lateral_accel.tolist(),
        speed_position_disagreement_mps=discrepancy.tolist(),
        derivative_semantics='Finite differences of window-average speeds and chord headings; not exact instantaneous acceleration/curvature')
    checks=[(np.any(p[:,3]<0) or np.any(p[:,3]>limits['native_speed_max_mps']),'TrajectorySpeed'),
        (np.any(length>limits['max_segment_length_m']),'TrajectoryJump'),
        (np.any(acceleration>limits['max_acceleration_mps2']) or np.any(acceleration< -limits['max_deceleration_mps2']),'TrajectoryAcceleration'),
        (np.any(curvature>limits['max_curvature_per_m']),'TrajectoryCurvature'),
        (np.any(lateral_accel>limits['max_lateral_acceleration_mps2']),'TrajectoryLateralAcceleration'),
        (np.any(discrepancy>limits['max_speed_position_disagreement_mps']),'TrajectorySpeedPositionMismatch')]
    return next((name for bad,name in checks if bad),None),diagnostics


def transport_native_path(path, input_carla_pose, current_carla_pose):
    """Only change reference frame; keep all 15 spatial samples and speeds."""
    p=np.asarray(path,float);a=np.asarray(input_carla_pose,float);b=np.asarray(current_carla_pose,float)
    if p.shape!=(15,2) or a.shape!=(4,4) or b.shape!=(4,4) or not all(np.isfinite(x).all() for x in (p,a,b)):
        raise ValueError('InvalidPathTransport')
    carla=np.vstack([p[:,1],p[:,0],np.zeros(15),np.ones(15)])
    current=np.linalg.inv(b)@a@carla
    return current[[1,0]].T


def sanitize_control(raw):
    """Never hide a controller error by silently clipping it to a legal value."""
    try: values={k:float(raw[k]) for k in ('steer','throttle','brake')}
    except (KeyError,TypeError,ValueError):return None,'InvalidControl'
    if not all(math.isfinite(x) for x in values.values()):return None,'NonfiniteControl'
    if not -1<=values['steer']<=1 or not all(0<=values[k]<=1 for k in ('throttle','brake')):
        return None,'ControlOutOfRange'
    if values['throttle']>0 and values['brake']>0:return None,'ConflictingControl'
    return values,None


class SafetyGate:
    """Latched stop; only a new run may rearm after a fault."""
    def __init__(self,config):self.config=config;self.reason=None;self.events=[]
    def stop(self,reason,sim_s,wall_s):
        if self.reason is None:
            self.reason=reason;self.events.append(dict(reason=reason,sim_s=sim_s,wall_s=wall_s))
        return dict(steer=0.,throttle=0.,brake=1.)
    def command(self,raw,*,speed,sim_s,wall_s,reason=None):
        if self.reason or reason:return self.stop(self.reason or reason,sim_s,wall_s)
        if not math.isfinite(speed):return self.stop('InvalidEgoSpeed',sim_s,wall_s)
        values,error=sanitize_control(raw)
        if error:return self.stop(error,sim_s,wall_s)
        if speed<-.2:return self.stop('ReverseMotion',sim_s,wall_s)
        if speed>self.config['max_speed_mps']:return self.stop('SpeedLimitExceeded',sim_s,wall_s)
        if speed>=self.config['speed_cap_brake_threshold_mps']:
            return dict(steer=values['steer'],throttle=0.,brake=max(values['brake'],self.config['speed_cap_brake']))
        return values


def watchdog_reason(heartbeat,now,timeout=.3,manual=False):
    if manual:return 'ManualEmergencyStop'
    if heartbeat is None:return 'MissingHeartbeat'
    if not math.isfinite(heartbeat['wall_s']) or now<heartbeat['wall_s']-1e-6:return 'InvalidHeartbeat'
    if now-heartbeat['wall_s']>timeout:return 'ControlWatchdogExpired'
    if now>heartbeat['command_valid_until_wall_s']+1e-6:return 'ControlCommandExpired'
    return None
