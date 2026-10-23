"""Opt-in G1: measured displacement plus negative velocity evidence.

Risk is observational and does not itself request brake. Emergency is latched.
Position/heading are raw HS snapshot values; target changes never reset anchor.
"""
from collections import deque
from dataclasses import dataclass,asdict
import math
import numpy as np

@dataclass(frozen=True)
class GuardConfig:
    window_s:float=.5
    negative_velocity_mps:float=-.03
    negative_samples:int=2
    risk_net_m:float=.003
    risk_cumulative_m:float=.005
    emergency_net_m:float=.008
    emergency_cumulative_m:float=.012
    emergency_min_net_m:float=.004
    hard_negative_mps:float=-.15
    hard_min_displacement_m:float=.002
    risk_release_s:float=.5
    risk_release_forward_m:float=.01
    max_age_s:float=.1
    max_gap_s:float=.075
    dt_s:float=.05

class DisplacementGuard:
    def __init__(self,config=GuardConfig()):
        self.config=config;self.history=deque();self.previous=None;self.anchor=None;self.axis=None;self.route_axis=None
        self.risk=False;self.last_risk_time=None;self.latched=None;self.events=[];self.last_level='clear'

    def _result(self,s,reason=None,**metrics):
        level='emergency' if self.latched else 'risk' if self.risk else 'clear'
        result=dict(level=level,risk=self.risk,emergency=self.latched is not None,reason=self.latched or reason,
            frame_id=s.frame_id,timestamp_s=s.timestamp_s,source='Privileged',**metrics)
        if level!=self.last_level:self.events.append(result.copy());self.last_level=level
        return result

    def update(self,s,position_xy,route_tangent_xy,now_s):
        cfg=self.config
        try:
            s.validate(now_s)
            if not -1e-6<=now_s-s.timestamp_s<=cfg.max_age_s+1e-6:raise ValueError('StaleSignedState')
            p=np.asarray(position_xy,float);rt=np.asarray(route_tangent_xy,float)
            if p.shape!=(2,) or rt.shape!=(2,) or not np.isfinite(np.r_[p,rt]).all() or np.linalg.norm(rt)<1e-8:raise ValueError('InvalidPositionOrRoute')
        except (ValueError,TypeError) as exc:
            self.latched=str(exc);return self._result(s)
        rt=rt/np.linalg.norm(rt);axis=np.array([math.cos(s.yaw_rad),math.sin(s.yaw_rad)])
        if self.previous is not None:
            old=self.previous;dt=s.timestamp_s-old['t'];df=s.frame_id-old['frame']
            if df<=0 or dt<=0:self.latched='NonmonotonicSnapshot';return self._result(s)
            if df!=1 or dt>cfg.max_gap_s or abs(dt-cfg.dt_s)>.005:self.latched='SnapshotGap';return self._result(s)
            delta=p-old['p'];forward=float(delta@old['axis']);along=float(delta@old['rt'])
        else:forward=along=0.
        if self.anchor is None:self.anchor=p.copy();self.axis=axis.copy();self.route_axis=rt.copy()
        self.history.append(dict(t=s.timestamp_s,back=max(0.,-forward),route_back=max(0.,-along),negative=s.v_forward_mps<cfg.negative_velocity_mps))
        while self.history and s.timestamp_s-self.history[0]['t']>cfg.window_s+1e-6:self.history.popleft()
        progress=float((p-self.anchor)@self.axis);route_progress=float((p-self.anchor)@self.route_axis)
        net=max(0.,-progress);net_route=max(0.,-route_progress)
        backward=sum(q['back'] for q in self.history);route_backward=sum(q['route_back'] for q in self.history);n=sum(q['negative'] for q in self.history)
        corroborated=n>=cfg.negative_samples
        evidence=corroborated and (net>=cfg.risk_net_m or backward>=cfg.risk_cumulative_m and net>=cfg.risk_net_m/2)
        if evidence:self.risk=True;self.last_risk_time=s.timestamp_s
        if (corroborated and (net>=cfg.emergency_net_m or backward>=cfg.emergency_cumulative_m and net>=cfg.emergency_min_net_m)) or (s.v_forward_mps<=cfg.hard_negative_mps and net>=cfg.hard_min_displacement_m):
            self.latched=self.latched or 'RollbackDisplacementConfirmed'
        # Body-forward and route-forward are different. Route retreat alone with
        # positive body speed signals route mismatch, not proof of body rollback.
        if net_route>=.03 and s.route_progress_rate_mps<-.1 and s.v_forward_mps>=0:self.latched=self.latched or 'ForwardRouteContractInvalid'
        if self.risk and not evidence and not self.latched and self.last_risk_time is not None and s.timestamp_s-self.last_risk_time>=cfg.risk_release_s and progress>=cfg.risk_release_forward_m:self.risk=False
        # Only move anchor on measured forward progress with no active risk.
        # Goal advancing, changing gear or a sign flip cannot reset accumulated risk.
        if not self.risk and not self.latched and progress>cfg.risk_release_forward_m:
            self.anchor=p.copy();self.axis=axis.copy();self.route_axis=rt.copy()
        self.previous=dict(t=s.timestamp_s,frame=s.frame_id,p=p.copy(),axis=axis,rt=rt)
        return self._result(s,'BackwardRisk' if self.risk else None,net_backward_m=net,window_backward_m=backward,route_net_backward_m=net_route,route_window_backward_m=route_backward,negative_samples=n)
