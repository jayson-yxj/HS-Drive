"""F0: unchanged C normal actuation, signed-contract protection and observation.

HoldEvidence monitors actual hold/release separately; it does not choose gains.
F1 is deliberately absent until physical hold/restart prerequisites pass.
"""
import math
from simulator.m6b_motion import LowSpeedController
from simulator.m6d_targets import TargetTracker

class SignedGuard:
    def __init__(self):self.previous=None;self.negative_since=None;self.latched=None;self.events=[]
    def check(self,signed,now_s):
        try:signed.validate(now_s)
        except ValueError as exc:return str(exc)
        if self.previous is not None and (signed.frame_id<=self.previous.frame_id or signed.timestamp_s<=self.previous.timestamp_s):return 'NonmonotonicSignedState'
        if signed.v_forward_mps<-.05:
            if self.negative_since is None:self.negative_since=signed.timestamp_s
            if signed.v_forward_mps<=-.15 or signed.timestamp_s-self.negative_since>=.1-1e-6:
                if self.latched is None:self.events.append(dict(timestamp_s=signed.timestamp_s,frame_id=signed.frame_id,reason='RollbackDetected',v_forward_mps=signed.v_forward_mps,gear=signed.gear))
                self.latched='RollbackDetected'
        else:self.negative_since=None
        if signed.route_progress_rate_mps<-.15 and signed.v_forward_mps>=-.05:self.latched='ForwardRouteContractInvalid'
        self.previous=signed
        # Rollback is latched for this trial; automatic release is F1 work.
        return self.latched

class HoldEvidence:
    def __init__(self):self.mode='MOVING';self.anchor=None;self.anchor_error=None;self.stable_since=None;self.transitions=[]
    def update(self,s,position,target_remaining,hold_requested,release_requested,plan_valid,reason=None):
        previous=self.mode;position=tuple(position)
        if reason=='RollbackDetected':mode='ROLLBACK_DETECTED';self.stable_since=None
        elif reason or not plan_valid:mode='DEGRADED_STOP';self.stable_since=None
        elif release_requested:
            mode='RELEASE_REQUESTED' if previous not in ['RELEASE_REQUESTED','RESTARTING','MOVING'] else 'RESTARTING'
            if s.v_forward_mps>.10:mode='MOVING'
            self.anchor=None;self.stable_since=None
        elif hold_requested:
            if self.anchor is None:self.anchor=position;self.anchor_error=abs(target_remaining) if target_remaining is not None else None
            drift=math.dist(position,self.anchor)
            growth=0 if target_remaining is None or self.anchor_error is None else abs(target_remaining)-self.anchor_error
            valid=(s.speed_norm_mps<=.02 and s.v_forward_mps>=-.02 and drift<=.05 and growth<=.02 and (s.hand_brake_applied or s.brake_applied>0))
            if valid:
                if self.stable_since is None:self.stable_since=s.timestamp_s
            else:self.stable_since=None
            mode='HOLD_STABLE' if self.stable_since is not None and s.timestamp_s-self.stable_since>=1.-1e-6 else 'HOLD_REQUESTED'
        else:
            mode='SETTLING' if s.speed_norm_mps<.2 else 'MOVING';self.anchor=None;self.stable_since=None
        self.mode=mode
        if mode!=previous:self.transitions.append(dict(timestamp_s=s.timestamp_s,frame_id=s.frame_id,previous=previous,current=mode,target_remaining_m=target_remaining,reason=reason))
        return mode

class SignedController:
    def __init__(self):
        self.base=LowSpeedController();self.guard=SignedGuard();self.targets=TargetTracker();self.hold=HoldEvidence()
    @property
    def transitions(self):return self.base.transitions
    def command(self,trajectory,state,reason=None,dt=.05,remaining_m=None,signed=None,target=None,position=None,**kwargs):
        self.targets.update(target)
        protection=self.guard.check(signed,state.timestamp_s)
        reason=reason or protection
        # Preserve frozen C numerical normal input/output for strict F0 comparison.
        # Explicit target context remains separate and cannot be overwritten by this adapter.
        u=self.base.command(trajectory,state,reason,dt,remaining_m=remaining_m)
        releasing=bool(u.get('transition_reason')=='target_advanced' or self.hold.mode in ['RELEASE_REQUESTED','RESTARTING'] and u['mode']!='HOLD')
        mode=self.hold.update(signed,position,target.active_remaining_m,u['mode']=='HOLD',releasing,reason is None,protection or reason)
        u.update(signed_guard_reason=protection,hold_evidence_mode=mode,active_stop_target=target.active_stop_target)
        return u
