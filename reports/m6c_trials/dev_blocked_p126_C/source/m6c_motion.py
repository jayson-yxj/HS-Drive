"""D: frozen C actuation plus empirical stopping decisions, opt-in only."""
from simulator.m6b_motion import LowSpeedController


class EnvelopeController(LowSpeedController):
    def __init__(self,envelope):
        super().__init__();self.envelope=envelope;self.decision={};self.braking=False

    def command(self,trajectory,state,reason=None,dt=.05,remaining_m=None,geometry=None,grade=0.,**kwargs):
        available=geometry['available_clearance_to_safe_stop'];obstacle=geometry['obstacle_obb_clearance_m'] is not None
        action='brake_001' if obstacle or grade<-.005 else 'coast'
        estimate=self.envelope.estimate(state.speed_mps,grade,action)
        emergency=self.envelope.estimate(state.speed_mps,grade,'emergency')
        margin=None if estimate.distance_m is None else available-estimate.distance_m
        feasible=estimate.supported and margin>=0
        self.decision=dict(estimated_stopping_distance_m=estimate.distance_m,conservative_required_stopping_distance=estimate.distance_m,
            stopping_margin_m=margin,stopping_feasible=feasible,action=action,estimate_reason=estimate.reason,
            emergency_required_m=emergency.distance_m,emergency_feasible=emergency.supported and available>=emergency.distance_m,
            source='Privileged',timestamp=state.timestamp_s)
        # Only route-relevant obstacle is assessed for 3m reserve. Bypass boxes
        # remain in original Safety; lateral passing is not a 3m standstill task.
        if obstacle and state.speed_mps>.02 and not feasible:
            reason=reason or 'InsufficientStoppingMargin'
        # Start normal stopping before the empirical limit; this changes only
        # the stopping decision, not frozen throttle feedforward or gearing.
        trigger=estimate.distance_m is not None and available<=estimate.distance_m+max(.25,state.speed_mps*.1)
        if self.braking and available>max(1.,(estimate.distance_m or 0)+.75):self.braking=False
        if trigger:self.braking=True
        decision_remaining=remaining_m
        if self.braking and not reason:
            # Frozen C enters SETTLE without advancing/relocating StopGoal.
            decision_remaining=min(remaining_m,self.coast_distance(state.speed_mps))
        u=super().command(trajectory,state,reason,dt,remaining_m=decision_remaining)
        if self.braking and not reason and trajectory is not None:
            u['throttle']=0.;self.previous_throttle=0.
            if action=='brake_001' and state.speed_mps>.02:u['brake']=max(u['brake'],.01)
        u['envelope_decision']=self.decision
        return u
