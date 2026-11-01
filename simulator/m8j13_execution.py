"""Opt-in execution contract, independent from model and CARLA imports."""
import math
from simulator.m8j13_safety import sanitize_control

STOP=dict(steer=0.,throttle=0.,brake=1.)
FIELDS=('steer','throttle','brake','hand_brake','reverse','manual_gear_shift','gear')


def control_record(control):return {k:getattr(control,k) for k in FIELDS}


def feedback_matches(request,feedback):
    # In automatic mode the gearbox chooses the reported gear itself.
    keys=FIELDS if request['manual_gear_shift'] else FIELDS[:-1]
    return all(abs(float(request[k])-float(feedback[k]))<1e-5 for k in keys)


def execution_reason(state,now,*,manual=False,timeout=.3):
    if manual:return 'ManualEmergencyStop'
    if not state.get('armed',False):return None
    if state.get('require_planner',False):
        plan=state.get('planner')
        if not plan:return 'MissingPlannerStatus'
        if now-plan['heartbeat_wall_s']>timeout:return 'PlannerWatchdogExpired'
        if now>plan['capture_wall_s']+timeout:return 'PlanWallExpired'
    issued=state.get('issued');ack=state.get('ack')
    if not issued:return 'MissingIssuedCommand'
    if now>issued['valid_until_wall_s']:return 'ControlCommandExpired'
    if not ack or now-ack['confirmed_wall_s']>timeout:return 'ControlFeedbackExpired'
    if not ack['matched']:return 'ControlFeedbackMismatch'
    clocks=[issued['issued_wall_s'],issued['valid_until_wall_s'],ack['confirmed_wall_s']]
    if not all(math.isfinite(x) for x in clocks) or max(issued['issued_wall_s'],ack['confirmed_wall_s'])>now+1e-6:
        return 'InvalidExecutionClock'
    return None


def issue_command(raw,*,now,source,sequence,plan=None,sim_s=None):
    if source not in ('IndependentTestControl','SafetyStop','IndependentWatchdog'):raise ValueError('ForbiddenControlSource')
    u,error=sanitize_control(raw)
    if error:raise ValueError(error)
    until=now+.3
    if source=='IndependentTestControl' and plan is not None:
        until=min(until,plan['capture_wall_s']+.3)
        if now>until or sim_s-plan['capture_sim_s']>.3+1e-6:raise ValueError('ExpiredTestPlan')
    return dict(sequence=sequence,source=source,issued_wall_s=now,valid_until_wall_s=until,
        control=dict(u,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0))


def round_exit_reason(index,speed,*,injected,start_deadline_ticks=200):
    if abs(speed)>2.:return 'TestSpeedLimitExceeded'
    if not injected and index>=start_deadline_ticks:return 'PreconditionsNotMet'
    return None
