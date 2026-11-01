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


def reconcile_watchdog_stop(feedback,events,fault,commands):
    """Use the first independent brake, not the later owner observation.

    A wall-clock trigger inside a synchronous tick has no invented exact
    simulation timestamp/position. Report the adjacent measured bracket.
    """
    delivered=next((x for x in events if x.get('submitted')),None)
    if delivered is None:return dict(status='CommandNotDelivered')
    first=next((i for i,x in enumerate(feedback) if x['tick_end_wall_s']>=delivered['call_end_wall_s']
        and x['full_control']['brake']>0 and x['full_control']['throttle']==0),None)
    if first is None:return dict(status='ResponseUnverified')
    previous=feedback[max(0,first-1)];brake=feedback[first]
    stopped=next((i for i in range(first,len(feedback)) if feedback[i]['speed_norm']<=.1),None)
    end=stopped
    if stopped is not None:
        for i in range(stopped,len(feedback)):
            row=feedback[i]
            if row['speed_norm']>.1 or row['full_control']['brake']<=0 or row['full_control']['throttle']!=0:break
            end=i
    held=0. if stopped is None else feedback[end]['sim_s']-feedback[stopped]['sim_s']
    last_command=next(x for x in reversed(commands) if x['issued_wall_s']<=fault['wall_s'])
    return dict(status='PhysicalStopObserved' if held>=1.-1e-6 else 'ResponseObserved_HoldIncomplete',
        source='IndependentWatchdog',reason=fault['reason'],speed_before_mps=previous['speed_norm'],
        first_brake_feedback_speed_mps=brake['speed_norm'],stop_hold_sim_s=held,
        detection_after_deadline_wall_ms=(fault['wall_s']-last_command['valid_until_wall_s'])*1000,
        detection_to_submission_wall_ms=(delivered['call_end_wall_s']-fault['wall_s'])*1000,
        submission_to_feedback_wall_ms=(brake['tick_end_wall_s']-delivered['call_end_wall_s'])*1000,
        previous_to_first_brake_sim_s=brake['sim_s']-previous['sim_s'],
        previous_to_stop_sim_s=None if stopped is None else feedback[stopped]['sim_s']-previous['sim_s'],
        previous_to_stop_distance_m=None if stopped is None else math.hypot(
            feedback[stopped]['position'][0]-previous['position'][0],feedback[stopped]['position'][1]-previous['position'][1]),
        timing_limit='Trigger occurred between measured snapshots; simulation duration/distance are observation brackets, not exact values at trigger',
        normal_commands_after_latch=sum(x['source']=='IndependentTestControl' and x['issued_wall_s']>fault['wall_s'] for x in commands))
