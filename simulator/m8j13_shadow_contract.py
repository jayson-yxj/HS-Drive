"""Parked-only admission and diagnostic IPC; no model or simulator imports."""
import math


def require_parked_command(command):
    if command!={'steer':0.,'throttle':0.,'brake':1.}:raise ValueError('ParkedOnlyControlViolation')


def require_completion_only(response):
    allowed={'status','reason','model_forward','finite','result_expired'}
    if not set(response)<=allowed:raise ValueError('ExecutableOrUnexpectedModelIPC')
    if not isinstance(response.get('model_forward'),bool):raise ValueError('InvalidModelCompletion')


def candidate_offset(index,interval=4,count=80):
    if index<0 or index>=count:raise ValueError('CandidateBudgetExceeded')
    return index*interval


def parked_measurement_valid(speed):return math.isfinite(speed) and speed<=.1


def snapshot_is_current(first,second,state):
    """No claim of current simulation age while the sole owner advances a tick."""
    return first is not None and first==second and not second['pending'] and state['frame']==second['frame']
