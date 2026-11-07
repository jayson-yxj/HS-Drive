"""Parked-only handoff credentials and persistent ordering uncertainty."""
import copy
import math
from simulator.m8j13_logic_runtime import STOP, canonical, frozen_equal

REVISION = 'parked_handoff_deadline_v1'
PROTOCOL = 'synchronous_logic_v1_parked_shadow'


def enabled(cfg):
    return cfg.get('parked_handoff_revision') == REVISION


def handoff_proof(cfg, source, control, current, state, *, pending=None,
                  guard_pending=None, uncertain=False, locked=False, active=None):
    if not enabled(cfg) or cfg.get('shadow_protocol') != PROTOCOL or cfg.get('mode') != 'ExplicitParkedModelShadow' or cfg.get('model_actuation_allowed') is not False:
        raise ValueError('ParkedHandoffMode')
    phases = {'PlannedHoldBrake': 'ConfirmedFrozenWorld', 'InitialProtectionBrake': 'FiniteExecution'}
    if source not in phases or state['phase'] != phases[source] or canonical(control) != canonical(STOP):
        raise ValueError('ParkedHandoffSourcePhaseOrControl')
    if pending or guard_pending or uncertain or locked or state.get('handoff_uncertain') or state.get('locked') or state.get('fault') or not state.get('armed'):
        raise ValueError('ParkedHandoffUncertainOrLocked')
    if not current or not math.isfinite(current['speed_norm']) or not 0 <= current['speed_norm'] <= .1 or canonical(current.get('full_control')) != canonical(STOP) or current.get('feedback_matches') is not True:
        raise ValueError('ParkedHandoffPhysicalStopUnconfirmed')
    if not state.get('frozen_confirmation') or not frozen_equal(current, state['frozen_confirmation']):
        raise ValueError('ParkedHandoffFrozenUnconfirmed')
    if not state.get('identity') or state.get('ego_id') is None or canonical(current.get('identity')) != canonical(state['identity']) or current.get('ego_id') != state['ego_id'] or current.get('epoch') != state['epoch']:
        raise ValueError('ParkedHandoffIdentity')
    previous = 'InitialProtectionBrake' if source == 'PlannedHoldBrake' else 'PlannedHoldBrake'
    if not active or active['source'] != previous or active['epoch'] != state['epoch'] or canonical(active['control']) != canonical(STOP):
        raise ValueError('ParkedHandoffPreviousCommand')
    return dict(revision=REVISION, identity=list(state['identity']), actor_id=state['ego_id'], epoch=state['epoch'],
                phase=state['phase'], phase_started=state['phase_started'], phase_deadline=state['phase_deadline'],
                previous_sequence=active['sequence'], previous_source=active['source'], frozen=copy.deepcopy(current))


def verify_io_proof(cfg, proof, command, current, state, operation, identity, actor_id,
                    io_uncertain=False, *, last=None, active=None):
    """Verify fresh physical feedback and I/O-owned previous command, not just owner flags."""
    if not state or io_uncertain:raise ValueError('ParkedHandoffIOUncertain')
    if state.get('operation') != operation:raise ValueError('ParkedHandoffOperationIdentity')
    expected=handoff_proof(cfg, command['source'], command['control'], last, state,
                           uncertain=state.get('handoff_uncertain',False), active=active)
    for key in ('revision','identity','actor_id','epoch','phase','phase_started','phase_deadline','previous_sequence','previous_source'):
        if canonical(proof.get(key)) != canonical(expected[key]):raise ValueError('ParkedHandoffCredential:'+key)
    if canonical(identity) != canonical(proof['identity']) or actor_id != proof['actor_id'] or command['epoch'] != proof['epoch']:
        raise ValueError('ParkedHandoffActorEpoch')
    if not frozen_equal(proof['frozen'],last) or not frozen_equal(last,current) or canonical(current['full_control']) != canonical(STOP):
        raise ValueError('ParkedHandoffWorldOrFeedbackChanged')
    if operation['operation_limit_s'] != .5 or operation['phase_deadline'] != proof['phase_deadline'] or operation['deadline'] > min(operation['started']+.5,proof['phase_deadline'],operation['run_deadline']):
        raise ValueError('ParkedHandoffDeadline')
    obs=command.get('observation')
    if not obs or operation['deadline'] > obs['input_wall_s']+4.:raise ValueError('ParkedHandoffObservationDeadline')


def uncertainty(state, io_failure=None):
    """Local completion never clears a possibly outstanding remote control write."""
    failed=state.get('failed_operation') or (io_failure or {}).get('operation')
    return bool(state.get('handoff_uncertain') or failed or io_failure),failed
