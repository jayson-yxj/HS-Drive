"""All events here are synthetic. No CARLA, model, RPC or real stop evidence."""
import ast
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from simulator.m8j13_logic_protocol import LogicProtocol, Snapshot, State
from simulator.m8j13_safety import PlanStamp, freshness

CFG = json.loads(Path('configs/m8j13_synchronous_logic_v1.draft.json').read_text())
GEN = ('synthetic_run', 'synthetic_episode', 'sensors_1')
IDS = {name: 100 + i for i, name in enumerate(CFG['camera_order'])}


def protocol():
    p = LogicProtocol(CFG, Snapshot(0, 0., 0.), sensor_instances=IDS)
    assert p.stable([Snapshot(i, i * .05, 0.) for i in range(1, 22)], .1)
    return p


def heartbeat_wait(p, duration):
    end = p.wall + duration
    while p.wall < end - 1e-9:
        p.advance(min(end, p.wall + .1), heartbeat=True)


def deliver(p):
    token = p.command(p.wall + .001)
    assert p.dispatch(token, p.wall + .001)
    assert p.command_return(token.sequence, p.wall + .001)
    return token


def tick(p, speed=0.):
    assert p.begin_tick(p.wall + .001)
    assert p.tick_return(Snapshot(p.snap.frame + 1, p.snap.sim + .05, speed), p.wall + .01)


def capture(p=None, omit=None):
    p = protocol() if p is None else p
    assert p.begin_capture(p.wall + .001, GEN)
    if p.target > p.snap.frame:
        deliver(p)
        tick(p)
    for name in CFG['camera_order']:
        if name != omit:
            assert p.sensor(name, p.target, p.target_sim, GEN, IDS[name], p.wall + .001)
    assert p.ego_metadata(p.target, p.target_sim, p.target, p.target_sim, GEN, True, p.wall)
    tick(p)
    tick(p)
    return p


def compute():
    p = capture()
    assert p.confirm_quiescent(p.snap, p.wall + .001, no_pending_tick=True)
    deliver(p)
    assert p.begin_compute(p.wall + .001)
    return p


def accept(p, result='result_1'):
    assert p.receive_result(result, p.input['frame'], p.snap, p.wall + .001, generation=GEN)
    assert p.accept_pid((0., 1., 0.), p.wall + .001, future_speed_max=8.)
    return p


def test_disabled_draft_has_no_runtime_path_or_dependency():
    assert not CFG['execution_enabled'] and not CFG['runtime_accepted'] and not CFG['model_actuation_allowed']
    assert CFG['runtime_entrypoint'] is None
    assert set(CFG['live_authorized_this_stage'].values()) == {0}
    tree = ast.parse(Path('simulator/m8j13_logic_protocol.py').read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(imports) <= {'dataclasses', 'enum', 'math', 'copy'}
    for field in ('execution_enabled', 'runtime_accepted', 'model_actuation_allowed'):
        cfg = copy.deepcopy(CFG)
        cfg[field] = True
        with pytest.raises(ValueError):
            LogicProtocol(cfg, Snapshot(0, 0, 0), sensor_instances=IDS)


def test_slow_frozen_compute_is_logic_valid_but_old_realtime_expired():
    p = compute()
    started = p.wall
    original = dict(p.input)
    heartbeat_wait(p, .6)
    assert p.receive_result('slow', p.input['frame'], p.snap, p.wall, generation=GEN)
    assert p.input == original
    stamp = PlanStamp('synthetic', p.input['frame'], p.input['sim'], p.input['wall'], started, p.wall, 'synthetic_hash')
    assert freshness(stamp, 'synthetic', p.snap.frame, p.snap.sim, p.wall) == 'PlanWallExpired'
    assert p.accept_pid((0., .2, 0.), p.wall)


def test_two_complete_cycles_preserve_cadence_original_expiry_and_one_pid_call():
    p = accept(compute())
    first = dict(p.input)
    token = deliver(p)
    assert token.valid_until_sim == pytest.approx(first['sim'] + .3)
    tick(p, 1.)
    tick(p, 1.)
    capture(p)
    assert p.target == first['frame'] + 4
    assert p.snap.sim == pytest.approx(token.valid_until_sim)
    assert p.pid_calls == 1 and p.physical_ticks == 28
    assert p.confirm_quiescent(p.snap, p.wall + .001, True)
    assert p.input['sim'] - first['sim'] == pytest.approx(.2)
    deliver(p)
    assert p.begin_compute(p.wall + .001)
    heartbeat_wait(p, .4)
    accept(p, 'result_2')
    deliver(p)
    tick(p, 1.)
    assert p.pid_calls == 2
    assert any(e.get('future_speed_above_cap') and e.get('saturated') for e in p.events)


@pytest.mark.parametrize('kind', ['advanced', 'unconfirmed', 'nonfinite', 'pending', 'pose'])
def test_result_requires_exact_confirmed_frozen_world(kind):
    p = compute()
    snap = p.snap
    if kind == 'advanced': snap = replace(snap, frame=snap.frame + 5, sim=snap.sim + .25)
    if kind == 'unconfirmed': snap = replace(snap, confirmed=False)
    if kind == 'nonfinite': snap = replace(snap, sim=float('nan'))
    if kind == 'pose': snap = replace(snap, pose=(1., 0., 0., 0.))
    if kind == 'pending': p.tick_pending = snap.frame + 1
    assert not p.receive_result('bad', p.input['frame'], snap, p.wall + .01, generation=GEN)
    assert p.state == State.FAULT


@pytest.mark.parametrize('kind', ['generation', 'run', 'frame', 'duplicate', 'actor', 'timestamp', 'nan', 'late'])
def test_raw_sensor_identity_failures_latch(kind):
    p = protocol()
    assert p.begin_capture(p.wall + .001, GEN)
    deliver(p)
    tick(p)
    name = CFG['camera_order'][0]
    frame, sim, generation, actor = p.target, p.target_sim, GEN, IDS[name]
    if kind == 'generation': generation = (GEN[0], GEN[1], 'old')
    if kind == 'run': generation = ('old', GEN[1], GEN[2])
    if kind == 'frame': frame += 1
    if kind == 'actor': actor += 1
    if kind == 'timestamp': sim += .05
    if kind == 'nan': sim = float('nan')
    if kind == 'duplicate': assert p.sensor(name, frame, sim, generation, actor, p.wall)
    if kind == 'late': heartbeat_wait(p, .51)
    assert not p.sensor(name, frame, sim, generation, actor, p.wall)
    assert p.state == State.FAULT


def test_missing_camera_has_bounded_capture_and_no_extra_flush():
    p = capture(omit=CFG['camera_order'][0])
    assert not p.confirm_quiescent(p.snap, p.wall, True)
    assert p.fault == 'MissingInput'
    q = capture()
    assert not q.begin_tick(q.wall + .001)
    assert q.fault == 'CaptureTickCap'


def test_current_snapshot_pending_is_never_confirmed_by_flag_only():
    p = capture()
    p.tick_pending = p.snap.frame + 1
    assert not p.confirm_quiescent(p.snap, p.wall, True)
    assert p.fault == 'UnconfirmedWorld'


def test_worker_deadline_survives_other_heartbeats():
    p = compute()
    deadline = p.deadline
    heartbeat_wait(p, 2.1)
    assert p.fault == 'PhaseDeadline:ExpectedComputeWait'
    assert p.wall > deadline
    assert not p.receive_result('late', p.input['frame'], p.snap, p.wall, generation=GEN)


def test_missing_owner_heartbeat_is_distinct_from_expected_worker_wait():
    p = compute()
    assert not p.advance(p.wall + .301, heartbeat=True)
    assert p.fault == 'OwnerUnresponsive'


@pytest.mark.parametrize('reason', ['trajectory', 'nan', 'range', 'conflict', 'manual'])
def test_invalid_output_and_manual_stop_are_latched(reason):
    p = compute()
    if reason == 'trajectory': assert not p.receive_result('invalid', p.input['frame'], p.snap, p.wall, False, generation=GEN)
    elif reason == 'manual': p.manual_stop(p.wall)
    else:
        assert p.receive_result('invalid_control', p.input['frame'], p.snap, p.wall, generation=GEN)
        raw = {'nan': (0., float('nan'), 0.), 'range': (2., 0., 0.), 'conflict': (0., 1., 1.)}[reason]
        assert not p.accept_pid(raw, p.wall)
    assert p.state == State.FAULT
    assert not p.receive_result('late_good', p.input['frame'], p.snap, p.wall, generation=GEN)
    assert p.command(p.wall).source == 'FaultBrake'


def test_stop_beats_unsent_old_command_and_rejects_replay():
    p = accept(compute())
    normal = p.command(p.wall)
    p.manual_stop(p.wall)
    assert not p.dispatch(normal, p.wall)
    brake = deliver(p)
    assert brake.source == 'FaultBrake'
    assert not p.dispatch(normal, p.wall)
    assert p.active == brake


def test_inflight_rpc_does_not_hold_stop_lock_but_remote_order_is_unverified():
    p = accept(compute())
    normal = p.command(p.wall)
    assert p.dispatch(normal, p.wall)
    p.manual_stop(p.wall)
    brake = deliver(p)  # Independent stop RPC can be submitted despite pending normal RPC.
    assert brake.source == 'FaultBrake'
    assert not p.begin_tick(p.wall)  # Cannot resume with unresolved remote ordering.
    assert not p.command_return(normal.sequence, p.wall + .01)
    assert p.active is None and p.delivery_status == 'ResponseUnverified'
    deliver(p)  # Fresh brake AFTER old RPC has returned.
    tick(p, .8)
    assert p.delivery_status == 'ControlFeedbackObserved'


@pytest.mark.parametrize('kind', ['no_delivery', 'rpc_timeout', 'tick_timeout', 'disconnect'])
def test_transport_failures_never_claim_physical_stop(kind):
    p = accept(compute())
    if kind == 'disconnect': p.connection_lost(p.wall)
    else:
        token = p.command(p.wall)
        assert p.dispatch(token, p.wall)
        if kind == 'no_delivery': assert not p.command_return(token.sequence, p.wall, delivered=False)
        elif kind == 'rpc_timeout': heartbeat_wait(p, .21)
        else:
            assert p.command_return(token.sequence, p.wall)
            assert p.begin_tick(p.wall)
            heartbeat_wait(p, .51)
    expected = 'CommandNotDelivered' if kind in ('no_delivery', 'disconnect') else 'ResponseUnverified'
    assert p.delivery_status == expected
    assert p.state == State.FAULT and p.stop_hold == 0
    heartbeat_wait(p, 10.1)
    assert p.state == State.CLOSED
    assert not p.delivery_status.startswith('PhysicalStop')


def test_pause_is_not_stop_hold_and_only_successive_feedback_ticks_count():
    p = accept(compute())
    p.manual_stop(p.wall)
    deliver(p)
    heartbeat_wait(p, 1.5)
    assert p.stop_hold == 0 and p.state == State.FAULT
    tick(p, .8)
    tick(p, .05)
    heartbeat_wait(p, 1.5)
    assert p.stop_hold == 0
    for _ in range(20): tick(p, .05)
    assert p.stop_hold == pytest.approx(1.)
    assert p.state == State.STOPPED
    assert p.delivery_status == 'PhysicalStopObserved_SyntheticOnly'


def test_incorrect_feedback_cannot_count_as_braking():
    p = accept(compute())
    p.manual_stop(p.wall)
    deliver(p)
    assert p.begin_tick(p.wall)
    assert not p.tick_return(Snapshot(p.snap.frame + 1, p.snap.sim + .05, 0.), p.wall + .01, feedback_matches=False)
    assert p.stop_hold == 0


def test_no_ticks_during_compute_no_duplicate_result_no_backlog():
    p = compute()
    assert not p.begin_tick(p.wall)
    assert p.fault == 'TickDuringExpectedWait'
    p = accept(compute())
    assert not p.receive_result('result_1', p.input['frame'], p.snap, p.wall, generation=GEN)
    p = accept(compute())
    deliver(p)
    assert p.begin_tick(p.wall)
    assert not p.begin_tick(p.wall)
    assert p.fault == 'ConcurrentTickOrBacklog'


def test_planned_hold_cannot_be_used_to_advance_normal_execution():
    p = accept(compute())
    assert not p.begin_tick(p.wall)
    assert p.fault == 'NormalHandoffUnconfirmed'


def test_command_keeps_original_wall_deadline_and_cannot_refresh_after_retry():
    p = accept(compute())
    first = p.command(p.wall)
    heartbeat_wait(p, .2)
    second = p.command(p.wall)
    assert first.valid_until_wall == second.valid_until_wall == p.input['wall'] + 4.
    assert first.valid_until_sim == second.valid_until_sim
    forged = replace(second, valid_until_sim=second.valid_until_sim + .3)
    assert not p.dispatch(forged, p.wall)


def test_speed_guard_intervention_is_not_model_success():
    p = accept(compute())
    deliver(p)
    assert p.begin_tick(p.wall)
    assert not p.tick_return(Snapshot(p.snap.frame + 1, p.snap.sim + .05, 4.5), p.wall + .01)
    assert p.fault == 'IndependentSpeedGuard' and p.interventions == 1
    assert p.command(p.wall).source == 'FaultBrake'


def test_whole_run_deadline_is_not_renewed_by_phase_or_heartbeat():
    p = compute()
    p.born = p.wall - CFG['wall_deadlines_s']['whole_run'] + .05
    heartbeat_wait(p, .06)
    assert p.fault == 'RunDeadline'


def test_input_frame_and_imu_must_match_navigation_contract():
    p = capture()
    assert not p.ego_metadata(p.target, p.target_sim, p.target + 1, p.target_sim, GEN, True, p.wall)
    assert p.fault == 'EgoIMUNavigationMismatch'


def test_normal_termination_uses_brake_observation_without_model_stop_credit():
    p = accept(compute())
    assert p.finish(p.wall)
    assert p.fault == 'NormalTerminationBrakeRequired'
    deliver(p)
    for _ in range(21): tick(p, 0.)
    assert p.state == State.STOPPED and p.pid_calls == 1


def test_control_replay_rejected_without_a_second_pid_update():
    p = accept(compute())
    token = deliver(p)
    assert not p.dispatch(token, p.wall)
    assert p.pid_calls == 1 and p.state == State.FAULT


def test_result_from_old_generation_cannot_reuse_same_frame_number():
    p = compute()
    assert not p.receive_result('old_episode', p.input['frame'], p.snap, p.wall,
                                generation=('old_run', GEN[1], GEN[2]))
    assert p.fault == 'ExpiredOrWrongInput'


def test_command_wall_budget_reserves_bounded_tick_without_reissuing_age():
    p = accept(compute())
    token = deliver(p)
    # Synthetic long but still progressing prior phase: isolate original command limit.
    p.deadline = token.valid_until_wall + 1.
    heartbeat_wait(p, token.valid_until_wall - p.wall - .49)
    assert not p.begin_tick(p.wall)
    assert p.fault == 'OriginalCommandValidityExhausted'


def test_unconfirmed_tick_cannot_release_inflight_slot_or_count_stop_hold():
    p = accept(compute())
    p.manual_stop(p.wall)
    deliver(p)
    assert p.begin_tick(p.wall)
    expected = p.tick_pending
    assert not p.tick_return(Snapshot(expected, p.snap.sim + .05, 0., confirmed=False), p.wall + .01)
    assert p.tick_pending == expected and p.stop_hold == 0.


def test_lock_does_not_restart_stopping_deadline_or_heartbeat_extend_it():
    p = accept(compute())
    p.manual_stop(p.wall)
    deadline = p.deadline
    heartbeat_wait(p, .2)
    p.manual_stop(p.wall)
    assert p.deadline == deadline
    heartbeat_wait(p, 10.)
    assert p.state == State.CLOSED


def test_lateral_motion_cannot_count_as_stopped_when_forward_speed_is_zero():
    p = accept(compute())
    p.manual_stop(p.wall)
    deliver(p)
    for _ in range(22):
        assert p.begin_tick(p.wall)
        assert p.tick_return(Snapshot(p.snap.frame + 1, p.snap.sim + .05, .4, forward_speed=0.), p.wall + .01)
    assert p.state == State.FAULT and p.stop_hold == 0.
