"""Budget checks before model/server/PID creation; configuration only."""
import copy
import json
from pathlib import Path

import pytest

from simulator.m8j13_logic_shadow_budget import validate_shadow_budget


def config(name='m8j13_logic_shadow_recheck.json'):
    return json.loads((Path('configs') / name).read_text())


def test_24_live_targets_fit_before_any_execution_and_config_is_not_mutated():
    cfg = config()
    before = copy.deepcopy(cfg)
    assert validate_shadow_budget(cfg) is True
    assert cfg == before
    assert cfg['live_pid_remaining'] == cfg['planned_targets'] == 24
    assert 3 + 23 * 4 == cfg['tick_caps']['stationary']


def test_legacy_budget_and_failure_evidence_remain_unchanged():
    cfg = config('m8j13_logic_shadow.json')
    before = copy.deepcopy(cfg)
    assert validate_shadow_budget(cfg) is False
    assert cfg == before
    assert cfg['preflight_pid_calls'] == 1 and cfg['live_pid_remaining'] == 23
    assert cfg['wall_deadlines_s']['quiesce_and_hold_delivery'] == .2


@pytest.mark.parametrize('field,value', [
    ('preflight_pid_calls', 1), ('live_pid_remaining', 23),
    ('max_pid_calls', 25), ('max_forward_attempts', 25), ('planned_targets', 23),
    ('offline_forward_cap', 1), ('model_forward_attempts', 1),
    ('max_starts', 2), ('max_model_starts', 2), ('max_active_ticks', 801),
    ('max_starts', True), ('planned_targets', 24.0),
    ('model_actuation_allowed', True), ('training', True),
    ('runtime_driving_accepted', True), ('all_faults_terminal', False),
    ('model_actuation_allowed', 0), ('parked_handoff_wall_s', .6),
    ('initialization_tick_wall_s', 3.), ('speed_limit_mps', 2.),
    ('shadow_protocol', 'realtime'), ('mode', 'ExplicitNoModelRuntimeTest'),
    ('revision', 'unknown'), ('parked_handoff_revision', 'unknown'),
    ('fault_injection', {'scenario': 'invalid_output'}),
])
def test_impossible_or_expanded_recheck_is_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match='ShadowRecheckConfiguration'):
        validate_shadow_budget(cfg)


@pytest.mark.parametrize('field', [
    'command_delivery', 'compute', 'validation', 'owner_heartbeat',
    'tick_rpc', 'capture', 'quiesce_and_hold_delivery',
    'observation_to_last_command_effect', 'fault_stop_observation',
])
def test_other_deadlines_cannot_be_silently_relaxed(field):
    cfg = config()
    cfg['wall_deadlines_s'][field] += .1
    with pytest.raises(ValueError, match='wall_deadlines_s'):
        validate_shadow_budget(cfg)


def test_budget_reallocation_or_partial_revision_rejected():
    cfg = config()
    cfg['tick_caps']['stationary'] += 4
    cfg['tick_caps']['failure_cleanup_reserve'] -= 4
    with pytest.raises(ValueError, match='tick_caps'):
        validate_shadow_budget(cfg)
    cfg = config()
    del cfg['parked_handoff_revision']
    with pytest.raises(ValueError, match='MissingRevision'):
        validate_shadow_budget(cfg)
