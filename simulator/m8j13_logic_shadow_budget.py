"""Pre-load admission for the explicit one-run parked handoff recheck.

Pure configuration validation: no model, PID, simulator, files, or counters.
Historical configurations remain readable and keep their original behavior.
"""

REVISION = 'parked_handoff_deadline_v1'
PROTOCOL = 'synchronous_logic_v1_parked_shadow'

TICK_CAPS = {
    'settings_startup': 30, 'initialization': 60, 'stationary': 95,
    'start': 0, 'fault_stop': 120, 'exit': 30, 'settings_restore': 30,
    'failure_cleanup_reserve': 435,
}
WALL_LIMITS = {
    'server_startup': 150., 'initialization': 30., 'capture': .5,
    'quiesce_and_hold_delivery': .5, 'compute': 2., 'validation': .1,
    'execute_to_next_target': 1., 'observation_to_last_command_effect': 4.,
    'owner_heartbeat': .3, 'owner_poll': .02, 'tick_rpc': .5,
    'command_delivery': .2, 'command_feedback': .5,
    'fault_stop_observation': 10., 'model_exit': 30., 'cleanup': 15.,
    'whole_run': 240.,
}


def _number(value, expected, name, integer=False):
    kinds = (int,) if integer else (int, float)
    if type(value) not in kinds or value != expected:
        raise ValueError('ShadowRecheckConfiguration:' + name)


def validate_shadow_budget(cfg):
    """Reject impossible/expanded recheck budgets before loading the model.

    Returns True for an admitted new revision, False for a legacy config.
    An unknown or partially enabled revision fails closed.
    """
    revision = cfg.get('parked_handoff_revision')
    if revision is None:
        if cfg.get('revision') == REVISION or 'parked_handoff_wall_s' in cfg:
            raise ValueError('ShadowRecheckConfiguration:MissingRevision')
        return False
    if revision != REVISION or cfg.get('revision') != REVISION:
        raise ValueError('ShadowRecheckConfiguration:UnknownRevision')
    if (cfg.get('protocol_id') != 'synchronous_logic_v1'
            or cfg.get('shadow_protocol') != PROTOCOL
            or cfg.get('mode') != 'ExplicitParkedModelShadow'):
        raise ValueError('ShadowRecheckConfiguration:ExplicitMode')
    for name in ('runtime_driving_accepted', 'model_actuation_allowed', 'training'):
        if cfg.get(name) is not False:
            raise ValueError('ShadowRecheckConfiguration:' + name)
    if cfg.get('all_faults_terminal') is not True or cfg.get('fault_injection'):
        raise ValueError('ShadowRecheckConfiguration:TerminalNoInjection')
    integer_fields = {
        'planned_targets': 24, 'max_forward_attempts': 24, 'max_pid_calls': 24,
        'live_pid_remaining': 24, 'preflight_pid_calls': 0,
        'model_forward_attempts': 0, 'offline_forward_cap': 0,
        'max_starts': 1, 'max_model_starts': 1, 'max_active_ticks': 800,
        'stable_ticks': 21, 'capture_flush_ticks': 2, 'target_interval_ticks': 4,
    }
    for name, expected in integer_fields.items():
        _number(cfg.get(name), expected, name, integer=True)
    caps = cfg.get('tick_caps')
    if not isinstance(caps, dict) or set(caps) != set(TICK_CAPS):
        raise ValueError('ShadowRecheckConfiguration:TickCaps')
    for name, expected in TICK_CAPS.items():
        _number(caps[name], expected, 'tick_caps.' + name, integer=True)
    if sum(caps.values()) != cfg['max_active_ticks']:
        raise ValueError('ShadowRecheckConfiguration:TickSum')
    # One target tick plus two drain ticks, then 23 complete four-tick cycles.
    measurement_ticks = 1 + cfg['capture_flush_ticks'] + (
        cfg['planned_targets'] - 1) * cfg['target_interval_ticks']
    if measurement_ticks != caps['stationary']:
        raise ValueError('ShadowRecheckConfiguration:MeasurementTicks')
    wall = cfg.get('wall_deadlines_s')
    if not isinstance(wall, dict) or set(wall) != set(WALL_LIMITS):
        raise ValueError('ShadowRecheckConfiguration:WallLimits')
    for name, expected in WALL_LIMITS.items():
        _number(wall[name], expected, 'wall_deadlines_s.' + name)
    for name, expected in {
        'parked_handoff_wall_s': .5, 'model_load_wall_s': 120.,
        'model_exit_wall_s': 30., 'initialization_tick_wall_s': 2.,
        'initialization_min_tick_interval_s': .05, 'physics_dt_s': .05,
        'input_wait_wall_s': .5, 'collect_return_reserve_s': .1,
        'speed_limit_mps': .1,
    }.items():
        _number(cfg.get(name), expected, name)
    return True
