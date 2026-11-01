import ast
import json
from pathlib import Path
from types import SimpleNamespace as S
import pytest
from simulator.m8j13_navigation import set_navigation_route
from simulator.m8j13_execution import execution_reason,issue_command,feedback_matches,round_exit_reason


def test_legacy_gps_and_world_routes_have_no_native_double_conversion():
    calls=[];planner=S(set_route=lambda route,gps:calls.append((route,gps)))
    gps=[({'lat':42.,'lon':2.},4)]
    assert set_navigation_route(planner,gps,coordinate_type='gps',client_version='0.9.15')=='LegacyGPS0915'
    assert calls[-1]==(gps,True)
    route=[(S(location=S(x=1.,y=2.)),4)]
    assert set_navigation_route(planner,route,coordinate_type='world_carla',client_version='0.9.16')=='WorldRouteUnchanged'
    assert calls[-1]==(route,False)
    with pytest.raises(ValueError,match='WorldRouteRequiresLocation'):
        set_navigation_route(planner,gps,coordinate_type='world_carla',client_version='0.9.16')
    with pytest.raises(ValueError,match='NativeMapInverseRequired'):
        set_navigation_route(planner,gps,coordinate_type='gps',client_version='0.9.16')


def state():
    return dict(armed=True,require_planner=True,
        planner=dict(heartbeat_wall_s=10.,capture_wall_s=10.),
        issued=dict(issued_wall_s=10.,valid_until_wall_s=10.3),
        ack=dict(confirmed_wall_s=10.,matched=True))


def test_fresh_planner_cannot_refresh_expired_command_or_feedback():
    s=state();assert execution_reason(s,10.1) is None
    s['planner']=dict(heartbeat_wall_s=10.5,capture_wall_s=10.5)
    assert execution_reason(s,10.5)=='ControlCommandExpired'
    s['issued']=dict(issued_wall_s=10.5,valid_until_wall_s=10.8)
    assert execution_reason(s,10.5)=='ControlFeedbackExpired'
    s['ack']=dict(confirmed_wall_s=10.5,matched=False)
    assert execution_reason(s,10.5)=='ControlFeedbackMismatch'


def test_live_execution_ack_cannot_hide_blocked_planner():
    s=state();s['issued']=dict(issued_wall_s=10.5,valid_until_wall_s=10.8)
    s['ack']=dict(confirmed_wall_s=10.5,matched=True)
    assert execution_reason(s,10.5)=='PlannerWatchdogExpired'
    assert execution_reason(s,10.5,manual=True)=='ManualEmergencyStop'


def test_command_never_outlives_original_plan_or_accepts_model_source():
    plan=dict(capture_wall_s=10.,capture_sim_s=2.)
    raw=dict(steer=0.,throttle=.5,brake=0.)
    cmd=issue_command(raw,now=10.2,source='IndependentTestControl',sequence=1,plan=plan,sim_s=2.1)
    assert cmd['valid_until_wall_s']==10.3
    for wall,sim in [(10.31,2.1),(10.2,2.31)]:
        with pytest.raises(ValueError,match='ExpiredTestPlan'):
            issue_command(raw,now=wall,source='IndependentTestControl',sequence=2,plan=plan,sim_s=sim)
    with pytest.raises(ValueError,match='ForbiddenControlSource'):
        issue_command(raw,now=10.2,source='Model',sequence=1)


def test_feedback_checks_handbrake_reverse_and_automatic_gear_semantics():
    cmd=issue_command(dict(steer=0.,throttle=.5,brake=0.),now=1.,source='IndependentTestControl',sequence=1)['control']
    assert feedback_matches(cmd,dict(cmd,gear=1))
    assert not feedback_matches(cmd,dict(cmd,hand_brake=True))
    assert not feedback_matches(cmd,dict(cmd,reverse=True))
    assert not feedback_matches(cmd,dict(cmd,brake=1.))


def test_missing_speed_precondition_reserves_half_round_for_braking():
    c=json.loads(Path('configs/m8j13_navigation_stop.json').read_text())
    assert round_exit_reason(199,.5,injected=False) is None
    assert round_exit_reason(200,.5,injected=False)=='PreconditionsNotMet'
    assert c['start_deadline_ticks']==200 and c['round_max_ticks']==400
    assert c['basic_total_tick_cap']+c['faults_total_tick_cap']==2400
    assert c['basic_round_max_ticks']+c['settings_tick_charge']<=400
    assert 4*c['round_max_ticks']+c['settings_tick_charge']<=2000


def test_new_owner_and_watchdog_never_apply_velocity_or_tick_twice():
    source=Path('tools/m8j13_navigation_stop.py').read_text();tree=ast.parse(source)
    banned={'set_target_velocity','enable_constant_velocity','set_simulate_physics','set_transform','apply_physics_control'}
    assert not any(isinstance(n,ast.Attribute) and n.attr in banned for n in ast.walk(tree))
    watchdog=ast.parse(Path('tools/m8j13_watchdog.py').read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr=='tick' for n in ast.walk(watchdog))
