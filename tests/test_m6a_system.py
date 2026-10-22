"""Safety-relevant schema and nontrivial metric regression, synthetic only."""
import numpy as np
import pytest
from datasets.m6a_interfaces import (OracleSceneState, RGBSpatialEvidence,
                                     DetectedObjectHypothesis, Unknown)
from diagnostics.m6a_oracle_metrics import analyze


@pytest.mark.parametrize('cls', [RGBSpatialEvidence, DetectedObjectHypothesis, Unknown])
def test_visual_and_unknown_never_enter_oracle(cls):
    message=cls(10, 1., 1.35)
    message.validate(1.1)
    with pytest.raises(ValueError, match='cannot feed'):
        message.validate(1.1, consumer='oracle_planner')
    assert message.to_record()['source'] != 'Privileged'


@pytest.mark.parametrize('now', [.99, 1.351, float('nan')])
def test_original_timestamp_fails_closed(now):
    with pytest.raises(ValueError):
        OracleSceneState(10,1.,1.35).validate(now,consumer='oracle_planner')


def test_privileged_schema_and_unknown():
    OracleSceneState(1,1.,1.35).validate(1.1,consumer='oracle_planner')
    OracleSceneState(1,1.,1.35,coordinate='hs_world_x_y_left_z_up_m').validate(1.1,consumer='oracle_planner')
    with pytest.raises(ValueError):
        RGBSpatialEvidence(1,1.,1.35,coordinate='hs_world_x_y_left_z_up_m').validate(1.1)
    assert not Unknown(1,1.,1.35).safety_free
    with pytest.raises(ValueError):
        DetectedObjectHypothesis(1,1.,1.35,position_xyz_m=(1,float('nan'),2)).validate(1.1)


def fixture(completion=.2):
    rows=[dict(source='Privileged',measurement=dict(position=[i*.1,-.2,0],speed=0),
               control_apply_wall=i*.06,sim_time=i*.05,reason=None,command={},deadline_missed=True) for i in range(3)]
    result=dict(scene='static_obstacle',route_completion=completion,collisions=0,lane_invasions=0,
                road_boundary_frames=0,normal_control_expired=0,rejected=[],collision_events=[],lane_events=[],
                wall_hz=16.6,fault_to_brake_s=None,planner_ms=None,plan_sim_age_s=None,plan_wall_age_s=None,
                stage_profile_ms={},log='synthetic',log_sha256='synthetic')
    return rows,result


def test_zero_collision_is_not_task_success_and_clock_metrics():
    rows,result=fixture()
    d=analyze(rows,[],np.array([[0,0,0],[10,0,0]]),result)
    assert not d['route_success'] and d['task_status']=='Partial'
    assert d['planner_failure_or_safety_rejection_rate'] is None
    assert d['control_intervals_over_50ms']==2 and not d['strict_20hz']
    assert d['cross_track_m']['p95']==pytest.approx(.2)
    assert d['sim_step_ms']['p95']==pytest.approx(50)


def test_stale_plan_not_hidden_in_average_and_failure_denominator():
    rows,result=fixture(.99)
    rows[1]['reason']='plan_expired'
    result['rejected']=[{'reason':'plan_expired'}]
    d=analyze(rows,[dict(reason='worker_error',error='bad'),dict(reason=None)],np.array([[0,0,0],[10,0,0]]),result)
    assert d['stale_plan_control_ticks']==1
    assert d['stale_received_plan_count']==1
    assert d['planner_failure_or_safety_rejection_rate']==.5
