from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('shapely')
from shapely.geometry import box,Polygon
from datasets.m7a_contracts import *
from diagnostics.m7a_planning import evaluate,select
from simulator.m47_motion import continuous_rectangles


def fixture(boxes=()):
    st=Stamp(10,1.,1.2,WORLD)
    req=PlanningRequest(OracleSceneState(st,(0.,0.,0.,2.),boxes),((0.,0.,0.),(30.,0.,0.)),'synthetic',(30.,0.,0.))
    t=tuple(np.arange(61)*.1)
    c=CandidateTrajectory(st,'A_keep_route',tuple((2*x,0.,0.,2.) for x in t),t,((0.,0.,0.),(30.,0.,0.)))
    return req,c


def test_clear_six_second_candidate_and_no_control_adapter():
    r,c=fixture();e=evaluate(r,c,box(-5,-3,35,3),1.)
    assert e.planning_feasible and e.vehicle_executed_successfully=='NotRun'
    with pytest.raises(TypeError):reject_control_export(c)


def test_rotated_narrow_crossing_cannot_tunnel_between_samples():
    # Cross ego x=.5 at t=.25s, strictly between .2/.3 samples.
    r,c=fixture((Box('fast',(.5,-30.),np.pi/4,.1,.1,(0.,120.)),))
    assert 'predicted_collision' in evaluate(r,c,box(-5,-3,35,3),1.).reasons


def test_road_hole_inside_body_cannot_pass_corner_only_check():
    r,c=fixture()
    road=box(-5,-3,35,3).difference(box(5,-.2,5.1,.2))
    assert 'body_outside_allowed_road' in evaluate(r,c,road,1.).reasons


@pytest.mark.parametrize('now',[.99,1.201,float('nan')])
def test_original_stamp_rejected(now):
    r,c=fixture();assert not evaluate(r,c,box(-5,-3,35,3),now).planning_feasible


def test_zero_speed_advancing_and_short_horizon_rejected():
    r,c=fixture()
    bad=replace(c,points=tuple((x,y,h,0.) for x,y,h,v in c.points))
    assert 'position_velocity_inconsistent' in evaluate(r,bad,box(-5,-3,35,3),1.).reasons
    bad=replace(c,points=c.points[:20],times_s=c.times_s[:20])
    assert 'six_second_horizon_required' in evaluate(r,bad,box(-5,-3,35,3),1.).reasons


def test_visual_cannot_be_oracle_and_unknown_never_free():
    r,c=fixture()
    visual=RGBSpatialEvidence(Stamp(10,1,1.2,EGO),'x',(),'sha')
    with pytest.raises(TypeError):replace(r,scene=visual).validate(1.)
    assert not Unknown('unseen').safety_free
    assert select([])['selected'] is None


def test_evaluator_future_cannot_change_snapshot_candidates():
    r,c=fixture((Box('lead',(20.,0.),0.,2.,1.,(1.,0.)),))
    a=evaluate(r,c,box(-5,-3,35,3),1.,future_reference={'lead':[(6.,(20.,0.))]})
    b=evaluate(r,c,box(-5,-3,35,3),1.,future_reference={'lead':[(6.,(26.,0.))]})
    assert a.reasons==b.reasons
    assert dict(a.metrics)['future_prediction_errors_m']==[6.]
    assert dict(b.metrics)['future_prediction_errors_m']==[0.]


def test_static_collision_beyond_horizon_is_not_bypass_success():
    r,c=fixture((Box('far',(25.,0.),0.,2.,1.),))
    e=evaluate(r,c,box(-5,-3,35,3),1.)
    assert 'guide_path_static_collision' in e.reasons
