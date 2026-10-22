import json,math
from pathlib import Path
import numpy as np
import pytest
from simulator.m6c_envelope import StoppingEnvelope
from simulator.m6c_geometry import stopping_geometry,polygon_gap
from simulator.m6c_motion import EnvelopeController
from simulator.m6b_motion import StopGoal,goal_route
from simulator.motion_oracle import OracleState,MotionPlanner
from simulator.oracle_driver import footprint


def test_current_signed_geometry_not_endpoint_clamp():
    route=np.c_[np.arange(11),np.zeros(11),np.zeros(11)];s=OracleState(11,.2,0,.1,3)
    e=dict(xy=[11,.2],yaw=0,half_length=2.4,half_width=1)
    g=stopping_geometry(route,s,StopGoal((10,0),0),e,[])
    assert g['route_remaining_s']==-1 and g['goal_signed_distance_m']==-1
    assert g['goal_lateral_error_m']==pytest.approx(-.2) and g['timestamp']==3


def test_obb_gap_once_not_actor_arc_or_double_margin():
    route=np.c_[np.arange(31),np.zeros(31),np.zeros(31)];s=OracleState(0,0,0,1,1)
    e=dict(xy=[.03,0],yaw=0,half_length=2.4,half_width=1)
    o=dict(xy=[10,0],yaw=0,half_length=2,half_width=1,velocity=[0,0],actor_id=1)
    g=stopping_geometry(route,s,StopGoal((30,0),0),e,[o])
    assert g['obstacle_obb_clearance_m']==pytest.approx(5.57)
    assert g['available_clearance_to_safe_stop']==pytest.approx(2.57)
    assert polygon_gap(footprint(0,0,.7,2,1),footprint(0,0,-.2,2,1))==0


def test_envelope_invalid_and_downhill_unknown():
    model=json.loads(Path('configs/m6c/envelope_model.json').read_text());e=StoppingEnvelope(model)
    assert not e.estimate(1,-.02,'coast').supported
    assert not e.estimate(6,0).supported
    assert not e.estimate(float('nan'),0).supported
    assert e.estimate(2,0,variant='C').distance_m>=e.estimate(2,0,variant='B').distance_m


def test_insufficient_margin_does_not_delay_emergency():
    e=StoppingEnvelope(json.loads(Path('configs/m6c/envelope_model.json').read_text()));c=EnvelopeController(e)
    r=np.c_[np.arange(31),np.zeros(31),np.zeros(31)];s=OracleState(0,0,0,2,1)
    tr,_=MotionPlanner().plan(r,s,[])
    g=dict(available_clearance_to_safe_stop=.01,obstacle_obb_clearance_m=3.01)
    u=c.command(tr,s,remaining_m=.01,geometry=g,grade=0)
    assert u['brake']==1 and u['throttle']==0 and u['reason']=='InsufficientStoppingMargin'
    assert not u['envelope_decision']['stopping_feasible']


def test_goal_adapter_keeps_frozen_one_m_reserve():
    r=np.c_[np.arange(10),np.zeros(10),np.zeros(10)]
    _,meta=MotionPlanner().plan(goal_route(r,StopGoal((9,0),0)),OracleState(0,0,0,0,1),[])
    assert meta['remaining_m']==9
