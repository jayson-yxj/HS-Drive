"""Synthetic contract, geometry, emergency and future-goal regression."""
import math
import numpy as np
import pytest
from simulator.m6b_motion import StopGoal,goal_route,remaining_distance,LowSpeedController
from simulator.m6b_bypass import offset_path,audit_candidate
from simulator.motion_oracle import MotionPlanner,OracleState
from simulator.m47_motion import safety_check


def test_goal_keeps_explicit_point_and_obstacle_gap():
    r=np.c_[np.arange(31),np.zeros(31),np.zeros(31)]
    g=StopGoal((30,0),0); adapted=goal_route(r,g)
    st=OracleState(0,0,0,0,1)
    _,a=MotionPlanner().plan(adapted,st,[])
    assert a['remaining_m']==30
    obstacle=dict(xy=[15,0],yaw=0,half_length=2,half_width=1,velocity=[0,0])
    _,b=MotionPlanner().plan(adapted,st,[obstacle])
    assert b['remaining_m']==pytest.approx(15-2-2.4-3)
    assert remaining_distance(r,OracleState(31,0,0,0,1),[],2.4,1)==-1


def test_stop_contract_rejects_visual_and_bad_values():
    for g in [StopGoal((1,2),0,source='RGB'),StopGoal((math.nan,0),0),StopGoal((1,2),0,position_tolerance_m=-.5)]:
        with pytest.raises(ValueError):g.validate()


def test_emergency_bypasses_all_comfort_states():
    ctl=LowSpeedController();ctl.mode='HOLD'
    st=OracleState(0,0,0,2,1)
    u=ctl.command(None,st,'worker_dead',remaining_m=10)
    assert u['brake']==1 and u['throttle']==0 and u['mode']=='EMERGENCY'


def test_no_premature_hold_or_residual_braking():
    r=np.c_[np.arange(21),np.zeros(21),np.zeros(21)];st=OracleState(0,0,0,.1,1)
    tr,_=MotionPlanner().plan(r,st,[])
    ctl=LowSpeedController();ctl.effort=-1
    u=ctl.command(tr,st,remaining_m=1)
    assert u['mode']!='HOLD' and u['brake']==0
    st.timestamp_s=1.05;st.speed_mps=.01
    u=ctl.command(tr,st,remaining_m=.2)
    assert u['mode']=='HOLD'
    st.timestamp_s=1.1
    u=ctl.command(tr,st,remaining_m=5)
    assert u['mode']!='HOLD' and u['brake']==0


def test_candidate_full_rectangles_return_and_road_hole():
    r=np.c_[np.arange(101),np.zeros(101),np.zeros(101)]
    p=offset_path(r,3.5)
    np.testing.assert_allclose(p[[0,-1],1],0,atol=1e-10)
    obj=dict(xy=[42,0],yaw=.3,half_length=2,half_width=.9,velocity=[0,0])
    report=audit_candidate(p,[obj],lambda x,y:-2<=y<=5.5,2.4,.9)
    assert report['accepted']
    blocked=dict(obj,xy=[42,3.5])
    assert not audit_candidate(p,[blocked],lambda x,y:True,2.4,.9)['accepted']
    assert not audit_candidate(p,[],lambda x,y:not(39<x<45 and 3<y<4),2.4,.9)['accepted']


def test_original_six_second_safety_rejects_obstacle():
    r=np.c_[np.arange(31),np.zeros(31),np.zeros(31)]
    st=OracleState(0,0,0,1,1);tr,_=MotionPlanner().plan(r,st,[],max_speed=2)
    assert tr.points.shape==(60,4) and tr.dt_s==.1
    obj=dict(xy=[5,0],yaw=.4,half_length=2,half_width=1,velocity=[0,0])
    assert safety_check(tr,st,[obj])=='obstacle_swept_bound'
