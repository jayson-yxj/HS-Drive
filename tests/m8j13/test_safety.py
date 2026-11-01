import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from simulator.m8j13_safety import PlanStamp,freshness,validate_trajectory,transport_native_path,sanitize_control,SafetyGate,watchdog_reason
from simulator.m8j13_controller import native_controller,native_command


CONFIG=json.loads(Path('configs/m8j13_protocol.json').read_text())
SOURCE=Path(CONFIG['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py'


def stamp():return PlanStamp('S0',10,1.,10.,10.01,10.1,'frozen')


def test_expiry_is_capture_based_in_both_clocks():
    p=stamp()
    assert freshness(p,'S0',11,1.1,10.2) is None
    assert freshness(p,'S0',17,1.35,10.2)=='PlanSimulationExpired'
    assert freshness(replace(p,output_wall_s=10.299),'S0',11,1.1,10.301)=='PlanWallExpired'
    assert freshness(p,'S1',11,1.1,10.2)=='CrossSequencePlan'
    assert freshness(p,'S0',9,1.1,10.2)=='FuturePlan'
    assert freshness(replace(p,input_wall_s=float('nan')),'S0',11,1.1,10.2)=='InvalidPlanClock'


def points():return np.column_stack([np.arange(1,7),np.zeros(6),np.zeros(6),np.full(6,2.)])


def test_native_first_point_is_half_second_not_current_position():
    reason,diag=validate_trajectory(points(),2.,CONFIG['trajectory_checks'])
    assert reason is None and diag['first_point_offset_s']==.5 and diag['initial_anchor']==[0,0]
    assert len(points())==6 and diag['horizon_s']==3.


@pytest.mark.parametrize('kind,expected',[('nan','NonfiniteTrajectory'),('jump','TrajectoryJump'),('speed','TrajectorySpeed'),('acceleration','TrajectoryAcceleration'),('shape','TrajectoryShape'),('curvature','TrajectoryCurvature')])
def test_invalid_trajectories_rejected(kind,expected):
    p=points()
    if kind=='nan':p[2,0]=np.nan
    if kind=='jump':p[2,0]=30
    if kind=='speed':p[0,3]=-1
    if kind=='acceleration':p[:,3]=10
    if kind=='shape':p=p[:4]
    if kind=='curvature':p[1,1]=2
    assert validate_trajectory(p,2.,CONFIG['trajectory_checks'])[0]==expected


def test_transport_preserves_world_points_without_replanning():
    path=np.column_stack([np.ones(15),np.arange(1,16)])
    a=np.eye(4);b=np.eye(4);b[0,3]=.2
    p=transport_native_path(path,a,b)
    np.testing.assert_allclose(p,path-[0,.2])
    np.testing.assert_allclose(np.diff(p,axis=0),np.diff(path,axis=0))


@pytest.mark.parametrize('raw,reason',[
    ({'steer':1.01,'throttle':.2,'brake':0},'ControlOutOfRange'),
    ({'steer':0,'throttle':1.2,'brake':0},'ControlOutOfRange'),
    ({'steer':0,'throttle':.2,'brake':.2},'ConflictingControl'),
    ({'steer':0,'throttle':np.nan,'brake':0},'NonfiniteControl')])
def test_invalid_control_fails_closed_and_latches(raw,reason):
    g=SafetyGate(CONFIG)
    assert g.command(raw,speed=2,sim_s=1,wall_s=2)=={'steer':0,'throttle':0,'brake':1}
    assert g.reason==reason
    assert g.command({'steer':0,'throttle':1,'brake':0},speed=0,sim_s=2,wall_s=3)['brake']==1


def test_speed_cap_does_not_edit_plan():
    g=SafetyGate(CONFIG);raw=dict(steer=.1,throttle=1,brake=0)
    u=g.command(raw,speed=4.5,sim_s=1,wall_s=1)
    assert u['throttle']==0 and u['brake']==.4 and raw['throttle']==1
    assert g.command(raw,speed=5.01,sim_s=2,wall_s=2)['brake']==1
    assert g.reason=='SpeedLimitExceeded'


def test_watchdog_independent_of_model_success():
    h=dict(wall_s=1,command_valid_until_wall_s=1.2)
    assert watchdog_reason(h,1.1) is None
    assert watchdog_reason(h,1.21)=='ControlCommandExpired'
    assert watchdog_reason(h,1.4)=='ControlWatchdogExpired'
    assert watchdog_reason(h,1.01,manual=True)=='ManualEmergencyStop'


@pytest.mark.parametrize('right,sign',[(0,0),(1,1),(-1,-1)])
def test_original_pid_straight_left_right_and_speed_units(right,sign):
    ctl=native_controller(SOURCE)
    path=np.column_stack([np.arange(1,16)*right*.2,np.arange(1,16)])
    u=native_command(ctl,path,np.full(6,2.),1.,[0.,10.])
    assert np.sign(u['native_agent_control']['steer'])==sign
    assert u['native_agent_control']['throttle']>0 and u['native_agent_control']['brake']==0
    stopped=native_command(ctl,path,np.zeros(6),np.float64(1.),[0.,10.])
    assert stopped['native_agent_control']['throttle']==0 and stopped['native_agent_control']['brake']==1
