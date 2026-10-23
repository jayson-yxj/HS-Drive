from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
from simulator.m6d_state import SignedMotionState,signed_from_measurement
from simulator.m6d_targets import separate_targets,TargetTracker
from simulator.m6d_motion import SignedGuard,HoldEvidence,SignedController
from simulator.m6b_motion import LowSpeedController
from simulator.motion_oracle import OracleState


def signed(v=0.,t=0.,frame=1,**kw):
    base=SignedMotionState(v,0.,abs(v),v,.871,0.,1,False,False,.12,0.,t,frame)
    return replace(base,**kw)


def measurement(velocity=(2,0,0),yaw=0):
    return dict(velocity=velocity,rotation=[0,0,yaw],position=[0,0,0],applied=dict(gear=1,reverse=False,hand_brake=False,brake=0),t=1.,frame=20)


@pytest.mark.parametrize('vel,yaw,expected', [((2,0,0),0,(2,0)),((-2,0,0),0,(-2,0)),((0,2,0),90,(2,0)),((0,-2,0),0,(0,2)),((.001,0,0),0,(.001,0))])
def test_raw_signed_coordinate_and_route_rate(vel,yaw,expected):
    s=signed_from_measurement(measurement(vel,yaw),[[0,0,0],[10,0,0]],-.4,-.02)
    assert (s.v_forward_mps,s.v_lateral_mps)==pytest.approx(expected,abs=1e-8)
    assert s.route_progress_rate_mps==vel[0]
    assert s.signed_goal_distance_m==-.4
    assert s.gear==1 and not s.reverse_commanded


def test_noise_lateral_and_gear_do_not_fabricate_rollback():
    g=SignedGuard()
    for i in range(10):assert g.check(signed(v=(-1)**i*.01,t=i*.05,frame=i+1),i*.05) is None
    assert g.check(signed(t=1,frame=21,v_lateral_mps=1.,speed_norm_mps=1.,gear=-1),1) is None
    assert g.check(signed(v=-.2,t=1.05,frame=22,gear=1,slope=-.02),1.05)=='RollbackDetected'
    assert g.check(signed(t=1.1,frame=23),1.1)=='RollbackDetected'


def test_persistent_negative_and_stale_protection():
    g=SignedGuard()
    assert g.check(signed(v=-.06,t=0,frame=1),0) is None
    assert g.check(signed(v=-.06,t=.05,frame=2),.05) is None
    assert g.check(signed(v=-.06,t=.1,frame=3),.1)=='RollbackDetected'
    assert SignedGuard().check(signed(),.2)=='StaleSignedState'
    assert SignedGuard().check(signed(t=1),.9)=='StaleSignedState'
    g=SignedGuard();g.check(signed(),0)
    assert g.check(signed(),0)=='NonmonotonicSignedState'


def geom(goal=.871,arc=.871,gap=None):return dict(goal_signed_distance_m=goal,route_remaining_s=arc,obstacle_obb_clearance_m=gap,timestamp=1.,source='Privileged')


def test_targets_do_not_alias_trigger_distance_or_move_goal():
    a=separate_targets(geom(),20,{'trigger_distance':0.000428})
    assert a.active_remaining_m==.871 and a.goal_signed_remaining_m==.871
    b=separate_targets(geom(goal=30,arc=31,gap=3.2),21)
    assert b.goal_signed_remaining_m==30 and b.goal_route_remaining_m==31
    assert b.obstacle_safe_stop_remaining_m==pytest.approx(.2)
    assert b.active_stop_target=='obstacle_constraint'
    c=separate_targets(geom(goal=-.4,arc=-.4,gap=30),22)
    assert c.active_remaining_m==-.4 and c.active_stop_target=='navigation_goal'
    f=separate_targets(geom(),23,fault='worker_dead');assert f.active_remaining_m is None
    tracker=TargetTracker()
    for v in (a,b,c,f):tracker.update(v)
    assert len(tracker.transitions)==4 and tracker.transitions[-1]['current']['timestamp_s']==1


def test_hold_requires_applied_action_position_and_time():
    h=HoldEvidence()
    for i in range(30):assert h.update(signed(t=i*.05,frame=i+1,brake_applied=0),(0,0),0,True,False,True)!='HOLD_STABLE'
    for i in range(21):mode=h.update(signed(t=2+i*.05,frame=50+i),(0,0),0,True,False,True)
    assert mode=='HOLD_STABLE'
    assert h.update(signed(t=3.1,frame=72),(.06,0),.06,True,False,True)!='HOLD_STABLE'
    assert h.update(signed(t=3.2,frame=73),(.06,0),.06,True,False,False)=='DEGRADED_STOP'


def test_f0_equals_c_normal_actuation_and_never_negative_oracle_speed():
    tr=SimpleNamespace(points=np.array([[i*.1,0,0,.6] for i in range(20)]))
    c=LowSpeedController();f=SignedController()
    for i in range(25):
        v=.1 if i>5 else 0;t=i*.05;s=OracleState(0,0,0,v,t)
        target=separate_targets(dict(geom(),timestamp=t),i+1,{'trigger_distance':0.000428})
        u=c.command(tr,s,remaining_m=.871)
        out=f.command(tr,s,remaining_m=.871,signed=signed(v,t,i+1),target=target,position=(0,0))
        for k in ['throttle','brake','steer','mode','reason']:assert out[k]==u[k]
        assert out['mode']!='HOLD' and s.speed_mps>=0
    s=OracleState(0,0,0,.2,2)
    out=f.command(tr,s,remaining_m=.871,signed=signed(-.2,2,50),target=separate_targets(dict(geom(),timestamp=2),50),position=(0,0))
    assert out['brake']==1 and out['throttle']==0 and out['reason']=='RollbackDetected'
    assert s.speed_mps==.2

@pytest.mark.xfail(strict=True,reason='Known M6-D NoGo: actual intermittent backward motion resets duration guard; no post-result threshold tuning')
def test_known_intermittent_rollback_gap():
    g=SignedGuard();observed=[]
    # Actual warm-release pattern alternates negative motion with positive speed.
    for i,v in enumerate([-.075,.02,-.081,.02,-.078,.02,-.067]):
        observed.append(g.check(signed(v=v,t=i*.05,frame=i+1),i*.05))
    assert 'RollbackDetected' in observed
