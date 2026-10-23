from types import SimpleNamespace
import pytest
from simulator.m6e_actuation import cycle_command,apply_variant


def test_reference_is_not_physical_velocity_override_and_jerk_not_claimed_enforced():
    kind,command,reference=cycle_command('C','cold_run',.2)
    assert kind=='ackermann' and command['speed']==pytest.approx(.1)
    assert command['acceleration']==.5 and command['jerk']==1.
    assert reference==command['speed']
    assert cycle_command('C','normal_stop',2.)[1]['speed']==0


def test_frozen_raw_controls_and_explicit_b_initialization():
    _,a,_=cycle_command('A','cold_run',0);_,b,_=cycle_command('B','cold_run',0)
    assert a['throttle']==b['throttle']==.15 and not a['manual_gear_shift']
    assert b['manual_gear_shift'] and b['gear']==1 and not b['hand_brake']
    assert cycle_command('A','normal_stop',0)[1]['brake']==.01
    assert cycle_command('B','hold_initial',0)[1]['hand_brake']


def test_variant_only_declared_field_family():
    wheels=[SimpleNamespace(max_brake_torque=700.,max_handbrake_torque=1400.,tire_friction=3.5) for _ in range(4)]
    p=SimpleNamespace(wheels=wheels,mass=1845.,clutch_strength=10.)
    applied=[];v=SimpleNamespace(get_physics_control=lambda:p,apply_physics_control=applied.append)
    apply_variant(v,dict(family='wheel_max_brake_torque',scale=.5))
    assert all(w.max_brake_torque==350 and w.max_handbrake_torque==1400 and w.tire_friction==3.5 for w in p.wheels)
    assert p.mass==1845 and p.clutch_strength==10 and applied==[p]
    with pytest.raises(ValueError):apply_variant(v,dict(family='transmission',scale=.5))
