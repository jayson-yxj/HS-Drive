import numpy as np
import pytest
from simulator.motion_oracle import *


def route():
    return np.c_[np.arange(100), np.zeros(100), np.zeros(100)]


def state(v=0):
    return OracleState(0, 0, 0, v, 1.0)


def obstacle(g):
    return dict(
        xy=[g + 4.8, 0], yaw=0.0, half_length=2.4, half_width=1.0, velocity=[0.0, 0.0]
    )


def test_stopped_positions_do_not_advance():
    t, m = MotionPlanner().plan(route(), state(), [obstacle(3)])
    assert np.array_equal(t.points, np.zeros_like(t.points))


@pytest.mark.parametrize("v,g", [(0, 20), (2, 10), (4, 15), (4, 8), (5, 20)])
def test_motion_integral_acceleration_jerk_and_stopping(v, g):
    policy = MotionPolicy()
    t, m = MotionPlanner(policy).plan(route(), state(v), [obstacle(g)], max_speed=5)
    velocities = np.r_[v, t.points[:, 3]]
    np.testing.assert_allclose(
        np.diff(np.r_[0, t.points[:, 0]]),
        (velocities[:-1] + velocities[1:]) * 0.05,
        atol=1e-8,
    )
    a = np.diff(velocities) / 0.1
    assert (
        a.min() >= -policy.comfortable_decel - 1e-7
        and a.max() <= policy.acceleration + 1e-7
    )
    assert np.abs(np.diff(np.r_[0, a])).max() / 0.1 <= policy.jerk + 1e-7
    assert m["kinematic_failure"] is None


def test_multi_actor_gap_and_safety_stationary_crossing():
    lead = obstacle(20)
    near = obstacle(10)
    side = obstacle(2)
    side["xy"][1] = 5
    found = clearance(route(), state(), [lead, near, side])
    assert found["gap_m"] == pytest.approx(10)
    t, _ = MotionPlanner().plan(route(), state(), [obstacle(3)])
    crossing = dict(
        xy=[0, 4], yaw=np.pi / 2, half_length=2, half_width=1, velocity=[0, -2]
    )
    assert safety_check(t, state(), [crossing]) == "obstacle_footprint"


def test_inconsistent_and_invalid_trajectory_fail_closed_and_emergency():
    t, _ = MotionPlanner().plan(route(), state(), [obstacle(3)])
    t.points[:, 0] = 1
    assert safety_check(t, state(), []) == "position_velocity_inconsistent"
    assert safety_check(None, state(), []) == "no_trajectory"
    control = MotionController()
    u = control.command(t, state(), "emergency_command")
    assert u["brake"] == 1 and u["throttle"] == 0


def test_frozen_positions_with_positive_speed_are_rejected():
    t = Trajectory(np.tile([0.0, 0.0, 0.0, 4.0], (10, 1)), 1.0, 0.1)
    assert safety_check(t, state(4), []) == "position_velocity_inconsistent"


def test_common_remaining_distance_holds_both_route_and_obstacle_stop():
    tr, _ = MotionPlanner().plan(route(), state(0.2), [obstacle(3)])
    a = MotionController().command(tr, state(0.2), remaining_m=0.2)
    b = MotionController().command(tr, state(0.2), clearance_m=3.2)
    assert a == b and a["brake"] > 0 and a["throttle"] == 0


def test_controller_timeout_preserves_emergency_brake():
    t, _ = MotionPlanner().plan(route(), state(), [])
    c = MotionController()
    c.command(t, state())
    later = state()
    later.timestamp_s += 1.0
    result = c.command(t, later)
    assert result["reason"] == "controller_timeout" and result["brake"] == 1.0
