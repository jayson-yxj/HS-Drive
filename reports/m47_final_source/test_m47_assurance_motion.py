import numpy as np
import pytest
from datasets.m47_assurance import quarantine_conflicts
from datasets.occupancy_labels import build_observed
from geometry.ray_tracing import VoxelGrid
from simulator.m47_motion import (
    continuous_rectangles,
    safety_check,
    CalibratedController,
)
from simulator.motion_oracle import MotionPlanner, OracleState, Trajectory


def test_quarantine_only_measured_conflicting_free_cells():
    grid = VoxelGrid((0, 5), (-1, 1), (0, 1), 1.0)
    labels, _ = build_observed(
        np.array([[0.1, 0.5, 0.5]]), np.array([[4.5, 0.5, 0.5]]), grid
    )
    state = labels["occupancy_state"].copy()
    out = quarantine_conflicts(labels, [3, 9])
    assert out["occupancy_state"].flat[3] == 0
    assert out["occupancy_state"].flat[9] == 2
    np.testing.assert_array_equal(labels["occupancy_state"], state)
    assert (out["occupancy_state"] != state).sum() == 1


@pytest.mark.parametrize("speed", [5, 40, 80, 160])
def test_continuous_translation_catches_between_sample_crossing(speed):
    assert continuous_rectangles(
        np.array([0.0, 0.0]),
        np.array([0.0, -speed * 0.025]),
        np.zeros(2),
        np.array([0.0, speed]),
        [2.6, 1.15],
        [0.1, 0.1],
        0,
        0,
        0.05,
    )


def test_parallel_separated_sweeps_are_clear():
    assert not continuous_rectangles(
        np.zeros(2),
        np.array([0.0, 5.0]),
        np.ones(2),
        np.ones(2),
        [1, 1],
        [1, 1],
        0,
        0,
        1,
    )


def test_rotation_bound_detects_sweeping_bar():
    state = OracleState(0, 0, 0, 0, 1)
    tr = Trajectory(np.zeros((20, 4)), 1, 0.1)
    obs = dict(
        xy=[0, 2.7],
        yaw=0.0,
        yaw_rate=np.pi / 2,
        half_length=2.0,
        half_width=0.05,
        velocity=[0, 0],
    )
    assert safety_check(tr, state, [obs]) == "obstacle_swept_bound"


def test_timeout_nan_and_emergency_still_fail_closed():
    st = OracleState(0, 0, 0, 0, 1)
    tr = Trajectory(np.zeros((10, 4)), 0.5, 0.1)
    assert safety_check(tr, st, []) == "trajectory_timeout"
    tr.timestamp_s = 1
    tr.points[0, 0] = np.nan
    assert safety_check(tr, st, []) == "invalid_trajectory"
    c = CalibratedController()
    r = c.command(None, st)
    assert r["brake"] == 1 and r["throttle"] == 0


def test_low_speed_hold_is_explicit_and_emergency_exempt():
    st = OracleState(0, 0, 0, 0.1, 1)
    tr = Trajectory(np.tile([0.005, 0, 0, 0], (10, 1)), 1, 0.1)
    c = CalibratedController()
    r = c.command(tr, st, remaining_m=0.4)
    assert r["mode"] == "HOLD" and c.hold_entry["speed"] == 0.1
    st.timestamp_s += 0.05
    r = c.command(tr, st, "emergency_command", remaining_m=0.4)
    assert r["brake"] == 1
