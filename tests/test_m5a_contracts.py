import numpy as np
import pytest
from simulator.m5a_async import validate_result
from simulator.oracle_driver import Trajectory


def result():
    return dict(
        scene="s",
        frame=10,
        state={"timestamp_s": 1.0},
        input_wall=100.0,
        start_wall=100.01,
        output_wall=100.1,
        trajectory=Trajectory(np.zeros((60, 4)), 1.0, 0.1),
        error=None,
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"scene": "x"}, "scene_mismatch"),
        ({"frame": 20}, "future_frame"),
        ({"output_wall": 99}, "invalid_time"),
        ({"error": "crash"}, "worker_error"),
    ],
)
def test_invalid_worker_results_rejected(change, expected):
    r = result()
    r.update(change)
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == expected


def test_both_sim_and_wall_age_expire_without_timestamp_reset():
    r = result()
    assert validate_result(r, "s", 11, 1.1, 100.4, 0.35) == "plan_expired"
    assert validate_result(r, "s", 11, 1.4, 100.1, 0.35) == "plan_expired"
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) is None
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35, r) == "out_of_order"
    r["trajectory"].timestamp_s = 1.1
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "generation_time_mismatch"


@pytest.mark.parametrize("r", [{}, None, {"frame": 1}])
def test_malformed_is_safe(r):
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "malformed_result"


def test_short_horizon_and_future_output_rejected():
    r = result()
    r["trajectory"].points = np.zeros((20, 4))
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "invalid_trajectory"
    r = result()
    r["output_wall"] = 100.2
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "invalid_time"


def test_nan_generation_and_horizon_fail_closed():
    r = result()
    r["trajectory"].timestamp_s = float("nan")
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "invalid_time"
    r = result()
    r["trajectory"].dt_s = float("nan")
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "invalid_time"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "10"])
def test_frame_must_be_nonnegative_integer(value):
    r = result()
    r["frame"] = value
    assert validate_result(r, "s", 11, 1.1, 100.1, 0.35) == "invalid_frame"
