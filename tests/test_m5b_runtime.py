from types import SimpleNamespace as S
import pytest
from simulator.m5b_runtime import snapshot_state_measurement


def vec(x, y, z):
    return S(x=x, y=y, z=z)


def test_single_snapshot_measurement_and_missing_actor():
    tf = S(
        location=vec(2, 3, 0.2),
        rotation=S(roll=0, pitch=0, yaw=90),
        get_forward_vector=lambda: vec(0, 1, 0),
    )
    actor = S(
        get_transform=lambda: tf,
        get_velocity=lambda: vec(0, 2, 0),
        get_acceleration=lambda: vec(0, -1, 0),
    )
    snap = S(
        frame=123,
        timestamp=S(elapsed_seconds=10.0),
        find=lambda x: actor if x == 1 else None,
    )
    control = S(throttle=0.1, brake=0, steer=0, hand_brake=False, reverse=False, gear=1)
    state, m = snapshot_state_measurement(snap, 1, control)
    assert (
        state.x == 2
        and state.y == -3
        and state.speed_mps == 2
        and state.timestamp_s == 10
    )
    assert (
        m["frame"] == 123 and m["long_speed"] == 2 and m["long_acceleration_api"] == -1
    )
    with pytest.raises(ValueError):
        snapshot_state_measurement(snap, 2, control)
