import numpy as np

from tools.m47_hold_replay import replay_hold


def test_replay_recovers_missing_mode_and_persistent_hold_then_release():
    def row(t, speed, remaining, end_x, reason=None):
        points = np.zeros((20, 4))
        points[:, 0] = np.linspace(0, end_x, 20)
        return dict(
            measurement={"t": 100 + t}, x=0, y=0, yaw=0, v=speed,
            trajectory=points.tolist(), trajectory_dt=0.1,
            safety_reason=reason, plan_remaining=remaining,
            t=t, gap=remaining + 3,
            command={"brake": 0.02},  # baseline does not log its internal mode
        )

    rows = [
        row(0, 0.5, 1, 2),
        row(0.05, 0.2, 0.2, 0.5),
        row(0.10, 0.1, 0.1, 0.5, "emergency_command"),
        row(0.15, 0.0, 3, 3),
    ]
    active, entry = replay_hold(rows)
    assert active == [False, True, True, False]
    assert entry["t"] == 0.05
    assert entry["speed"] == 0.2
    assert entry["gap"] == 3.2
