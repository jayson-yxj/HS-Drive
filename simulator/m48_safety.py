"""M48 safety: unchanged swept obstacle check, batched certified road queries."""
import time
import numpy as np
from simulator.motion_oracle import MotionPolicy,validate_kinematics,wrap
from simulator.m47_motion import continuous_rectangles

def safety_check(
    trajectory,
    state,
    obstacles,
    *,
    half_length=2.4,
    half_width=1.0,
    policy=MotionPolicy(),
    road_predicate=None,
    profile=None,
):
    times = {} if profile is None else profile
    if trajectory is None:
        return "no_trajectory"
    if not np.isfinite(trajectory.timestamp_s):
        return "invalid_trajectory"
    if not 0 <= state.timestamp_s - trajectory.timestamp_s <= 0.2:
        return "trajectory_timeout"
    bad = validate_kinematics(trajectory, state, policy)
    if bad:
        return bad
    p = trajectory.points
    if (p[:, 3] > 5.6).any():
        return "speed_limit"
    start = time.perf_counter()
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    xy = p[:, :2] @ np.array([[c, s], [-s, c]]) + [state.x, state.y]
    poses = np.vstack(([state.x, state.y, state.yaw], np.c_[xy, state.yaw + p[:, 2]]))
    samples = []
    ts = []
    for i, (a, b) in enumerate(zip(poses[:-1], poses[1:])):
        steps = max(
            2,
            int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / policy.safety_step)),
            int(np.ceil(abs(wrap(b[2] - a[2])) / 0.025)),
        )
        f = np.linspace(0, 1, steps, endpoint=False)
        samples.extend(
            np.c_[a[:2] + f[:, None] * (b[:2] - a[:2]), a[2] + f * wrap(b[2] - a[2])]
        )
        ts.extend((i + f) * trajectory.dt_s)
    samples = np.vstack((samples, poses[-1]))
    ts = np.r_[ts, len(p) * trajectory.dt_s]
    hl = half_length + policy.footprint_margin
    hw = half_width + policy.lateral_margin
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    co = np.cos(samples[:, 2])
    si = np.sin(samples[:, 2])
    corners = np.stack(
        (
            co[:, None] * local[:, 0] - si[:, None] * local[:, 1],
            si[:, None] * local[:, 0] + co[:, None] * local[:, 1],
        ),
        -1,
    ) + samples[:, :2, None].transpose(0, 2, 1)
    times["footprint_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    if road_predicate is not None:
        if hasattr(road_predicate, "batch"):
            if not road_predicate.batch(corners.reshape(-1, 2)).all():
                times["road_ms"] = (time.perf_counter() - start) * 1000
                return "road_boundary"
        else:
            for x, y in corners.reshape(-1, 2):
                if not road_predicate(x, y):
                    times["road_ms"] = (time.perf_counter() - start) * 1000
                    return "road_boundary"
    times["road_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for obj in obstacles:
        velocity = np.asarray(obj.get("velocity", [0, 0]))
        omega = float(obj.get("yaw_rate", 0))
        for i in range(len(samples) - 1):
            dt = ts[i + 1] - ts[i]
            a = samples[i]
            b = samples[i + 1]
            ego_mid = a[2] + wrap(b[2] - a[2]) / 2
            other_mid = obj["yaw"] + omega * (ts[i] + dt / 2)
            # Bound rotational corner displacement over interval. Translation
            # is continuous SAT; padding makes rotation conservative.
            er = np.hypot(hl, hw) * abs(wrap(b[2] - a[2])) / 2
            obr = np.hypot(obj["half_length"], obj["half_width"]) * abs(omega) * dt / 2
            if continuous_rectangles(
                a[:2],
                np.asarray(obj["xy"]) + ts[i] * velocity,
                (b[:2] - a[:2]) / dt,
                velocity,
                [hl + er, hw + er],
                [obj["half_length"] + obr, obj["half_width"] + obr],
                ego_mid,
                other_mid,
                dt,
            ):
                times["collision_ms"] = (time.perf_counter() - start) * 1000
                return "obstacle_swept_bound"
    times["collision_ms"] = (time.perf_counter() - start) * 1000
    return None
