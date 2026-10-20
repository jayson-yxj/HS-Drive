"""Calibrated longitudinal adapter and vectorized safety; M46 defaults untouched."""

import json, math, time
from pathlib import Path
import numpy as np
from simulator.motion_oracle import (
    MotionPlanner,
    MotionPolicy,
    validate_kinematics,
    wrap,
)


def continuous_rectangles(a, b, va, vb, ha, hb, yawa, yawb, duration):
    """Exact SAT time intervals for constant-yaw, constant-velocity rectangles.

    Rotation, acceleration and prediction uncertainty require separate bounds.
    """
    axes = np.array(
        [
            [np.cos(yawa), np.sin(yawa)],
            [-np.sin(yawa), np.cos(yawa)],
            [np.cos(yawb), np.sin(yawb)],
            [-np.sin(yawb), np.cos(yawb)],
        ]
    )
    aa = axes[:2]
    bb = axes[2:]
    radius = np.abs(axes @ aa.T) @ np.asarray(ha) + np.abs(axes @ bb.T) @ np.asarray(hb)
    position = axes @ (np.asarray(b) - a)
    velocity = axes @ (np.asarray(vb) - va)
    low = 0.0
    high = duration
    for r, p, v in zip(radius, position, velocity):
        if abs(v) < 1e-10:
            if abs(p) > r:
                return False
        else:
            t0, t1 = sorted(((-r - p) / v, (r - p) / v))
            low = max(low, t0)
            high = min(high, t1)
            if low > high:
                return False
    return True


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


class CalibratedController:
    """Measured feedforward, velocity feedback, explicit COAST/HOLD transitions."""

    def __init__(self, calibration="reports/m47_dynamics_calibration.json"):
        model = json.loads(Path(calibration).read_text())
        self.coeff = np.array(model["high_speed_linear"]["coefficients"])
        self.last_time = None
        self.steer = 0.0
        self.effort = 0.0
        self.mode = "TRACK"
        self.integral = 0.0
        self.hold_entry = None
        self.release_ticks = 0

    @staticmethod
    def coast_distance(v):
        # Measured zero-throttle drag increases with speed; integral v/(a+b*v).
        a = 0.20
        b = 0.40
        v = max(v, 0.0)
        return v / b - a / b**2 * np.log1p(b * v / a)

    def command(
        self,
        trajectory,
        state,
        reason=None,
        dt=0.05,
        clearance_m=None,
        remaining_m=None,
    ):
        if (
            self.last_time is not None
            and not 0 < state.timestamp_s - self.last_time <= 0.25
        ):
            reason = reason or "controller_timeout"
        self.last_time = state.timestamp_s
        if reason or trajectory is None:
            self.mode = "EMERGENCY"
            self.effort = -1.0
            return dict(
                steer=self.steer,
                throttle=0.0,
                brake=1.0,
                reason=reason or "no_trajectory",
                mode=self.mode,
            )
        p = trajectory.points
        v = state.speed_mps
        look = max(3, v * 0.8)
        idx = int(np.argmin(abs(np.linalg.norm(p[:, :2], axis=1) - look)))
        x, y = p[idx, :2]
        requested = -np.clip(math.atan2(5.6 * y, max(x * x + y * y, 0.1)) / 0.6, -1, 1)
        self.steer = float(
            np.clip(requested, self.steer - 1.5 * dt, self.steer + 1.5 * dt)
        )
        if remaining_m is None:
            remaining_m = 1000 if clearance_m is None else clearance_m - 3.0
        if self.mode == "EMERGENCY":
            self.mode = "TRACK"
            self.effort = 0.0
        if self.mode == "HOLD" and remaining_m > 1.5:
            self.mode = "RELEASE"
            self.release_ticks = 0
        if self.mode == "RELEASE":
            self.release_ticks += 1
            if self.release_ticks >= 4:
                self.mode = "TRACK"
        if self.mode == "TRACK" and remaining_m <= max(
            0.8, self.coast_distance(v) + 0.25
        ):
            self.mode = "COAST"
        # Measured coast can settle at nonzero signed rolling speed. Do not
        # wait indefinitely for exactly zero while consuming the stopping gap.
        if self.mode == "COAST" and (v < 0.12 or (remaining_m <= 0.4 and v < 1.2)):
            self.mode = "HOLD"
            self.hold_entry = dict(
                speed=v, remaining_m=remaining_m, t=state.timestamp_s
            )
        if self.mode == "COAST" and remaining_m > self.coast_distance(v) + 2.0:
            self.mode = "TRACK"
        desired = (p[0, 3] - v) / trajectory.dt_s
        error = float(p[min(4, len(p) - 1), 3]) - v
        desired = float(np.clip(desired + 0.3 * error, -1.8, 1.5))
        bias, slope, kt, kb = self.coeff
        zero = bias + slope * v
        throttle = max(0, (desired - zero) / kt)
        brake = max(0, (desired - zero) / kb) if desired < zero else 0.0
        effort = throttle if desired >= zero else -brake
        if v < 2.2 and effort < 0:
            effort = 0.0
        if self.mode == "COAST":
            effort = 0.0
        if self.mode == "HOLD":
            effort = -0.12
        if self.mode == "RELEASE":
            effort = 0.0
        rate = 0.3 if effort > self.effort else 0.6
        self.effort = float(
            np.clip(effort, self.effort - rate * dt, self.effort + rate * dt)
        )
        # No residual service brake across the measured low-speed stiction
        # region; emergency branch above is intentionally exempt.
        if self.mode != "HOLD" and v < 2.2:
            self.effort = max(0, self.effort)
        return dict(
            steer=self.steer,
            throttle=float(np.clip(self.effort, 0, 0.65)),
            brake=float(np.clip(-self.effort, 0, 1)),
            reason=None,
            mode=self.mode,
            requested_acceleration=desired,
        )
