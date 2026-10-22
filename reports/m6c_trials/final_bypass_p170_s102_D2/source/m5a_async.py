"""Spawned planner process; it owns an offline map, NEVER a world/tick/client."""

import copy, queue, time, multiprocessing as mp, traceback
import numpy as np
from simulator.motion_oracle import MotionPlanner, OracleState
from simulator.m47_motion import safety_check
from simulator.m48_schedule import TimedPlan


def _worker(requests, results, xodr, route, delay):
    import carla

    cmap = carla.Map("m5a_offline_map", xodr)
    planner = MotionPlanner()
    cache = {}
    while True:
        job = requests.get()
        if job is None:
            return
        started = time.perf_counter()
        try:
            job_delay = job.get("delay", delay)
            if job_delay:
                time.sleep(job_delay)
            state = OracleState(**job["state"])
            trajectory, meta = planner.plan(
                route,
                state,
                job["obstacles"],
                max_speed=2.5,
                half_length=job["half_length"],
                half_width=job["half_width"],
            )
            z = job["z"]
            calls = 0

            def road(x, y):
                nonlocal calls
                if len(cache) > 100000:
                    cache.clear()
                key = (float(x), float(y), z)
                if key not in cache:
                    calls += 1
                    cache[key] = (
                        cmap.get_waypoint(
                            carla.Location(x=float(x), y=-float(y), z=z),
                            project_to_road=False,
                            lane_type=carla.LaneType.Driving,
                        )
                        is not None
                    )
                return cache[key]

            reason = safety_check(
                trajectory,
                state,
                job["obstacles"],
                half_length=job["half_length"],
                half_width=job["half_width"],
                road_predicate=road,
            )
            result = dict(
                scene=job["scene"],
                frame=job["frame"],
                state=job["state"],
                input_wall=job["input_wall"],
                start_wall=started,
                output_wall=time.perf_counter(),
                trajectory=trajectory,
                meta=meta,
                reason=reason,
                map_calls=calls,
                error=None,
            )
        except BaseException:
            result = dict(
                scene=job["scene"],
                frame=job["frame"],
                state=job["state"],
                input_wall=job["input_wall"],
                start_wall=started,
                output_wall=time.perf_counter(),
                trajectory=None,
                meta={},
                reason="worker_error",
                error=traceback.format_exc(),
            )
        try:
            results.put_nowait(result)
        except queue.Full:
            try:
                results.get_nowait()
            except queue.Empty:
                pass
            results.put(result)


class AsyncPlanner:
    def __init__(self, xodr, route, *, delay=0.0, max_age=0.35):
        context = mp.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=2)
        self.worker = context.Process(
            target=_worker,
            args=(self.requests, self.results, xodr, np.asarray(route), delay),
            daemon=True,
        )
        self.worker.start()
        self.max_age = max_age
        self.latest = None
        self.history = []
        self.rejected = []

    def submit(self, job):
        if not self.worker.is_alive():
            return False
        try:
            self.requests.put_nowait(copy.deepcopy(job))
            return True
        except queue.Full:
            return False

    def poll(self, scene, frame, state, now):
        while True:
            try:
                r = self.results.get_nowait()
            except queue.Empty:
                break
            if not isinstance(r, dict):
                self.rejected.append(dict(frame=None, reason="malformed_result"))
                self.latest = None
                continue
            self.history.append(
                {k: v for k, v in r.items() if k not in ["trajectory", "meta"]}
            )
            why = validate_result(
                r, scene, frame, state.timestamp_s, now, self.max_age, self.latest
            )
            if why:
                self.rejected.append(dict(frame=r.get("frame"), reason=why))
                if why not in ["out_of_order", "plan_expired"]:
                    self.latest = None
                continue
            self.latest = r
        if not self.worker.is_alive():
            return None, "worker_dead", None
        r = self.latest
        if r is None:
            return None, "no_valid_plan", None
        why = validate_result(
            r, scene, frame, state.timestamp_s, now, self.max_age, None
        )
        if why:
            return None, why, r
        if r.get("reason"):
            return None, r["reason"], r
        buffer = TimedPlan(max_age=self.max_age)
        buffer.store(r["trajectory"], OracleState(**r["state"]))
        tr, why = buffer.sample(state)
        return tr, why, r

    def close(self):
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=0.5)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=2)
        self.requests.close()
        self.results.close()


def validate_result(r, scene, frame, timestamp, now, max_age, latest=None):
    try:
        return _validate_result(r, scene, frame, timestamp, now, max_age, latest)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return "malformed_result"


def _validate_result(r, scene, frame, timestamp, now, max_age, latest=None):
    if not np.isfinite(
        [
            timestamp,
            now,
            r["start_wall"],
            r["input_wall"],
            r["output_wall"],
            r["state"]["timestamp_s"],
        ]
    ).all():
        return "invalid_time"
    if not isinstance(r["frame"], (int, np.integer)) or r["frame"] < 0:
        return "invalid_frame"
    if r["scene"] != scene:
        return "scene_mismatch"
    if r["frame"] > frame:
        return "future_frame"
    if latest is not None and r["frame"] <= latest["frame"]:
        return "out_of_order"
    age = timestamp - r["state"]["timestamp_s"]
    wall_age = now - r["input_wall"]
    if (
        not np.isfinite([age, wall_age, r["output_wall"]]).all()
        or age < -1e-6
        or wall_age < 0
    ):
        return "invalid_time"
    if age > max_age or wall_age > max_age:
        return "plan_expired"
    if r["output_wall"] > now + 1e-6:
        return "invalid_time"
    if r["output_wall"] < r["start_wall"] or r["start_wall"] < r["input_wall"]:
        return "invalid_time"
    if r.get("error"):
        return "worker_error"
    if r["trajectory"] is None:
        return "no_trajectory"
    if not np.isfinite([r["trajectory"].timestamp_s, r["trajectory"].dt_s]).all():
        return "invalid_time"
    if abs(r["trajectory"].timestamp_s - r["state"]["timestamp_s"]) > 1e-6:
        return "generation_time_mismatch"
    if (
        r["trajectory"].points.shape != (60, 4)
        or not np.isfinite(r["trajectory"].points).all()
    ):
        return "invalid_trajectory"
    if abs(r["trajectory"].dt_s - 0.1) > 1e-8:
        return "invalid_horizon"
    return None
