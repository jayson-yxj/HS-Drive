"""Actual saved CARLA states: component wall-time profile and swept stress tests."""

import argparse, json, time, cProfile, pstats, io
from pathlib import Path
import numpy as np
from simulator.motion_oracle import (
    OracleState,
    MotionPlanner,
    safety_check as baseline,
    clearance,
)
from simulator.m47_motion import safety_check as improved, continuous_rectangles
from simulator.oracle_driver import Trajectory, footprint, polygons_intersect
from simulator.recorder import atomic_json, file_hash


def main():
    import carla

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m47_latency_profile.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    c = carla.Client("127.0.0.1", 2100)
    c.set_timeout(30)
    w = c.get_world()
    if not w.get_map().name.endswith("Town03"):
        raise ValueError("Town03 required")
    cmap = w.get_map()
    root = Path("reports/m46_routes_verified_artifacts")
    route = np.array(
        json.loads((root / "m46_junction_turn_route.json").read_text())["route"]
    )
    rows = [
        json.loads(x)
        for x in (root / "m46_junction_turn.jsonl").read_text().splitlines()
    ]
    timings = []
    prof = cProfile.Profile()
    prof.enable()
    for version, fn in [("m46", baseline), ("m47", improved)]:
        cache = {}
        planner = MotionPlanner()
        for r in rows[::5]:
            s = OracleState(r["x"], r["y"], r["yaw"], r["v"], r["t"])
            profile = {}
            begin = time.perf_counter()
            clearance(route, s, r["obstacles"])
            route_ms = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            tr, meta = planner.plan(route, s, r["obstacles"])
            planner_ms = (time.perf_counter() - begin) * 1000
            api = [0.0, 0]

            def road(x, y):
                cell = (round(x, 2), round(y, 2))
                if cell not in cache:
                    start = time.perf_counter()
                    cache[cell] = (
                        cmap.get_waypoint(
                            carla.Location(x=float(x), y=float(-y), z=0),
                            project_to_road=False,
                            lane_type=carla.LaneType.Driving,
                        )
                        is not None
                    )
                    api[0] += (time.perf_counter() - start) * 1000
                    api[1] += 1
                return cache[cell]

            start = time.perf_counter()
            reason = fn(
                tr,
                s,
                r["obstacles"],
                half_length=2.395889759,
                half_width=1.081725,
                road_predicate=road,
                **({"profile": profile} if version == "m47" else {}),
            )
            safety_ms = (time.perf_counter() - start) * 1000
            timings.append(
                dict(
                    version=version,
                    t=r["t"],
                    route_query_ms=route_ms,
                    planner_ms=planner_ms,
                    safety_ms=safety_ms,
                    total_ms=route_ms + planner_ms + safety_ms,
                    carla_map_api_ms=api[0],
                    carla_map_api_calls=api[1],
                    reason=reason,
                    **profile,
                )
            )
    prof.disable()
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumtime").print_stats(35)
    profile_path=Path("reports/m47_execution_logs")/(Path(a.output).stem+"_cprofile.txt")
    with profile_path.open("x") as output:output.write(stream.getvalue())
    aggregates = {
        v: {
            k: float(np.quantile([r[k] for r in timings if r["version"] == v], 0.95))
            for k in [
                "route_query_ms",
                "planner_ms",
                "safety_ms",
                "total_ms",
                "carla_map_api_ms",
                "carla_map_api_calls",
            ]
        }
        for v in ["m46", "m47"]
    }
    stress = []
    for speed in [5, 40, 80, 160]:
        for cross in [0.225, 0.475]:
            state = OracleState(0, 0, 0, 0, 1)
            tr = Trajectory(np.zeros((20, 4)), 1, 0.1)
            obj = dict(
                xy=[0, -speed * cross],
                yaw=0.0,
                half_length=0.1,
                half_width=0.1,
                velocity=[0, speed],
            )
            stress.append(
                dict(
                    type="translation",
                    speed=speed,
                    crossing_time=cross,
                    analytic_collision=True,
                    m46_detects=baseline(tr, state, [obj]) is not None,
                    m47_detects=improved(tr, state, [obj]) is not None,
                )
            )
    for rate in [np.pi / 2, np.pi, 2 * np.pi]:
        obj = dict(
            xy=[0, 2.7],
            yaw=0.0,
            half_length=2.0,
            half_width=0.05,
            velocity=[0, 0],
            yaw_rate=float(rate),
        )
        reference = any(
            polygons_intersect(
                footprint(0, 0, 0, 2.6, 1.15), footprint(0, 2.7, rate * t, 2, 0.05)
            )
            for t in np.arange(0, 2, 0.001)
        )
        stress.append(
            dict(
                type="rotation",
                yaw_rate=rate,
                reference_1ms=reference,
                m47_detects=improved(tr, state, [obj]) is not None,
            )
        )
    atomic_json(
        a.output,
        dict(
            kind="offline_recorded_carla_states_with_live_map_api_and_synthetic_collision_stress",
            rows=timings,
            p95=aggregates,
            stress=stress,
            selected_cycle_s=0.05,
            meets_20hz_on_profile=aggregates["m47"]["total_ms"] <= 50,
            source_sha256={
                p: file_hash(p)
                for p in [
                    __file__,
                    "simulator/m47_motion.py",
                    "simulator/motion_oracle.py",
                ]
            },
            limitations=[
                "Profile overhead included; closed-loop latency separately measured.",
                "No camera/network runtime in budget.",
                "Constant-yaw linear rectangles use continuous SAT intervals; rotation uses conservative padding. Not exact rotating mesh swept volume.",
                "Map.get_waypoint is client-side parsed-map lookup; method name alone does not mean RPC.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
