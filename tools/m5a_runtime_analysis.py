"""Analyze accepted-plan age and fault stopping separately from cached stale results."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json, file_hash


def stats(x):
    return (
        dict(
            p50=float(np.quantile(x, 0.5)),
            p95=float(np.quantile(x, 0.95)),
            p99=float(np.quantile(x, 0.99)),
            max=float(np.max(x)),
            n=len(x),
        )
        if len(x)
        else None
    )


def main():
    output = Path("reports/m5a_closedloop_safety_corrected.json")
    if output.exists():
        raise FileExistsError(output)
    runtime = json.loads(Path("reports/m5a_async_isolated_runtime.json").read_text())
    records = []
    for r in runtime["results"]:
        rows = [json.loads(l) for l in Path(r["log"]).read_text().splitlines()]
        active = [
            x for x in rows if x["reason"] is None and x["plan_wall_age"] is not None
        ]
        plans = json.loads(Path(r["worker_history"]).read_text())
        v = np.array([x["measurement"]["long_speed"] for x in rows])
        t = np.array([x["sim_time"] for x in rows])
        t -= t[0]
        fault = (
            65
            if r["scene"]
            in ["worker_crash", "stalled_worker", "invalid_input", "emergency"]
            else 0
        )
        search_start = (
            fault
            if fault
            else next((i for i in range(len(v)) if abs(v[i]) > 0.2), len(v) - 20)
        )
        stop = next(
            (
                i
                for i in range(search_start, len(v) - 20)
                if (abs(v[i : i + 21]) < 0.02).all()
            ),
            None,
        )
        degrades = [x for x in rows[fault:] if x["reason"]]
        valid_brake = all(
            x["command"]["brake"] >= 0.99 and x["command"]["throttle"] == 0
            for x in degrades
        )
        rec = dict(
            scene=r["scene"],
            kind="actual_CARLA_async_closedloop",
            route_completion=r["route_completion"],
            collisions=r["collisions"],
            lane_invasions=r["lane_invasions"],
            road_boundary_frames=r["road_boundary_frames"],
            active_plan_wall_age=stats([x["plan_wall_age"] for x in active]),
            active_plan_sim_age=stats([x["plan_sim_age"] for x in active]),
            expired_used_for_normal_control=sum(
                x["plan_wall_age"] > 0.35 or x["plan_sim_age"] > 0.350001
                for x in active
            ),
            all_degradation_full_brake=valid_brake,
            stopped_1s=stop is not None,
            final_1s_stable=bool((abs(v[-20:]) < 0.02).all()),
            stop_after_fault_s=(
                None if stop is None or fault == 0 else float(t[stop] - t[fault])
            ),
            final_signed_speed=float(v[-1]),
            final_position=rows[-1]["measurement"]["position"],
            raw_dynamics=r["actual_dynamics"],
            log_sha256=file_hash(r["log"]),
            planner_input_to_output_ms=stats(
                [(p["output_wall"] - p["input_wall"]) * 1000 for p in plans]
            ),
            no_exact_continuous_road_mesh_claim=True,
        )
        records.append(rec)
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for name in ["junction_turn", "worker_crash", "stalled_worker", "invalid_input"]:
        r = next(x for x in runtime["results"] if x["scene"] == name)
        rows = [json.loads(l) for l in Path(r["log"]).read_text().splitlines()]
        t = np.array([x["sim_time"] for x in rows])
        t -= t[0]
        ax[0].plot(t, [x["measurement"]["long_speed"] for x in rows], label=name)
        ax[1].plot(t, [x["command"]["brake"] for x in rows])
        ax[2].plot(
            t,
            [
                (
                    np.nan
                    if x["plan_wall_age"] is None or x["reason"]
                    else x["plan_wall_age"]
                )
                for x in rows
            ],
        )
    ax[0].legend()
    ax[0].set_ylabel("actual signed speed m/s")
    ax[1].set_ylabel("brake command")
    ax[2].set_ylabel("actively used plan age s")
    ax[2].axhline(0.35, color="r", linestyle="--")
    ax[2].set_xlabel("simulation time s")
    fig.tight_layout()
    fig.savefig("reports/m5a_async_faults_corrected.png", dpi=150)
    plt.close(fig)
    atomic_json(
        output,
        dict(
            kind="actual_CARLA_log_analysis",
            runtime="reports/m5a_async_isolated_runtime.json",
            runtime_sha256=file_hash("reports/m5a_async_isolated_runtime.json"),
            records=records,
            source_sha256=file_hash(__file__),
            scope="Oracle route/actor/state input, full six-second worker safety. Fault stopping independent of comfort. Scripted static obstacle only; dynamic dense traffic unverified.",
            status="Partial",
            deadline_definition="runtime deadline_miss_rate = tick completion after wall deadline. control_interval_ms separately measures application-to-application gaps; average Hz does not imply zero jitter.",
            age_note="Runtime includes cached expired plans for diagnosis; active age here excludes all degraded control. Original plan age never reset on resample.",
        ),
    )


if __name__ == "__main__":
    main()
