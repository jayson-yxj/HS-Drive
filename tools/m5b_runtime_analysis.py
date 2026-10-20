"""Causal timing attribution from actual control logs; retain all deadline tails."""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    records = []
    for name in [
        "m5b_async_runtime",
        "m5b_async_isolated_runtime",
        "m5b_async_final_runtime",
    ]:
        path = Path("reports", name + ".json")
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        for x in r["results"]:
            rows = [json.loads(l) for l in Path(x["log"]).read_text().splitlines()]
            active = [
                y
                for y in rows
                if y["reason"] is None and y["plan_wall_age"] is not None
            ]
            gaps = np.diff([y["control_apply_wall"] for y in rows])
            worst = np.argsort(gaps)[-5:][::-1]
            tails = []
            for i in worst:
                tails.append(
                    dict(
                        gap_ms=float(gaps[i] * 1000),
                        preceding_frame=rows[i]["frame"],
                        preceding_profile=rows[i]["profile"],
                        current_profile=rows[i + 1]["profile"],
                        preceding_work_ms=rows[i]["cycle_work_ms"],
                        current_reason=rows[i + 1]["reason"],
                    )
                )
            plans = json.loads(Path(x["worker_history"]).read_text())
            expired = sum(
                y["plan_wall_age"] > 0.35 or y["plan_sim_age"] > 0.350001
                for y in active
            )
            speed = np.array([y["measurement"]["long_speed"] for y in rows])
            records.append(
                dict(
                    run=name,
                    scene=x["scene"],
                    tails=tails,
                    expired_plan_normal_control=expired,
                    final_1s_stationary=bool((abs(speed[-20:]) < 0.02).all()),
                    all_degradation_brake=all(
                        y["command"]["brake"] >= 0.99 and y["command"]["throttle"] == 0
                        for y in rows
                        if y["reason"]
                    ),
                    control_interval_50ms_miss_rate=float((gaps > 0.05).mean()),
                    control_interval_55ms_miss_rate=float((gaps > 0.055).mean()),
                    cycle_deadline_miss_rate=x["deadline_miss_rate"],
                    collision=x["collisions"],
                    lane_invasions=x["lane_invasions"],
                    boundary=x["road_boundary_frames"],
                    max_active_age=max(
                        (y["plan_wall_age"] for y in active), default=None
                    ),
                    stage_profile=x["stage_profile_ms"],
                    source_log_sha256=file_hash(x["log"]),
                )
            )
    atomic_json(
        "reports/m5b_runtime_attribution_final.json",
        dict(
            kind="offline_actual_CARLA_profile_analysis",
            records=records,
            changes="cached static extent; crash injection terminate on helper thread with no join in critical path; logging buffered, disk only after loop. Planner algorithm/horizon unchanged.",
            limitations=[
                "Submit timing contains deepcopy/queue enqueue; feeder serialization measured separately in microbenchmark.",
                "CARLA synchronous tick can block sole owner despite worker independence.",
                "No shortened safety horizon or unsafe road fast accept.",
                "Repeated run conditions stated; no hard realtime claim from average Hz.",
            ],
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
