"""Aggregate every runtime attempt; tails/failures never hidden by averaging."""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_runtime_stress.json")
    if dest.exists():
        raise FileExistsError(dest)
    paths = sorted(Path("reports").glob("m5c_runtime_*trial*.json")) + list(
        Path("reports").glob("m5c_runtime_isolated*.json")
    )
    records = []
    for path in paths:
        report = json.loads(path.read_text())
        for r in report["results"]:
            rows = [json.loads(s) for s in Path(r["log"]).read_text().splitlines()]
            tops = []
            for i in np.argsort([a["cycle_work_ms"] for a in rows])[-5:][::-1]:
                a = rows[i]
                tops.append(
                    dict(
                        frame=a["frame"],
                        cycle_work_ms=a["cycle_work_ms"],
                        profile=a["profile"],
                        reason=a["reason"],
                    )
                )
            interval = r["control_interval_ms"]
            gate = (
                interval["p99"] <= 55
                and interval["max"] <= 100
                and r["deadline_miss_rate"] <= 0.001
                and r["normal_control_expired"] == 0
            )
            records.append(
                dict(
                    report=str(path),
                    report_sha256=file_hash(path),
                    **r,
                    cycle_gate_passed=gate,
                    top_work_cycles=tops,
                )
            )
    atomic_json(
        dest,
        dict(
            kind="actual_CARLA_wallpaced_repeated_and_GPU_inference_stress",
            records=records,
            source_sha256=file_hash(__file__),
            strict_cycle_gate="P99<=55ms,Max<=100ms,tickdeadline miss<=0.1%,expired normal control0; each run individually; no gate reduction",
            cycle_status=(
                "Passed_measured_runs"
                if records and all(r["cycle_gate_passed"] for r in records)
                else "Partial"
            ),
            oracle_safety_status="Partial",
            pure_visual_safety="NotValidated",
            horizon_seconds=6,
            planner_hz=5,
            controller_hz=20,
            history_warning="First named isolated1 trial overlaps CPU label generation, so is CPU processing load, not a truly isolated run. Later trial descriptions identify isolation/pressure. All results retained.",
            limits="Oracle only; inference-pressure process does not feed visual predictions to driver. Hold-after-route soak mostly stationary after route; moving repetitions also reported. No hard real-time or long-duration fleet guarantee. Fault latency starts injection at tick65 and ends apply_control brake call; physics response occurs subsequent tick.",
        ),
    )


if __name__ == "__main__":
    main()
