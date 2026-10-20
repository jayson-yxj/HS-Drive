"""Measure immutable job copy/pickle separately; not a replacement for loop timing."""

import copy, json, pickle, time
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    plans = json.loads(
        Path("reports/m5b_async_artifacts/static_obstacle_plans.json").read_text()
    )
    p = plans[len(plans) // 2]
    job = dict(
        scene=p["scene"],
        frame=p["frame"],
        state=p["state"],
        input_wall=p["input_wall"],
        obstacles=[
            dict(
                xy=[0.0, 20.0],
                velocity=[0.0, 0.0],
                yaw=0.0,
                half_length=2.2,
                half_width=1.0,
            )
        ],
        half_length=2.4,
        half_width=1.0,
        z=0.3,
    )
    times = {k: [] for k in ["deepcopy", "pickle", "unpickle"]}
    for _ in range(1000):
        s = time.perf_counter()
        copied = copy.deepcopy(job)
        times["deepcopy"].append((time.perf_counter() - s) * 1e6)
        s = time.perf_counter()
        blob = pickle.dumps(copied)
        times["pickle"].append((time.perf_counter() - s) * 1e6)
        s = time.perf_counter()
        pickle.loads(blob)
        times["unpickle"].append((time.perf_counter() - s) * 1e6)
    atomic_json(
        "reports/m5b_serialization_profile.json",
        dict(
            kind="actual_server_microbenchmark_representative_job",
            bytes=len(blob),
            microseconds={
                k: dict(
                    p50=float(np.quantile(v, 0.5)),
                    p95=float(np.quantile(v, 0.95)),
                    p99=float(np.quantile(v, 0.99)),
                    max=float(max(v)),
                )
                for k, v in times.items()
            },
            scope="Pickle costs only; actual queue scheduling and worker completion age measured in CARLA logs.",
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
