"""Empirical stopping reachability from preserved raw trials; no new PID/probes."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json, file_hash


def main():
    output = Path("reports/m5a_stopping_reachability.json")
    if output.exists():
        raise FileExistsError(output)
    sources = [
        "reports/m47_vehicle_dynamics.json",
        "reports/m48_dynamics_feasibility.json",
        "reports/m48_low_speed_followup.json",
    ]
    paths = []
    for f in sources:
        report = json.loads(Path(f).read_text())
        for r in report.get("results", report.get("records", [])):
            if "log" in r:
                paths.append(r["log"])
    # M47 report uses trials in some versions. The verified v3 raw trials are explicit.
    paths += list(map(str, Path("reports/m47_dynamics_v3_artifacts").glob("*.jsonl")))
    paths = sorted(set(p for p in paths if "warmup" not in p))
    records = []
    curves = []
    for p in paths:
        rows = [json.loads(l) for l in Path(p).read_text().splitlines()]
        t = np.array([r["t"] for r in rows])
        t -= t[0]
        v = np.array([r["long_speed"] for r in rows])
        xy = np.array([r["position"][:2] for r in rows])
        travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        acc = np.diff(v) / np.diff(t)
        jerk = np.diff(acc) / np.diff(t)[1:]
        stable = next(
            (i for i in range(len(v) - 20) if (abs(v[i : i + 21]) < 0.02).all()), None
        )
        comfort = bool(len(acc) and -acc.min() <= 3 and abs(jerk).max() <= 6)
        command = rows[0].get("command", {})
        records.append(
            dict(
                log=p,
                sha256=file_hash(p),
                initial_actual_speed=float(v[0]),
                initial_command=command,
                stopped_1s=stable is not None,
                stop_time=None if stable is None else float(t[stable]),
                stop_distance=None if stable is None else float(travel[stable]),
                raw_max_deceleration=float(max(0, -acc.min())),
                raw_jerk_peak=float(abs(jerk).max()),
                reverse_ticks=int((v < -0.01).sum()),
                comfort_full_trace=comfort,
                scope="sampled observed execution, no interpolation guarantee",
            )
        )
        if any(
            k in Path(p).stem
            for k in ["coast_v2.0", "brake0.02_v2.0", "coast_to_0.02", "coast_to_0.2"]
        ):
            curves.append((p, t, v, travel, acc, jerk))
    # Exact triangular/trapezoid ideal jerk-limited lower distance, a(0)=a(stop)=0.
    # Physics need not realize it: this is a necessary reference, NOT certified reachable set.
    envelope = []
    for speed in [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 4, 5]:
        A = 3.0
        J = 6.0
        T = 2 * np.sqrt(speed / J) if speed <= A * A / J else speed / A + A / J
        ideal = 0.5 * speed * T
        nearest = [
            r
            for r in records
            if abs(r["initial_actual_speed"] - speed) <= max(0.015, 0.06 * speed)
        ]
        stops = [r for r in nearest if r["stopped_1s"]]
        envelope.append(
            dict(
                speed=speed,
                ideal_jerk_limited_distance=ideal,
                ideal_stop_time=T,
                delay_distance_at_age_035=speed * 0.35,
                empirical_stopped_runs=len(stops),
                empirical_comfort_runs=sum(r["comfort_full_trace"] for r in stops),
                observed_stop_distance_range=(
                    [
                        min(r["stop_distance"] for r in stops),
                        max(r["stop_distance"] for r in stops),
                    ]
                    if stops
                    else None
                ),
            )
        )
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for p, t, v, d, a, j in curves[:10]:
        name = Path(p).stem
        ax[0].plot(t, v, label=name)
        ax[1].plot(t[1:], a)
        ax[2].plot(t[2:], j)
    for a, l in zip(ax, ["signed speed m/s", "raw acceleration m/s2", "raw jerk m/s3"]):
        a.set_ylabel(l)
        a.grid()
    ax[0].legend(fontsize=7, ncol=2)
    ax[2].set_xlabel("simulation time s")
    fig.tight_layout()
    fig.savefig("reports/m5a_stopping_curves.png", dpi=150)
    plt.close(fig)
    atomic_json(
        output,
        dict(
            kind="offline_analysis_of_actual_M47_M48_CARLA_raw_logs",
            sources={f: file_hash(f) for f in sources},
            records=records,
            reachable_reference=envelope,
            comfort_limits=dict(deceleration=3, jerk_peak=6),
            physics_changed=False,
            new_probes=0,
            policy_analysis={
                "coast": "Observed coast deceleration can reduce speed but may roll backward; no stable-stop guarantee. Start coast only with measured remaining-distance envelope plus age margin, keep emergency override.",
                "normal_brake": "At low speed even tiny brake produces immediate impulse; ideal jerk bound is not a calibrated achievable command. No globally certified comfortable interval.",
                "hold": "Enter only at measured near-zero window (~0.02 m/s in historical cases), and verify consecutive speed/position stability; local window does not certify approach reachability.",
                "emergency": "Preserved full brake independent of comfort; report separately.",
            },
            status="Partial",
            conclusion="Finite sampled feasible points only. No full comfortable speed-to-prescribed-distance trajectory established; no proof that all possible controls are infeasible.",
        ),
    )


if __name__ == "__main__":
    main()
