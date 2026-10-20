"""Separate server blocking, scheduling and moving/parked wall-clock intervals."""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def stats(v):
    a = np.asarray(v, float)
    return (
        dict(
            n=len(a),
            mean=float(a.mean()),
            p50=float(np.quantile(a, 0.5)),
            p95=float(np.quantile(a, 0.95)),
            p99=float(np.quantile(a, 0.99)),
            max=float(a.max()),
        )
        if len(a)
        else dict(n=0)
    )


def main():
    dest = Path("reports/m5d_runtime_attribution.json")
    if dest.exists():
        raise FileExistsError(dest)
    trial = json.loads(Path("reports/m5d_runtime_stress.json").read_text())
    out = []
    for case in trial["records"]:
        for result in case.get("results", []):
            rows = [json.loads(l) for l in Path(result["log"]).read_text().splitlines()]
            t = np.array([r["control_apply_wall"] for r in rows])
            interval = np.diff(t) * 1000
            speed = np.array([r["measurement"]["speed"] for r in rows])
            part = {}
            for name, m in [
                ("moving", speed[1:] > 0.2),
                ("stationary", speed[1:] <= 0.2),
            ]:
                part[name] = dict(
                    interval_ms=stats(interval[m]),
                    deadline_miss_rate=(
                        float(
                            np.mean(
                                [
                                    r["deadline_missed"]
                                    for r, yes in zip(rows[1:], m)
                                    if yes
                                ]
                            )
                        )
                        if m.any()
                        else None
                    ),
                )
            tail = []
            for i in np.argsort(interval)[-10:][::-1]:
                before, after = rows[i], rows[i + 1]
                tail.append(
                    dict(
                        frame=after["frame"],
                        interval_ms=float(interval[i]),
                        preceding_tick_ms=before["profile"]["world_tick_ms"],
                        preceding_compute_ms=before["compute_ms"],
                        current_compute_ms=after["compute_ms"],
                        between_cycles_ms=(
                            after["wall_begin"] - before["tick_end_wall"]
                        )
                        * 1000,
                        scheduled_wakeup_lateness_ms=max(
                            0,
                            after["wall_begin"]
                            - max(before["tick_end_wall"], before["deadline_wall"]),
                        )
                        * 1000,
                        speed_mps=after["measurement"]["speed"],
                        deadline_missed=after["deadline_missed"],
                    )
                )
            tele = json.loads(Path(case["telemetry"]).read_text())
            proc = {}
            prev = {}
            gpus = []
            for sample in tele["records"]:
                try:
                    gpus.append([float(v) for v in sample["gpu"].split(",")])
                except ValueError:
                    pass
                for p in sample["processes"]:
                    key = str(p["pid"])
                    last = prev.get(key)
                    prev[key] = (sample["wall"], p)
                    if last is None:
                        continue
                    dt = sample["wall"] - last[0]
                    if dt <= 0:
                        continue
                    r = proc.setdefault(
                        key,
                        dict(
                            command=p["cmd"],
                            cpu_one_core_percent=[],
                            schedule_wait_ms=[],
                            cpu_core_ids=[],
                        ),
                    )
                    r["cpu_one_core_percent"].append(
                        100
                        * (p["cpu_ticks"] - last[1]["cpu_ticks"])
                        / tele["clock_ticks_per_s"]
                        / dt
                    )
                    r["cpu_core_ids"].append(p["processor"])
                    ss = [int(x) for x in p["schedstat"].split()]
                    old = [int(x) for x in last[1]["schedstat"].split()]
                    if len(ss) > 1 and len(old) > 1:
                        r["schedule_wait_ms"].append((ss[1] - old[1]) / 1e6)
            resources = {
                k: dict(
                    command=v["command"],
                    cpu_one_core_percent=stats(v["cpu_one_core_percent"]),
                    main_thread_schedule_wait_ms_per_sample=stats(
                        v["schedule_wait_ms"]
                    ),
                    cpu_cores=sorted(set(v["cpu_core_ids"])),
                )
                for k, v in proc.items()
            }
            gm = np.array(gpus)
            out.append(
                dict(
                    case=case["name"],
                    phase=part,
                    server_snapshot_interval_ms=stats(
                        np.diff(
                            [r["server_snapshot_platform_timestamp"] for r in rows[1:]]
                        )
                        * 1000
                    ),
                    tail=tail,
                    processes=resources,
                    gpu_utilization_percent=stats(gm[:, 0]) if len(gm) else None,
                    gpu_memory_mib=stats(gm[:, 2]) if len(gm) else None,
                    period_gate=result["control_interval_ms"]["p99"] <= 55
                    and result["control_interval_ms"]["max"] <= 100
                    and result["deadline_miss_rate"] <= 0.001
                    and result["normal_control_expired"] == 0,
                )
            )
    atomic_json(
        dest,
        dict(
            kind="offline_attribution_actual_CARLA_telemetry",
            records=out,
            source_sha256=file_hash(__file__),
            limits="Server snapshot platform interval is wall frame-advance proxy, not Unreal CPU/GPU render duration. proc stat process CPU includes threads, schedstat is main thread only. One-factor sequential runs lack randomization. Tail previous tick blocks next command; do not attribute solely from current row tick. GPU utilization is device-wide.",
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for case in trial["records"]:
        for r in case.get("results", []):
            rows = [json.loads(l) for l in Path(r["log"]).read_text().splitlines()]
            t = np.array([r["control_apply_wall"] for r in rows])
            ax[0].plot(t[1:] - t[0], np.diff(t) * 1000, label=case["name"], alpha=0.7)
            ax[1].plot(
                t - t[0], [v["profile"]["world_tick_ms"] for v in rows], alpha=0.7
            )
    ax[0].axhline(50, color="k", ls="--")
    ax[0].legend(fontsize=8)
    ax[0].set(ylabel="Control command interval ms")
    ax[1].set(xlabel="Trial wall seconds", ylabel="world.tick blocking ms")
    fig.tight_layout()
    Path("reports/m5d_figures").mkdir(exist_ok=True)
    fig.savefig("reports/m5d_figures/m5d_runtime_tails.png", dpi=150)


if __name__ == "__main__":
    main()
