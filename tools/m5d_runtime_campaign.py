"""Sequential factorial load trials with resource telemetry; single CARLA tick owner."""

import json, subprocess, time, threading, os
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def snapshot():
    records = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (
                (p / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode(errors="replace")
            )
            if not any(s in cmd for s in ["CarlaUE4-Linux", "m5d_", "spawn_main"]):
                continue
            fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
            status = (p / "status").read_text()
            records.append(
                dict(
                    pid=int(p.name),
                    cmd=cmd,
                    cpu_ticks=int(fields[11]) + int(fields[12]),
                    processor=int(fields[36]),
                    voluntary_context_switches=next(
                        (
                            int(l.split()[-1])
                            for l in status.splitlines()
                            if l.startswith("voluntary_ctxt_switches")
                        ),
                        None,
                    ),
                    nonvoluntary_context_switches=next(
                        (
                            int(l.split()[-1])
                            for l in status.splitlines()
                            if l.startswith("nonvoluntary_ctxt_switches")
                        ),
                        None,
                    ),
                    schedstat=(p / "schedstat").read_text().strip(),
                )
            )
        except (OSError, ValueError, IndexError):
            continue
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return dict(wall=time.perf_counter(), processes=records, gpu=gpu)


def main():
    out = Path("reports/m5d_runtime_stress.json")
    if out.exists():
        raise FileExistsError(out)
    records = []
    cases = [
        ("alone", False, False, 1800, 250),
        ("cpu", True, False, 1800, 250),
        ("gpu", False, True, 1800, 250),
        ("mixed", True, True, 1800, 250),
        ("parked_mixed", True, True, 6000, 0),
        ("alone_repeat", False, False, 1800, 250),
    ]
    for name, cpu, gpu, ticks, distance in cases:
        duration = ticks * 0.05 + 25
        loads = []
        for flag, module in [(cpu, "cpu"), (gpu, "gpu")]:
            if flag:
                log = open(f"reports/m5d_logs/m5d_{name}_{module}load.log", "w")
                proc = subprocess.Popen(
                    [
                        ".venv-gpu/bin/python",
                        "-m",
                        f"tools.m5d_{module}_load",
                        "--duration",
                        str(duration),
                        "--report",
                        f"reports/m5d_{name}_{module}load.json",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                loads.append((proc, log))
        telemetry = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                telemetry.append(snapshot())
                stop.wait(1)

        th = threading.Thread(target=monitor, daemon=True)
        th.start()
        path = f"reports/m5d_runtime_{name}.json"
        log = open(f"reports/m5d_logs/m5d_runtime_{name}.log", "w")
        command = [
            ".venv/bin/python",
            "-m",
            "tools.m5d_async_drive",
            "--output",
            path,
            "--artifacts",
            f"reports/m5d_runtime_{name}",
            "--scenarios",
            "straight",
            "--ticks",
            str(ticks),
            "--hold-after-route",
            "--route-distance",
            str(distance),
            "--load-description",
            name,
        ]
        ret = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        log.close()
        stop.set()
        th.join()
        for proc, log in loads:
            proc.wait()
            log.close()
        tele = f"reports/m5d_runtime_{name}_resources.json"
        atomic_json(
            tele,
            dict(
                kind="actual_proc_schedstat_and_nvidia_smi_1Hz",
                clock_ticks_per_s=os.sysconf("SC_CLK_TCK"),
                records=telemetry,
            ),
        )
        record = dict(
            name=name,
            cpu_load=cpu,
            gpu_load=gpu,
            ticks=ticks,
            route_distance_m=distance,
            exit_code=ret,
            command=command,
            report=path,
            telemetry=tele,
        )
        if Path(path).exists():
            record["results"] = json.loads(Path(path).read_text())["results"]
        records.append(record)
        atomic_json("reports/m5d_runtime_progress.json", dict(records=records))
        print(name, ret, flush=True)
    atomic_json(
        out,
        dict(
            kind="actual_CARLA_factorial_shared_resource_trials",
            records=records,
            source_sha256=file_hash(__file__),
            control_hz=20,
            planner_hz=5,
            safety_seconds=6,
            limits="One sequential run per load plus isolated repeat; no randomization, shared-host background load can confound causal attribution. GPU load actualRGBencoder10Hz. Longer route follows legal waypoint sequence with closest heading, may degrade at bends; report actual motion/travel separately. Parked soak not continuous motion. schedstat kernel CPU running/waitns; server tick wall blocking is not direct Unreal GPU frame profiling.",
            gates=dict(
                interval_p99_ms=55,
                interval_max_ms=100,
                deadline_miss_rate=0.001,
                expired_normal_control=0,
            ),
        ),
    )


if __name__ == "__main__":
    main()
