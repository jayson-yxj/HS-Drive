"""Read-only process resource telemetry for M5-E."""
import subprocess,time
from pathlib import Path

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
            if not any(s in cmd for s in ["CarlaUE4-Linux", "m5e_", "m5d_", "spawn_main"]):
                continue
            fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
            status = (p / "status").read_text()
            records.append(
                dict(
                    pid=int(p.name),
                    cmd=cmd,
                    cpu_ticks=int(fields[11]) + int(fields[12]),
                    processor=int(fields[36]),
                    allowed_cpus=next((l.split(':',1)[1].strip() for l in status.splitlines() if l.startswith('Cpus_allowed_list:')),None),
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

