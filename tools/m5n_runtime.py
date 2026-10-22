"""Actual GPU offline latency and formal RGB-only contract, no simulator load."""

import time, json
from pathlib import Path
import torch, numpy as np
from tools.m5n_common import rows, write, asset, stats
from tools.m5f_common import inputs
from models.m5n_candidate import DirectActorCandidate
from training.runtime import prepare_cuda
from simulator.recorder import file_hash


def main():
    prepare_cuda()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    model = DirectActorCandidate(enable=True)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t) * 1000
    rs = rows("val")
    chosen = [rs[i] for i in [0, 1, 2, 26, 27, 52, 78, 104, 105, 113]]
    xx = [inputs(r) for r in chosen]
    records = []
    first = None
    with torch.no_grad():
        for rep in range(4):
            for row, x in zip(chosen, xx):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = model(now_s=float(x["timestamps_s"][0]), **x)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t) * 1000
                record = dict(
                    sample_id=row["sample_id"], rep=rep, ms=elapsed, candidates=len(out)
                )
                records.append(record)
                if first is None:
                    first = record
                assert all(
                    o["status"] != "VerifiedContact" and not o["research_qualified"]
                    for o in out
                )
    violations = []
    x = xx[0]
    for name, bad in [
        ("actor", dict(x, actor_gt=torch.zeros(1, 3, device="cuda"))),
        ("depth", dict(x, depth_gt=torch.ones(1, 6, 384, 640, device="cuda"))),
        ("stale", x),
    ]:
        try:
            model(
                now_s=float(x["timestamps_s"][0]) + (0.5 if name == "stale" else 0),
                **bad,
            )
        except (ValueError, TypeError, KeyError, AssertionError) as e:
            violations.append(dict(test=name, rejected=True, error=str(e)))
        else:
            violations.append(dict(test=name, rejected=False))
    assert all(v["rejected"] for v in violations)
    write("runs/m5n_logs/runtime_raw.json", records)
    write(
        "reports/m5n_runtime_budget.json",
        dict(
            kind="actual_RTX4090_offline_sixRGB_inference_no_CARLA",
            load_ms=load_ms,
            cold=first,
            steady=stats([r["ms"] for r in records[1:]]),
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            contract_tests=violations,
            raw=asset("runs/m5n_logs/runtime_raw.json"),
            source_sha256=file_hash(__file__),
            inherited_runtime="M5M Partial",
            limits="Inputs preloadedGPU, excludes disk/H2D/sensor IPC. Full Mobile feature+foreground/candidates+DirectHead+geometry+empirical calibration included. No Dense branch needed. Finite40frames correlated10samples; no online tick/control test.",
        ),
    )


if __name__ == "__main__":
    main()
