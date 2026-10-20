"""Finite RGB-only inference pressure process; never connects to CARLA/ticks."""

import argparse, time, json
from pathlib import Path
import numpy as np
import torch
from tools.m5a_learning import encoder_load, stack
from datasets.m5a_training import VisualEvidenceDataset
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=420)
    p.add_argument("--hz", type=float, default=10)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    if Path(a.report).exists():
        raise FileExistsError(a.report)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    item = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_index",
        "train",
        supervision=False,
    )[0]
    inp = stack([item], "inputs")
    times = []
    starts = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        while time.perf_counter() - start < a.duration:
            t = time.perf_counter()
            net(**inp)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            starts.append(t)
            times.append(dt)
            time.sleep(max(0, 1 / a.hz - dt))
    atomic_json(
        a.report,
        dict(
            kind="actual_RTX4090_RGB_encoder_inference_pressure_no_CARLA_API",
            duration_s=time.perf_counter() - start,
            target_hz=a.hz,
            actual_hz=len(times) / (time.perf_counter() - start),
            calls=len(times),
            elapsed_s=times,
            starts_wall=starts,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            source_sha256=file_hash(__file__),
            encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
        ),
    )


if __name__ == "__main__":
    main()
