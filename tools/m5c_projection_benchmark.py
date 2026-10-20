"""No-training splat-only resource benchmark; stride8 features synthetic interpolation."""

import time, json
from pathlib import Path
import torch, numpy as np
from torch.nn import functional as F
from tools.m5a_learning import encoder_load, stack
from models.m5b_cached_encoder import cache_rgb, cached_forward
from datasets.m5a_training import VisualEvidenceDataset
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_projection_benchmark.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    item = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_index", "val", supervision=False
    )[0]
    inputs = stack([item], "inputs")
    rows = []
    with torch.no_grad():
        cache = cache_rgb(net, inputs)
        ctx = cached_forward(net, cache, return_parts=True)["context"]
        for stride in [16, 8]:
            c = ctx
            p = cache["probs"]
            if stride == 8:
                c = F.interpolate(
                    ctx.flatten(0, 1).float(),
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(1, 6, ctx.shape[2], 48, 80)
                p = F.interpolate(
                    p.flatten(0, 1),
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(1, 6, 48, 48, 80)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            lat = []
            for repeat in range(35):
                torch.cuda.synchronize()
                t = time.perf_counter()
                net.splat(c, p, cache["K"], cache["T"], cache["image_hw"])
                torch.cuda.synchronize()
                if repeat >= 5:
                    lat.append((time.perf_counter() - t) * 1000)
            rows.append(
                dict(
                    stride=stride,
                    point_count=p.numel(),
                    p50_ms=float(np.median(lat)),
                    p95_ms=float(np.quantile(lat, 0.95)),
                    incremental_peak_mib=(torch.cuda.max_memory_allocated() - base)
                    / 2**20,
                    peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                )
            )
        full = []
        for i in range(35):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                net(**inputs)
            torch.cuda.synchronize()
            if i >= 5:
                full.append((time.perf_counter() - t) * 1000)
    atomic_json(
        dest,
        dict(
            kind="actual_RTX4090_microbenchmark_under_CARLA_capture_load",
            records=rows,
            original_rgb_encoder_p50_ms=float(np.median(full)),
            original_rgb_encoder_p95_ms=float(np.quantile(full, 0.95)),
            source_sha256=file_hash(__file__),
            limits="Stride8 is interpolated context/probabilities solely for workload estimation; no higher-resolution visual features or quality claims. Existing splat chunking; no architecture modification. Benchmark concurrent capture, not isolated latency gate.",
        ),
    )


if __name__ == "__main__":
    main()
