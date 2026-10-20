"""Formal RGB-only path vs frozen cache; report precision-induced score/decision drift."""

import json, time
from pathlib import Path
import numpy as np, torch
from tools.m5a_learning import encoder_load, stack
from datasets.m5a_training import VisualEvidenceDataset
from models.m5b_cached_encoder import cache_rgb, cached_forward
from models.m48_evidence_head import EvidenceHead
from models.m5c_obstacle_head import VisibleObstacleHead
from training.m5b_change_loss import obstacle_score
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_bf16_equivalence.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5c_index", "val", supervision=False
    )
    study = json.loads(Path("reports/m5c_model_ablation.json").read_text())
    records = []
    for r in study["records"]:
        if r["name"] not in ["C_pair", "D_direct"]:
            continue
        net = encoder_load()
        ck = torch.load(r["checkpoint"], weights_only=True)
        net.context_head.load_state_dict(ck["context"])
        head = (
            (VisibleObstacleHead() if r["name"] == "D_direct" else EvidenceHead())
            .cuda()
            .eval()
        )
        head.load_state_dict(ck["head"])
        for i in [1, 3, 5, 17, 19, 21, 33, 35, 37, 49, 51, 53]:
            inp = stack([ds[i]], "inputs")
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize()
                t = time.perf_counter()
                p = head(net(**inp)["bev_current"])
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1000
                c = cache_rgb(net, inp)
                q = head(cached_forward(net, c))
                pa = (
                    p["visible_obstacle"].float().sigmoid()
                    if "visible_obstacle" in p
                    else obstacle_score(p)
                )
                pb = (
                    q["visible_obstacle"].float().sigmoid()
                    if "visible_obstacle" in q
                    else obstacle_score(q)
                )
                err = (pa - pb).abs()
                th = ck["selection"]["threshold"]
            records.append(
                dict(
                    model=r["name"],
                    seed=r["seed"],
                    sample=ds.rows[i]["sample_id"],
                    score_mae=float(err.mean()),
                    score_max=float(err.max()),
                    threshold_disagreement=int(((pa > th) != (pb > th)).sum()),
                    full_rgb_inference_ms=ms,
                    peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                )
            )
    atomic_json(
        dest,
        dict(
            kind="actual_RTX4090_validation_formal_RGB_inference_noGT",
            records=records,
            source_sha256=file_hash(__file__),
            notes="Direct vs cache BF16 rounding/scatter nondeterminism measured, FP32 equivalence in tests. Warmup included in listed timings. No GT inputs.",
        ),
    )


if __name__ == "__main__":
    main()
