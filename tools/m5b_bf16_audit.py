"""Quantify cached vs direct BF16 inference; FP32 algebra verified by test."""

import json
from pathlib import Path
import numpy as np, torch
from tools.m5a_learning import encoder_load, stack
from models.m5b_cached_encoder import cache_rgb, cached_forward
from models.m48_evidence_head import EvidenceHead
from datasets.m5a_training import VisualEvidenceDataset
from training.m5b_change_loss import obstacle_score
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_index", "val", supervision=False
    )
    net = encoder_load()
    ck = torch.load("runs/m5b_learning/m5b_C_pair_61.pt", weights_only=True)
    net.context_head.load_state_dict(ck["context"])
    head = EvidenceHead().cuda()
    head.load_state_dict(ck["head"])
    records = []
    with torch.no_grad():
        for i in range(0, len(ds), 4):
            inp = stack([ds[i]], "inputs")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = net(**inp)["bev_current"]
                pa = obstacle_score(head(a))
            cache = cache_rgb(net, inp)
            b = cached_forward(net, cache)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pb = obstacle_score(head(b))
            e = (a - b).abs()
            s = (pa - pb).abs()
            records.append(
                dict(
                    sample=ds.rows[i]["sample_id"],
                    bev_mae=float(e.mean()),
                    bev_max=float(e.max()),
                    score_mae=float(s.mean()),
                    score_max=float(s.max()),
                )
            )
    atomic_json(
        "reports/m5b_bf16_equivalence.json",
        dict(
            kind="actual_GPU_validation_only_cached_vs_formal",
            records=records,
            source_sha256=file_hash(__file__),
            limits="Non-bitwise GPU scatter reduction and BF16 rounding at sparse points; FP32 equivalence separately tested. Does not use GT input.",
        ),
    )


if __name__ == "__main__":
    main()
