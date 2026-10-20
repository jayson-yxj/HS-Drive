"""Additional validation-only frozen-RGB ablations, never formal performance."""

import json
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from models.m5b_cached_encoder import cache_rgb, cached_forward
from tools.m5a_learning import encoder_load, stack
from tools.m5b_geometry import oracle
from models.m48_evidence_head import EvidenceHead
from training.m5b_change_loss import obstacle_score
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_index", "val"
    )
    study = json.loads(Path("reports/m5b_context_ablation_corrected.json").read_text())
    records = []
    with torch.no_grad():
        for r in [
            x
            for x in study["records"]
            if x["name"] in ["B_frozen", "C_context"] and x["seed"] == 61
        ]:
            net = encoder_load()
            ck = torch.load(r["checkpoint"], weights_only=True)
            head = EvidenceHead().cuda()
            head.load_state_dict(ck["head"])
            if ck["context"] is not None:
                net.context_head.load_state_dict(ck["context"])
            for group in sorted(set(x["group"] for x in ds.rows)):
                ia = next(
                    i
                    for i, x in enumerate(ds.rows)
                    if x["group"] == group and x["configuration"] == 0
                )
                ib = next(
                    i
                    for i, x in enumerate(ds.rows)
                    if x["group"] == group and x["configuration"] == 2
                )
                ca = cache_rgb(net, stack([ds[ia]], "inputs"))
                cb = cache_rgb(net, stack([ds[ib]], "inputs"))
                a = cached_forward(net, ca, return_parts=True)
                b = cached_forward(net, cb, return_parts=True)
                variants = {}
                for name, p in [
                    ("normal", b),
                    (
                        "fixed_empty_context",
                        cached_forward(
                            net, cb, context=a["context"], return_parts=True
                        ),
                    ),
                    (
                        "fixed_empty_depth",
                        cached_forward(
                            net, cb, probabilities=ca["probs"], return_parts=True
                        ),
                    ),
                ]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = head(p["bev"])
                        base = head(a["bev"])
                    variants[name] = dict(
                        score_delta_l1=float(
                            (obstacle_score(o) - obstacle_score(base)).abs().mean()
                        ),
                        height_delta_relative=float(
                            (p["height"] - a["height"]).abs().mean()
                            / a["height"].abs().mean().clamp_min(1e-8)
                        ),
                        bev_delta_relative=float(
                            (p["bev"] - a["bev"]).abs().mean()
                            / a["bev"].abs().mean().clamp_min(1e-8)
                        ),
                    )
                records.append(dict(name=r["name"], group=group, variants=variants))
    atomic_json(
        "reports/m5b_geometry_factorization.json",
        dict(
            kind="offline_actual_GPU_val_only",
            records=records,
            source_sha256=file_hash(__file__),
            limits="Feature L1 scales across stages not commensurate causal importance; frozen depth/context interventions can be OOD. Cannot infer unique bottleneck from amplitude alone.",
        ),
    )


if __name__ == "__main__":
    main()
