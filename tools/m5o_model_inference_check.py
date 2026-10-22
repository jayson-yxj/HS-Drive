"""Numerical cache/full RGB feature equivalence on new Train only; never control."""

import json
from pathlib import Path
import numpy as np, torch
from tools.m5o_common import rows, write, asset, provenance
from tools.m5f_common import inputs, input_path
from tools.m5i_data import load
from models.m5n_direct import feature_vector, DirectCenterHead, reconstruct
from diagnostics.m5l_objects import components
from training.runtime import prepare_cuda


@torch.no_grad()
def main():
    prepare_cuda()
    torch.set_num_threads(2)
    root = Path("runs/m5o_cache_train")
    cc = json.loads((root / "candidates.json").read_text())
    X = torch.load(root / "features.pt", weights_only=True)["rgb_features"]
    cfg = json.loads(Path("configs/m5j_models.json").read_text())
    training = json.loads(Path("reports/m5o_training.json").read_text())
    checks = []
    rr = rows("train")
    selected = [rr[0], rr[len(rr) // 2], rr[-1]]
    for seed in [101, 102]:
        mobile = load(
            next(
                r
                for r in cfg["pixel"]
                if r["name"] == "mobile_gt" and r["seed"] == seed
            )
        )[0]
        rec = next(
            r
            for r in training["records"]
            if r["seed"] == seed and r["fraction"] == 1 and r["arm"] == "direct"
        )["selected"]
        ck = torch.load(rec["path"], weights_only=True)
        head = DirectCenterHead(600).cuda().eval()
        head.load_state_dict(ck["model"])
        for row in selected:
            x = inputs(row)
            inp = json.loads(input_path(row).read_text())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = mobile.fuse(mobile.levels(x["images"]))
                logits = mobile.head(features, (384, 640)).float().sigmoid()
            q = (
                (torch.clamp((logits * 256).floor(), max=255) / 256)
                .reshape(6, 384, 640)
                .cpu()
                .numpy()
            )
            for ci, cam in enumerate(row["camera_names"]):
                cs = components(q[ci])
                cache = [
                    c
                    for c in cc
                    if c["seed"] == seed
                    and c["sample_id"] == row["sample_id"]
                    and c["camera"] == cam
                ]
                assert len(cs) == len(cache)
                for c, saved in zip(cs, cache):
                    K = np.array(inp["intrinsics"][ci])
                    T = np.array(inp["cam_to_ego"][ci])
                    v = feature_vector(features[ci : ci + 1], c, q[ci], K, T)
                    old = X[saved["feature_index"]].cuda()
                    delta = float((v - old).abs().max())
                    region = torch.tensor([saved["region"]], device="cuda")
                    k = torch.tensor(K[None], device="cuda")
                    t = torch.tensor(T[None], device="cuda")
                    xyz = reconstruct(head(v[None]), region, k, t)
                    xyz0 = reconstruct(head(old[None]), region, k, t)
                    error = float((xyz - xyz0).abs().max())
                    assert delta < 0.002 and error < 0.002
                    checks.append(
                        dict(
                            candidate_id=saved["candidate_id"],
                            feature_max_abs=delta,
                            position_max_abs_m=error,
                        )
                    )
    write(
        "reports/m5o_inference_equivalence.json",
        dict(
            kind="actual_RTX4090_new_Train_RGB_replay",
            passed=True,
            records=checks,
            provenance=provenance(),
            limits="Train replay numerical path check, not generalization or latency. No GT features.",
        ),
    )


if __name__ == "__main__":
    main()
