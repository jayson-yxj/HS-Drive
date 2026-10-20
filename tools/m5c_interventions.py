"""Validation-only component interventions; oracle shifts are diagnostic, not upper bounds."""

import json, time
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from datasets.m5b_pairs import camera_rays
from models.m48_evidence_head import EvidenceHead
from models.m5b_cached_encoder import cache_rgb, cached_forward
from models.view_transform import feature_uv
from tools.m5a_learning import encoder_load, stack
from training.m5b_change_loss import obstacle_score
from diagnostics.m5a_metrics import binary_metrics
from tools.m5c_evaluate import sum_metrics
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def main():
    dest = Path("reports/m5c_component_interventions.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    index = "/root/gpufree-data/hs-drive-data/processed/m5c_index"
    ds = VisualEvidenceDataset(index, "val")
    study = json.loads(Path("reports/m5c_model_ablation.json").read_text())
    r = next(x for x in study["records"] if x["name"] == "C_pair" and x["seed"] == 71)
    net = encoder_load()
    ck = torch.load(r["checkpoint"], weights_only=True)
    net.context_head.load_state_dict(ck["context"])
    head = EvidenceHead().cuda().eval()
    head.load_state_dict(ck["head"])
    th = ck["selection"]["threshold"]
    refs = {
        v["sample_id"]: v
        for v in json.loads(Path(index, "m5c_surface_index.json").read_text())[
            "records"
        ]
    }
    records = []
    empties = {}
    with torch.no_grad():
        for i, row in enumerate(ds.rows):
            if row["configuration"] == 0:
                c = cache_rgb(net, stack([ds[i]], "inputs"))
                empties[row["group"]] = (c, cached_forward(net, c, return_parts=True))
        for i, row in enumerate(ds.rows):
            if row["configuration"] not in [1, 3, 4, 5, 6, 9, 11, 12, 13]:
                continue
            inp = stack([ds[i]], "inputs")
            c = cache_rgb(net, inp)
            parts = cached_forward(net, c, return_parts=True)
            ec, ep = empties[row["group"]]
            frame = json.loads(Path(row["raw_frame"]).read_text())
            raw = Path(row["raw_frame"]).parent.parent
            man = json.loads((raw / "manifest.json").read_text())
            _, n, _, h, w = c["features"].shape
            uv = feature_uv(h, w, c["image_hw"], "cuda").round().long().cpu().numpy()
            u, v = uv[..., 0], uv[..., 1]
            depth = []
            valid = []
            maskrgb = torch.zeros_like(inp["images"][:, :, :1], dtype=torch.bool)
            for j, name in enumerate(row["camera_names"]):
                rays, data = camera_rays(raw, man, frame, name)
                mask = np.zeros(data[0].shape, bool)
                mask[rays["uv"][:, 1], rays["uv"][:, 0]] = rays["reliable"]
                depth.append(data[0][v, u])
                valid.append(mask[v, u])
                actor = np.zeros(len(rays["endpoints"]), bool)
                for obj in frame["actors"]:
                    if obj["is_ego"]:
                        continue
                    box = np.array(obj["corners_hs_ego_m"])
                    actor |= (
                        (rays["endpoints"] >= box.min(0) - 0.05)
                        & (rays["endpoints"] <= box.max(0) + 0.05)
                    ).all(-1)
                pix = rays["uv"][actor]
                maskrgb[0, j, 0, pix[:, 1], pix[:, 0]] = True
            d = torch.tensor(np.array(depth)[None], device="cuda")
            valid = torch.tensor(np.array(valid)[None], device="cuda")
            bins = (
                (d[:, :, None] - net.splat.depth_centers[None, None, :, None, None])
                .abs()
                .argmin(2)
            )
            onehot = (
                torch.nn.functional.one_hot(bins, 48).permute(0, 1, 4, 2, 3).float()
            )
            oracle = torch.where(valid[:, :, None], onehot, c["probs"])
            occluded = {
                **inp,
                "images": torch.where(
                    maskrgb, torch.full_like(inp["images"], 0.5), inp["images"]
                ),
            }
            occ = cache_rgb(net, occluded)
            variants = {
                "original": parts["bev"],
                "oracle_nearest_bin_reliable": cached_forward(
                    net, c, probabilities=oracle
                ),
                "empty_depth_fixed_context_object": cached_forward(
                    net, c, probabilities=ec["probs"]
                ),
                "empty_context_fixed_depth_object": cached_forward(
                    net, c, context=ep["context"]
                ),
                "empty_bev": ep["bev"],
                "image_target_masked": cached_forward(net, occ),
            }
            # Repeat native learned height/mass through the same encoder is a wiring identity.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rebuilt = net.bev_encoder(
                    torch.cat(
                        (parts["height"].flatten(1, 2), torch.log1p(parts["mass"])), 1
                    )
                )
            identity = float((rebuilt - parts["bev"]).abs().max())
            with np.load(refs[row["sample_id"]]["path"]) as z:
                target = z["target"]
                known = z["valid"]
                actorcells = z["actor_masks"].any(0)
            for mode, bev in variants.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    score = obstacle_score(head(bev))[0].cpu().numpy()
                records.append(
                    dict(
                        group=row["group"],
                        configuration=row["configuration"],
                        mode=mode,
                        absolute=binary_metrics(score, target, known, th),
                        target_actor=binary_metrics(
                            score, target, known & actorcells, th
                        ),
                        bev_encoder_identity_max_abs=identity,
                    )
                )
    summary = {
        mode: dict(
            absolute=sum_metrics([r["absolute"] for r in records if r["mode"] == mode]),
            target_actor=sum_metrics(
                [r["target_actor"] for r in records if r["mode"] == mode]
            ),
        )
        for mode in variants
    }
    atomic_json(
        dest,
        dict(
            kind="actual_GPU_validation_only_interventions_not_formal_performance",
            records=records,
            summary=summary,
            source_sha256=file_hash(__file__),
            checkpoint_sha256=file_hash(
                next(
                    x["checkpoint"]
                    for x in study["records"]
                    if x["name"] == "C_pair" and x["seed"] == 71
                )
            ),
            notes="Fixed RGB/context/probability isolates pathways but oracle causes distribution shift. No module causality ranking from feature L1. BEV identity verifies wiring only; architectural BEV suppression remains unproven. Target-local image mask uses GT solely as an offline diagnostic, never deployment. Native oracle replacement uses nearest pixel depth assigned to original halfpixel ray; quantization conclusions use separate strictly matched UV audit.",
        ),
    )


if __name__ == "__main__":
    main()
