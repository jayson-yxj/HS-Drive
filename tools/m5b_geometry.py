"""Train/val-only isolated projection diagnosis. Oracle is never formal input."""

import json, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from tools.m5a_learning import encoder_load, stack
from models.m5b_cached_encoder import cache_rgb, cached_forward
from models.m48_evidence_head import EvidenceHead
from datasets.m5a_training import VisualEvidenceDataset
from models.view_transform import feature_uv, unproject
from diagnostics.m5b_metrics import change_metrics, aggregate
from tools.m5b_learning import read_pairs
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def oracle(cache, item, row, net):
    f = json.loads(Path(row["raw_frame"]).read_text())
    root = Path(row["raw_frame"]).parent.parent
    b, n, c, h, w = cache["features"].shape
    uv = feature_uv(h, w, cache["image_hw"], "cuda")
    u = uv[..., 0].round().long().cpu().numpy()
    v = uv[..., 1].round().long().cpu().numpy()
    depth = []
    mask = []
    for name in row["camera_names"]:
        with np.load(root / f["cameras"][name]["depth"]) as z:
            d = z["depth_m"][v, u]
        with np.load(root / f["cameras"][name]["instance"]) as z:
            s = z["semantic"][v, u]
        depth.append(d)
        mask.append(np.isin(s, [12, 13, 14, 15, 16, 18, 19]))
    d = torch.tensor(np.array(depth)[None], device="cuda")
    valid = torch.isfinite(d) & (d >= 1.5) & (d < 80)
    centers = net.splat.depth_centers
    bins = (d[:, :, None] - centers[None, None, :, None, None]).abs().argmin(2)
    onehot = F.one_hot(bins, 48).permute(0, 1, 4, 2, 3).float() * valid[:, :, None]
    actor = torch.tensor(np.array(mask)[None], device="cuda") & valid
    gtpoints = unproject(uv.expand(1, n, h, w, 2), d, cache["K"], cache["T"])
    r, inside = net.splat.ranks(gtpoints)
    actor &= inside
    points = unproject(
        uv.expand(1, n, 48, h, w, 2),
        centers[None, None, :, None, None].expand(1, n, 48, h, w),
        cache["K"],
        cache["T"],
    )
    pr, pv = net.splat.ranks(points)
    same = (pr == r[:, :, None]) & pv
    prob = cache["probs"]
    xyerr = torch.linalg.vector_norm(
        points[..., :2] - gtpoints[:, :, None, :, :, :2], dim=-1
    )
    stats = dict(
        n_feature_actor_pixels=int(actor.sum()),
        gt_nearest_bin_probability=(
            float(prob.gather(2, bins[:, :, None]).squeeze(2)[actor].mean())
            if actor.any()
            else None
        ),
        within3m_probability=(
            float(
                (
                    prob
                    * ((centers[None, None, :, None, None] - d[:, :, None]).abs() < 3)
                )
                .sum(2)[actor]
                .mean()
            )
            if actor.any()
            else None
        ),
        correct_voxel_probability=(
            float((prob * same).sum(2)[actor].mean()) if actor.any() else None
        ),
        expected_xy_error_m=(
            float((prob * xyerr).sum(2)[actor].mean()) if actor.any() else None
        ),
        oracle_bin_correct_voxel_fraction=(
            float((onehot * same).sum(2)[actor].mean()) if actor.any() else None
        ),
    )
    return onehot, stats, actor


def main():
    out = Path("reports/m5b_geometry_diagnostics.json")
    if out.exists():
        raise FileExistsError(out)
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_index", "val"
    )
    lookup = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
    pairs = [
        (r, d)
        for r, d in read_pairs(ds.root, "val")
        if r["a_configuration"] == 0 and r["b_configuration"] in [1, 2, 3, 4, 7, 9]
    ]
    study = json.loads(Path("reports/m5b_context_ablation_corrected.json").read_text())
    records = []
    art = Path("reports/m5b_geometry_artifacts")
    art.mkdir(exist_ok=False)
    for model in [
        r for r in study["records"] if r["name"] in ["B_frozen", "C_context"]
    ]:
        net = encoder_load()
        ck = torch.load(model["checkpoint"], weights_only=True)
        head = EvidenceHead().cuda().eval()
        head.load_state_dict(ck["head"])
        if ck["context"] is not None:
            net.context_head.load_state_dict(ck["context"])
        cache = {}
        items = {}
        parts = {}
        oracles = {}
        scores = {}
        infos = {}
        with torch.no_grad():
            for i in sorted(set(lookup[r[k]] for r, _ in pairs for k in ["a", "b"])):
                item = ds[i]
                items[i] = item
                c = cache_rgb(net, stack([item], "inputs"))
                cache[i] = c
                p = cached_forward(net, c, return_parts=True)
                parts[i] = p
                o, info, actor = oracle(c, item, ds.rows[i], net)
                oracles[i] = o
                infos[i] = info
                scores[i] = {}
                for name, bev in [
                    ("learned", p["bev"]),
                    ("oracle_bin", cached_forward(net, c, probabilities=o)),
                ]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = head(bev)
                    from training.m5b_change_loss import obstacle_score

                    scores[i][name] = obstacle_score(pred)[0].cpu().numpy()
            for r, d in pairs:
                a, b = [lookup[r[k]] for k in ["a", "b"]]
                variants = {}
                for mode in ["learned", "oracle_bin"]:
                    variants[mode] = change_metrics(
                        scores[a][mode],
                        scores[b][mode],
                        d,
                        ck["selection"]["threshold"],
                    )
                # Hold empty context fixed; only obstacle-depth distribution changes.
                oa = cached_forward(
                    net, cache[a], probabilities=oracles[a], context=parts[a]["context"]
                )
                ob = cached_forward(
                    net, cache[b], probabilities=oracles[b], context=parts[a]["context"]
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pa = head(oa)
                    pb = head(ob)
                variants["oracle_fixed_empty_context"] = change_metrics(
                    obstacle_score(pa)[0].cpu().numpy(),
                    obstacle_score(pb)[0].cpu().numpy(),
                    d,
                    ck["selection"]["threshold"],
                )
                changes = {}
                for name in ["context", "height", "bev"]:
                    aa = parts[a][name].float()
                    bb = parts[b][name].float()
                    changes[name] = dict(
                        relative_l1=float(
                            (bb - aa).abs().mean() / aa.abs().mean().clamp_min(1e-8)
                        ),
                        mean_l1=float((bb - aa).abs().mean()),
                    )
                changes["image_features"] = dict(
                    relative_l1=float(
                        (cache[b]["features"] - cache[a]["features"]).abs().mean()
                        / cache[a]["features"].abs().mean().clamp_min(1e-8)
                    )
                )
                records.append(
                    dict(
                        name=model["name"],
                        seed=model["seed"],
                        group=r["group"],
                        configuration=r["b_configuration"],
                        depth=infos[b],
                        variants=variants,
                        stage_changes=changes,
                    )
                )
        if model["seed"] == 61 and model["name"] == "B_frozen":
            i = lookup[pairs[0][0]["b"]]
            np.savez_compressed(
                art / "m5b_projection_example.npz",
                learned_mass=parts[i]["mass"].cpu().numpy(),
                oracle_mass=cached_forward(
                    net, cache[i], probabilities=oracles[i], return_parts=True
                )["mass"]
                .detach()
                .cpu()
                .numpy(),
                learned_score=scores[i]["learned"],
                oracle_score=scores[i]["oracle_bin"],
            )
    gains = {}
    for name in ["B_frozen", "C_context"]:
        for seed in [61, 62]:
            rr = [x for x in records if x["name"] == name and x["seed"] == seed]
            means = {
                v: aggregate(
                    [dict(group=x["group"], metrics=x["variants"][v]) for x in rr]
                )["mean_scene_iou"]
                for v in ["learned", "oracle_bin", "oracle_fixed_empty_context"]
            }
            gains[f"{name}_{seed}"] = means
    D = all(
        gains[f"C_context_{s}"]["oracle_bin"] - gains[f"C_context_{s}"]["learned"]
        >= 0.10
        and next(
            r["selected"]["val_change_iou"]
            for r in study["records"]
            if r["name"] == "C_context" and r["seed"] == s
        )
        - next(
            r["selected"]["val_change_iou"]
            for r in study["records"]
            if r["name"] == "B_frozen" and r["seed"] == s
        )
        < 0.05
        for s in [61, 62]
    )
    atomic_json(
        out,
        dict(
            kind="actual_GPU_offline_validation_only_geometry_interventions",
            records=records,
            grouped_variant_scores=gains,
            D_condition_met=D,
            test_opened=False,
            source_sha256=file_hash(__file__),
            limits=[
                "Oracle nearest original LID bin, not exact continuous depth; quantization error reported.",
                "Frozen head receives out-of-distribution oracle projection. No causal proof from one perturbation.",
                "Actor semantic feature-center pixels may undersample small pedestrians.",
                "Fixed-context isolates depth change under constant background context; not formal performance.",
            ],
        ),
    )
    atomic_json(
        "reports/m5b_pretest_decision.json",
        dict(
            D_condition_met=D,
            D_started=False,
            decision="D needs a separate predeclared budget even if criterion met; no automatic large training.",
            checkpoint_sha256={r["checkpoint"]: r["sha256"] for r in study["records"]},
            thresholds={
                f"{r['name']}_{r['seed']}": r["selected"]["threshold"]
                for r in study["records"]
            },
            test_opened=False,
        ),
    )


if __name__ == "__main__":
    main()
