"""One sealed-test evaluation after checkpoint/threshold/geometry decision freeze."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from models.m48_evidence_head import EvidenceHead, PositionPrior
from models.m5b_cached_encoder import cache_rgb, cached_forward
from tools.m5a_learning import encoder_load, stack, task_arrays
from tools.m5b_learning import read_pairs
from tools.m5a_evaluate import summarize
from training.m5b_change_loss import obstacle_score
from diagnostics.m5b_metrics import change_metrics, aggregate
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index", default="/root/gpufree-data/hs-drive-data/processed/m5b_index"
    )
    a = p.parse_args()
    decision = json.loads(Path("reports/m5b_pretest_decision.json").read_text())
    study = json.loads(Path("reports/m5b_context_ablation_corrected.json").read_text())
    out = Path("reports/m5b_counterfactual_detection.json")
    if out.exists():
        raise FileExistsError(out)
    for r in study["records"]:
        assert decision["checkpoint_sha256"][r["checkpoint"]] == file_hash(
            r["checkpoint"]
        )
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(a.index, "test")
    items = [ds[i] for i in range(len(ds))]
    lookup = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
    pairs = read_pairs(a.index, "test")
    assert len(pairs) == 66
    net = encoder_load()
    caches = [cache_rgb(net, stack([it], "inputs")) for it in items]
    labels = stack(items, "targets", "cpu")
    art = Path("reports/m5b_predictions")
    art.mkdir(exist_ok=False)
    records = []
    for r in study["records"]:
        net = encoder_load()
        ck = torch.load(r["checkpoint"], weights_only=True)
        head = (
            (PositionPrior() if r["name"] == "A_prior" else EvidenceHead())
            .cuda()
            .eval()
        )
        head.load_state_dict(ck["head"])
        if ck["context"] is not None:
            net.context_head.load_state_dict(ck["context"])
        pred = {k: [] for k in ["surface", "free", "semantic"]}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for c in caches:
                for k, v in head(cached_forward(net, c)).items():
                    pred[k].append(v.float().cpu())
        pred = {k: torch.cat(v) for k, v in pred.items()}
        score = obstacle_score(pred).numpy()
        th = ck["selection"]["threshold"]
        metrics = []
        for row, d in pairs:
            ia, ib = [lookup[row[k]] for k in ["a", "b"]]
            m = change_metrics(score[ia], score[ib], d, th)
            sub = {}
            # Strata explicit, including Unknown in original V2; point-witness may support an event there without volume claim.
            for label, mask in [
                (
                    "ground",
                    labels["surface_semantic"][ia, 0].numpy()
                    | labels["surface_semantic"][ib, 0].numpy(),
                ),
                (
                    "obstacle",
                    labels["surface_semantic"][ia, 1].numpy()
                    | labels["surface_semantic"][ib, 1].numpy(),
                ),
                (
                    "unknown",
                    ~labels["observability"][ia].numpy()
                    | ~labels["observability"][ib].numpy(),
                ),
                (
                    "mixed",
                    labels["mixed_cell_mask"][ia].numpy()
                    | labels["mixed_cell_mask"][ib].numpy(),
                ),
            ]:
                sub[label] = change_metrics(
                    score[ia],
                    score[ib],
                    {
                        k: v & mask
                        for k, v in d.items()
                        if k in ["appeared", "vanished", "stable"]
                    },
                    th,
                )
            # Actor AABB intersects XY grid: geometry support region only, NOT exact mesh/physical occupancy.
            fb = json.loads(Path(ds.rows[ib]["raw_frame"]).read_text())
            foot = np.zeros((160, 120), bool)
            x = -20 + (np.arange(160) + 0.5) * 0.5
            y = -30 + (np.arange(120) + 0.5) * 0.5
            for actor in fb["actors"]:
                if actor["is_ego"]:
                    continue
                corners = np.asarray(actor["corners_hs_ego_m"])
                low = corners.min(0)
                high = corners.max(0)
                foot |= (
                    (x[:, None] + 0.25 > low[0])
                    & (x[:, None] - 0.25 < high[0])
                    & (y[None, :] + 0.25 > low[1])
                    & (y[None, :] - 0.25 < high[1])
                )
            target = d["appeared"] & foot[None]
            hit = (score[ib] - score[ia] > th) & target
            m["visible_actor_footprint_witness"] = dict(
                n=int(target.sum()),
                miss=int((target & ~hit).sum()),
                recall=float(hit.sum() / target.sum()) if target.any() else None,
                reference="render surface witness intersecting actor AABB; not exact mesh footprint",
            )
            metrics.append(
                dict(
                    group=row["group"],
                    a=row["a_configuration"],
                    b=row["b_configuration"],
                    metrics=m,
                    strata=sub,
                )
            )
        conditional = {
            k: tuple(x.numpy() for x in v) for k, v in task_arrays(pred, labels).items()
        }
        event = summarize(conditional, ds.rows, {k: 0.5 for k in conditional})
        returns = []
        for group in sorted(set(row["group"] for row in ds.rows)):
            i = next(
                i
                for i, x in enumerate(ds.rows)
                if x["group"] == group and x["configuration"] == 0
            )
            j = next(
                i
                for i, x in enumerate(ds.rows)
                if x["group"] == group and x["configuration"] == 11
            )
            mask = (
                labels["observability"][i].numpy() & labels["observability"][j].numpy()
            )
            returns.append(
                dict(
                    group=group,
                    n=int(mask.sum()),
                    empty_return_mean_abs_change=float(
                        abs(score[j] - score[i])[mask].mean()
                    ),
                    empty_return_large_change=float(
                        (abs(score[j] - score[i])[mask] > th).mean()
                    ),
                )
            )
        path = art / f"m5b_{r['name']}_{r['seed']}.npz"
        np.savez_compressed(
            path, score=score, **{k: v.numpy() for k, v in pred.items()}
        )
        record = dict(
            name=r["name"],
            seed=r["seed"],
            threshold=th,
            summary=aggregate(metrics),
            pairs=metrics,
            conditional_events=event,
            empty_returns=returns,
            predictions=str(path),
            sha256=file_hash(path),
        )
        records.append(record)
        print(r["name"], r["seed"], record["summary"], flush=True)
    atomic_json(
        out,
        dict(
            kind="actual_GPU_one_shot_new_group_test",
            records=records,
            pretest_decision_sha256=file_hash("reports/m5b_pretest_decision.json"),
            groups=sorted(set(r["group"] for r in ds.rows)),
            source_sha256=file_hash(__file__),
            scope="Signed change localization on supported visible-point witness and stable observed surface domain. No whole-volume detection claim. 3 independent groups; configuration cells not independent.",
            score="surface sigmoid * conditional obstacle sigmoid, not calibrated occupancy probability",
        ),
    )
    comparison = []
    for r in records:
        a = next(
            x for x in records if x["name"] == "A_prior" and x["seed"] == r["seed"]
        )
        b = next(
            x for x in records if x["name"] == "B_frozen" and x["seed"] == r["seed"]
        )
        c = next(
            x for x in records if x["name"] == "C_context" and x["seed"] == r["seed"]
        )
        comparison.append(
            dict(
                name=r["name"],
                seed=r["seed"],
                delta_vs_prior={
                    g: v["iou"] - a["summary"]["by_scene"][g]["iou"]
                    for g, v in r["summary"]["by_scene"].items()
                },
                delta_vs_frozen=r["summary"]["mean_scene_iou"]
                - b["summary"]["mean_scene_iou"],
                delta_vs_context=r["summary"]["mean_scene_iou"]
                - c["summary"]["mean_scene_iou"],
            )
        )
    atomic_json(
        "reports/m5b_visual_prior_comparison.json",
        dict(
            kind="scene_paired_test_comparison",
            records=comparison,
            source_sha256=file_hash(__file__),
            independent_test_groups=3,
            statistical_significance="Insufficient scene count for broad population claim. Seeds are training repeats, not extra independent scenes.",
        ),
    )


if __name__ == "__main__":
    main()
