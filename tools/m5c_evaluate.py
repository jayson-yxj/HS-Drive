"""Single-frame held-out surface localization; val selects thresholds, test reads freeze."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from datasets.m5a_training import VisualEvidenceDataset
from models.m48_evidence_head import EvidenceHead, PositionPrior
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5b_cached_encoder import cache_rgb, cached_forward
from tools.m5a_learning import encoder_load, stack
from tools.m5c_learning import read_pairs
from tools.m5b_learning import choose
from training.m5b_change_loss import obstacle_score
from diagnostics.m5b_metrics import change_metrics, aggregate
from diagnostics.m5c_localization import absolute_metrics, scene_score, matching
from diagnostics.m5a_metrics import binary_metrics
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def sum_metrics(rows):
    counts = {
        k: sum(r[k] for r in rows) for k in ["tp", "fp", "fn", "tn", "positive", "n"]
    }
    tp, fp, fn = [counts[k] for k in ["tp", "fp", "fn"]]
    return dict(
        **counts,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
    )


def summarize(rows):
    byscene = {
        g: sum_metrics([r["metrics"]["exact"] for r in rows if r["group"] == g])
        for g in sorted(set(r["group"] for r in rows))
    }
    actors = [a for r in rows for a in r["metrics"]["actors"] if a["eligible"]]
    return dict(
        overall=sum_metrics([r["metrics"]["exact"] for r in rows]),
        by_scene=byscene,
        mean_scene_iou=float(
            np.mean([v["iou"] for v in byscene.values() if v["iou"] is not None])
        ),
        by_distance={
            d: sum_metrics([r["metrics"]["by_distance"][d]["exact"] for r in rows])
            for d in rows[0]["metrics"]["by_distance"]
        },
        by_layer={
            z: sum_metrics([r["metrics"]["by_layer"][z] for r in rows])
            for z in rows[0]["metrics"]["by_layer"]
        },
        by_weather={
            w: sum_metrics([r["metrics"]["exact"] for r in rows if r["weather"] == w])
            for w in sorted(set(r["weather"] for r in rows))
        },
        target_actor_surface=sum_metrics(
            [r["metrics"]["target_actor_surface"] for r in rows]
        ),
        target_actor_by_distance={
            d: sum_metrics([r["metrics"]["target_actor_by_distance"][d] for r in rows])
            for d in rows[0]["metrics"]["target_actor_by_distance"]
        },
        corridor=sum_metrics([r["metrics"]["corridor"] for r in rows]),
        actor_detection=dict(
            eligible=len(actors),
            detected=sum(a["detected"] for a in actors),
            rate=float(np.mean([a["detected"] for a in actors])) if actors else None,
            by_actor_type={
                t: dict(
                    eligible=sum(a["type_id"] == t for a in actors),
                    detected=sum(a["detected"] for a in actors if a["type_id"] == t),
                )
                for t in sorted(set(a["type_id"] for a in actors))
            },
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    a = p.parse_args()
    out = Path(
        f'reports/m5c_{"validation" if a.split=="val" else "absolute_obstacle_detection"}.json'
    )
    if out.exists():
        raise FileExistsError(out)
    study = json.loads(Path("reports/m5c_model_ablation.json").read_text())
    protocol = json.loads(Path("configs/m5c_protocol.json").read_text())
    frozen = None
    if a.split == "test":
        frozen = json.loads(Path("reports/m5c_pretest_decision.json").read_text())
        assert frozen["protocol_sha256"] == file_hash("configs/m5c_protocol.json")
        for r in study["records"]:
            assert frozen["checkpoints"][r["checkpoint"]] == file_hash(r["checkpoint"])
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(a.index, a.split)
    items = [ds[i] for i in range(len(ds))]
    lookup = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
    pairs = read_pairs(a.index, a.split)
    surf = {
        r["sample_id"]: r
        for r in json.loads(Path(a.index, "m5c_surface_index.json").read_text())[
            "records"
        ]
    }
    data = []
    for row in ds.rows:
        r = surf[row["sample_id"]]
        assert file_hash(r["path"]) == r["sha256"]
        with np.load(r["path"]) as z:
            data.append({k: z[k] for k in z.files})
    eval_labels = dict(
        obstacle_valid=np.array([d["valid"] for d in data]),
        surface_semantic=np.stack(
            [
                np.array([d["ground"] for d in data]),
                np.array([d["target"] for d in data]),
            ],
            axis=1,
        ),
    )
    net = encoder_load()
    caches = [cache_rgb(net, stack([it], "inputs")) for it in items]
    labels = stack(items, "targets", "cpu")
    records = []
    art = Path(f"reports/m5c_{a.split}_predictions")
    art.mkdir(exist_ok=False)
    for record in study["records"]:
        name, seed = record["name"], record["seed"]
        net = encoder_load()
        ck = torch.load(record["checkpoint"], weights_only=True)
        head = (
            (
                PositionPrior()
                if name == "P_prior"
                else VisibleObstacleHead() if name == "D_direct" else EvidenceHead()
            )
            .cuda()
            .eval()
        )
        head.load_state_dict(ck["head"])
        if ck["context"] is not None:
            net.context_head.load_state_dict(ck["context"])
        pred = {}
        lat = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for c in caches:
                torch.cuda.synchronize()
                t = time.perf_counter()
                outputs = head(cached_forward(net, c))
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t) * 1000)
                for k, v in outputs.items():
                    pred.setdefault(k, []).append(v.float().cpu())
        pred = {k: torch.cat(v) for k, v in pred.items()}
        q = obstacle_score(pred).numpy()
        score = pred["visible_obstacle"].sigmoid().numpy() if name == "D_direct" else q
        threshold = ck["selection"]["threshold"]
        if a.split == "val":
            _, change_threshold = choose(
                score, pairs, lookup, protocol["evaluation"]["change_thresholds"]
            )
            _, qthreshold = scene_score(
                q, eval_labels, ds.rows, protocol["training"]["thresholds"]
            )
        else:
            selection = frozen["evaluation"][f"{name}_{seed}"]
            change_threshold = selection["change_threshold"]
            qthreshold = selection["q_threshold"]
        frame_metrics = []
        qmetrics = []
        for i, row in enumerate(ds.rows):
            r = surf[row["sample_id"]]
            d = data[i]
            am = [dict(**x, mask=m) for x, m in zip(r["actors"], d["actor_masks"])]
            metric = absolute_metrics(
                score[i], d["target"], d["valid"], threshold, actor_masks=am
            )
            union = d["actor_masks"].any(0)
            metric["target_actor_surface"] = binary_metrics(
                score[i], d["target"], d["valid"] & union, threshold
            )
            x = -20 + (np.arange(160) + 0.5) * 0.5
            y = -30 + (np.arange(120) + 0.5) * 0.5
            distance = np.hypot(x[:, None], y[None, :])
            metric["target_actor_by_distance"] = {
                f"{lo}_{hi}": binary_metrics(
                    score[i],
                    d["target"],
                    d["valid"] & union & ((distance >= lo) & (distance < hi))[None],
                    threshold,
                )
                for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]
            }
            metric["conditional_semantic"] = binary_metrics(
                pred["semantic"][i, 1].sigmoid().numpy(),
                labels["surface_semantic"][i, 1].numpy(),
                labels["obstacle_valid"][i].numpy(),
                0.5,
            )
            metric["mixed_known_surface"] = binary_metrics(
                score[i], d["target"], d["valid"] & d["mixed"], threshold
            )
            frame_metrics.append(
                dict(
                    group=row["group"],
                    sample_id=row["sample_id"],
                    configuration=row["configuration"],
                    weather=r["weather"],
                    metrics=metric,
                )
            )
            qmetrics.append(binary_metrics(q[i], d["target"], d["valid"], qthreshold))
        changes = []
        for row, d in pairs:
            ia, ib = [lookup[row[k]] for k in ["a", "b"]]
            m = change_metrics(score[ia], score[ib], d, change_threshold)
            domain = d["appeared"] | d["vanished"] | d["stable"]
            delta = score[ib] - score[ia]
            m["appearance_spatial_same_height"] = matching(
                (delta > change_threshold) & domain, d["appeared"]
            )
            m["disappearance_spatial_same_height"] = matching(
                (delta < -change_threshold) & domain, d["vanished"]
            )
            changes.append(
                dict(
                    group=row["group"],
                    a=row["a_configuration"],
                    b=row["b_configuration"],
                    metrics=m,
                )
            )
        returns = []
        for group in sorted(set(r["group"] for r in ds.rows)):
            ii = [i for i, r in enumerate(ds.rows) if r["group"] == group]
            i = min(ii, key=lambda j: ds.rows[j]["configuration"])
            j = max(ii, key=lambda j: ds.rows[j]["configuration"])
            valid = data[i]["valid"] & data[j]["valid"]
            neg = valid & ~data[i]["target"] & ~data[j]["target"]
            returns.append(
                dict(
                    group=group,
                    known_ground_cells=int(neg.sum()),
                    empty_fp_initial=int(((score[i] > threshold) & neg).sum()),
                    empty_fp_return=int(((score[j] > threshold) & neg).sum()),
                    mean_absolute_change=float(abs(score[j] - score[i])[valid].mean()),
                    large_change_fraction=float(
                        (abs(score[j] - score[i])[valid] > change_threshold).mean()
                    ),
                )
            )
        path = art / f"m5c_{name}_{seed}.npz"
        np.savez_compressed(path, score=score, q=q)
        result = dict(
            name=name,
            seed=seed,
            threshold=threshold,
            change_threshold=change_threshold,
            q_threshold=qthreshold,
            summary=summarize(frame_metrics),
            frames=frame_metrics,
            paired_summary=aggregate(changes),
            pairs=changes,
            combination_score_secondary=sum_metrics(qmetrics),
            empty_returns=returns,
            predictions=str(path),
            prediction_sha256=file_hash(path),
            cached_context_splat_bev_head_ms=dict(
                p50=float(np.median(lat)),
                p95=float(np.quantile(lat, 0.95)),
                max=max(lat),
            ),
            checkpoint_sha256=file_hash(record["checkpoint"]),
        )
        records.append(result)
        print(
            name,
            seed,
            result["summary"]["overall"],
            result["paired_summary"]["mean_scene_iou"],
            flush=True,
        )
    atomic_json(
        out,
        dict(
            kind=f"actual_RTX4090_{a.split}_singleframe_inference_offline_evaluation",
            records=records,
            independent_groups=sorted(set(r["group"] for r in ds.rows)),
            source_sha256=file_hash(__file__),
            index_sha256=file_hash(Path(a.index, "index.json")),
            scope="absolute rendered surface localization on heldout known Ground/Obstacle support; full BEV unknown predictions reported separately, not counted as FP or safe negatives; no exact volume claim; same-renderer reference",
            pretest_sha256=(
                file_hash("reports/m5c_pretest_decision.json") if frozen else None
            ),
        ),
    )
    if a.split == "val":
        atomic_json(
            "reports/m5c_pretest_decision.json",
            dict(
                kind="frozen_before_new_test_capture_and_evaluation",
                protocol_sha256=file_hash("configs/m5c_protocol.json"),
                checkpoints={
                    r["checkpoint"]: file_hash(r["checkpoint"])
                    for r in study["records"]
                },
                evaluation={
                    f"{r['name']}_{r['seed']}": {
                        k: r[k]
                        for k in ["threshold", "change_threshold", "q_threshold"]
                    }
                    for r in records
                },
                research_decisions="No further architecture, loss, threshold, sampling or recipe changes after this freeze. D direct candidate predeclared. Test evaluated once. All models reported; no posttest selection.",
                code_sha256={
                    str(f): file_hash(f)
                    for pattern in [
                        "tools/m5c*.py",
                        "models/m5c*.py",
                        "training/m5c*.py",
                        "datasets/m5c*.py",
                        "diagnostics/m5c*.py",
                    ]
                    for f in Path(".").glob(pattern)
                },
            ),
        )


if __name__ == "__main__":
    main()
