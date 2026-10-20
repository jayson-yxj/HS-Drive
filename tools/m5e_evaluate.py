"""Fixed singleframe models, witness tradeoff curves, absolute localization and unknown."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import encoder_load, stack
from tools.m5e_common import pairs as read_pairs, load_checkpoint
from tools.m5d_learning import predict
from models.m5e_paths import depth_logits
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5d_prior import ExistencePrior
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb
from diagnostics.m5d_metrics import (
    witness_metrics,
    aggregate_witness,
    voxel_3d_matching,
)
from diagnostics.m5c_localization import absolute_metrics
from diagnostics.m5a_metrics import binary_metrics
from diagnostics.m5b_metrics import change_metrics, aggregate
from tools.m5c_evaluate import sum_metrics
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    a = p.parse_args()
    out = Path(
        f'reports/m5e_{"validation" if a.split=="val" else "presence_absence"}.json'
    )
    if out.exists():
        raise FileExistsError(out)
    study = json.loads(Path("reports/m5e_model_ablation.json").read_text())
    records = study["records"][:]
    for seed in [81,82]:
        for model in ['A_matched','B_existence']:
            records.append(dict(name='Historical_'+model,seed=seed,stride=16,checkpoint=f'runs/m5d_learning/m5d_{model}_{seed}.pt'))
    if a.split == "test":
        freeze = json.loads(Path("reports/m5e_pretest_decision.json").read_text())
        assert freeze["protocol_sha256"] == file_hash("configs/m5e_protocol.json")
        for r in records:
            assert freeze["checkpoints"][r["checkpoint"]] == file_hash(r["checkpoint"])
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset(a.index, a.split)
    lookup = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
    items = [ds[i] for i in range(len(ds))]
    pairs = read_pairs(a.index, a.split)
    surf = {
        r["sample_id"]: r
        for r in json.loads(Path(a.index, "m5e_surface_index.json").read_text())[
            "records"
        ]
    }
    data = []
    for r in ds.rows:
        ref = surf[r["sample_id"]]
        assert file_hash(ref["path"]) == ref["sha256"]
        with np.load(ref["path"]) as z:
            data.append({k: z[k] for k in z.files})
    net = encoder_load()
    cache = {
        s: [fn(net, stack([it], "inputs")) for it in items]
        for s, fn in [(16, cache_rgb), (8, cache_p8)]
    }
    art = Path(f"reports/m5e_{a.split}_predictions")
    art.mkdir(exist_ok=False)
    results = []
    for record in records:
        name, seed = record["name"], record["seed"]
        net = encoder_load()
        ck = load_checkpoint(record["checkpoint"])
        head = (
            (ExistencePrior() if name == "P_prior" else VisibleObstacleHead())
            .cuda()
            .eval()
        )
        head.load_state_dict(ck["head"])
        if ck["context"] is not None:
            net.context_head.load_state_dict(ck["context"])
        if 'depth' in ck:net.depth_head.load_state_dict(ck['depth'])
        cc=cache[record['stride']]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            modelcache=[dict(c,probs=depth_logits(net,c).float().softmax(2)) for c in cc]
        score = predict(net, head, modelcache)
        th = ck["selection"]["threshold"]
        frames = []
        for i, row in enumerate(ds.rows):
            d = data[i]
            actor = d["actor_masks"].any(0)
            am = [
                dict(**r, mask=m)
                for r, m in zip(surf[row["sample_id"]]["actors"], d["actor_masks"])
            ]
            metric = absolute_metrics(
                score[i], d["target"], d["valid"], th, actor_masks=am
            )
            metric["voxel_center_3d"] = voxel_3d_matching(
                (score[i] > th) & d["valid"], d["target"] & d["valid"]
            )
            metric["background_only"] = binary_metrics(
                score[i], d["target"], d["valid"] & ~actor, th
            )
            metric["placed_actor_only"] = binary_metrics(
                score[i], d["target"], d["valid"] & actor, th
            )
            metric["mixed_known"] = binary_metrics(
                score[i], d["target"], d["valid"] & d["mixed"], th
            )
            frames.append(
                dict(
                    group=row["group"],
                    configuration=row["configuration"],
                    sample_id=row["sample_id"],
                    weather=surf[row["sample_id"]]["weather"],
                    metrics=metric,
                )
            )
        pairrows = []
        legacy = []
        changes = []
        for r, d in pairs:
            pa, pb = score[lookup[r["a"]]], score[lookup[r["b"]]]
            pairrows.append(
                dict(
                    group=r["group"],
                    a=r["a_configuration"],
                    b=r["b_configuration"],
                    witness=witness_metrics(pa, pb, d, th),
                )
            )
            legacy.append(
                dict(group=r["group"], witness=witness_metrics(pa, pb, d, th, True))
            )
            changes.append(
                dict(group=r["group"], metrics=change_metrics(pa, pb, d, 0.1))
            )
        curves = []
        for t in np.linspace(0, 1, 41):
            rr = [
                dict(
                    group=r["group"],
                    witness=witness_metrics(
                        score[lookup[r["a"]]], score[lookup[r["b"]]], d, float(t)
                    ),
                )
                for r, d in pairs
            ]
            curves.append(dict(threshold=float(t), **aggregate_witness(rr)))
        returns = []
        for g in sorted(set(r["group"] for r in ds.rows)):
            ids = [i for i, r in enumerate(ds.rows) if r["group"] == g]
            i = min(ids, key=lambda i: ds.rows[i]["configuration"])
            j = max(ids, key=lambda i: ds.rows[i]["configuration"])
            m = data[i]["valid"] & data[j]["valid"]
            returns.append(
                dict(
                    group=g,
                    mean_abs_score_change=float(abs(score[i] - score[j])[m].mean()),
                    threshold_flip=int(((score[i] > th) != (score[j] > th))[m].sum()),
                    ground_cells=int((m & ~data[i]["target"]).sum()),
                    ground_fp_initial=int(
                        ((score[i] > th) & m & ~data[i]["target"]).sum()
                    ),
                    ground_fp_return=int(
                        ((score[j] > th) & m & ~data[j]["target"]).sum()
                    ),
                )
            )
        path = art / f"m5e_{name}_{seed}.npz"
        np.savez_compressed(path, score=score)
        result = dict(
            name=name,
            seed=seed,
            stride=record["stride"],
            threshold=th,
            presence_absence=aggregate_witness(pairrows),
            legacy_same_definition=aggregate_witness(legacy),
            witnesses=pairrows,
            threshold_curve=curves,
            change=aggregate(changes),
            frames=frames,
            absolute=dict(
                overall=sum_metrics([r["metrics"]["exact"] for r in frames]),
                background_only=sum_metrics(
                    [r["metrics"]["background_only"] for r in frames]
                ),
                placed_actor_only=sum_metrics(
                    [r["metrics"]["placed_actor_only"] for r in frames]
                ),
                corridor=sum_metrics([r["metrics"]["corridor"] for r in frames]),
                by_scene={
                    g: sum_metrics(
                        [r["metrics"]["exact"] for r in frames if r["group"] == g]
                    )
                    for g in sorted(set(r["group"] for r in frames))
                },
            ),
            empty_returns=returns,
            predictions=str(path),
            prediction_sha256=file_hash(path),
            checkpoint_sha256=file_hash(record["checkpoint"]),
        )
        results.append(result)
        print(name, seed, result["presence_absence"], flush=True)
    atomic_json(
        out,
        dict(
            kind=f"actual_GPU_{a.split}_singleframe_RGB_evaluation",
            records=results,
            source_sha256=file_hash(__file__),
            index_sha256=file_hash(Path(a.index, "index.json")),
            groups=sorted(set(r["group"] for r in ds.rows)),
            limits="Presence/absence on certified point-witness support, no full-cell free claim. Legacy side-by-side uses exact original mask definition. Unknown high scores uncertified, not FP. Test curves descriptive only, no threshold reselection. Historical models keep old frozen thresholds.",
        ),
    )
    if a.split == "val":
        atomic_json(
            "reports/m5e_pretest_decision.json",
            dict(
                protocol_sha256=file_hash("configs/m5e_protocol.json"),
                checkpoints={
                    r["checkpoint"]: file_hash(r["checkpoint"]) for r in records
                },
                models=[
                    dict(
                        name=r["name"],
                        seed=r["seed"],
                        stride=r["stride"],
                        checkpoint=r["checkpoint"],
                    )
                    for r in records
                ],
                code_sha256={
                    str(f): file_hash(f)
                    for pat in [
                        "tools/m5e*.py",
                        "models/m5e*.py",
                        "datasets/m5e*.py",
                        "diagnostics/m5e*.py",
                        "training/m5e*.py",
                    ]
                    for f in Path(".").glob(pat)
                },
                decisions="No changes after freeze; A/B/C/D all reported, no additional training/threshold changes. P8 diagnosticfactorization only.",
            ),
        )


if __name__ == "__main__":
    main()
