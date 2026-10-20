"""Matched from-scratch Head A/B/C and single-variable C+pair study; val only."""

import argparse, json, time
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from models.m48_evidence_head import EvidenceHead, PositionPrior
from models.m5c_obstacle_head import VisibleObstacleHead
from training.m5c_obstacle_loss import visible_obstacle_loss
from diagnostics.m5c_localization import scene_score
from models.m5b_cached_encoder import cache_rgb, cached_forward
from tools.m5a_learning import encoder_load, stack
from training.m48_evidence_loss import evidence_loss
from training.m5b_change_loss import obstacle_score, change_loss
from diagnostics.m5b_metrics import change_metrics, aggregate
from simulator.recorder import atomic_json, file_hash
from tools.m47_common import freeze_sources
from training.runtime import prepare_cuda


def read_pairs(index, split):
    rows = json.loads(Path(index, "m5c_pairs.json").read_text())["pairs"]
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        if file_hash(r["path"]) != r["sha256"]:
            raise ValueError("pair checksum")
        with np.load(r["path"]) as z:
            d = {k: z[k] for k in z.files}
        out.append((r, d))
    return out


def choose(prob, pairs, lookup, thresholds):
    scores = []
    for th in thresholds:
        rows = [
            dict(
                group=r["group"],
                metrics=change_metrics(
                    prob[lookup[r["a"]]], prob[lookup[r["b"]]], d, th
                ),
            )
            for r, d in pairs
        ]
        scores.append((aggregate(rows)["mean_scene_iou"], th))
    return max(scores, key=lambda x: (x[0], -x[1]))


def predict(net, head, caches, frozen=None):
    output = []
    with torch.no_grad():
        for i, c in enumerate(caches):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = head(frozen[i] if frozen is not None else cached_forward(net, c))
            output.append(
                (
                    p["visible_obstacle"].float().sigmoid()
                    if "visible_obstacle" in p
                    else obstacle_score(p)
                )[0]
                .cpu()
                .numpy()
            )
    return np.array(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index", default="/root/gpufree-data/hs-drive-data/processed/m5c_index"
    )
    p.add_argument("--run", default="runs/m5c_learning")
    a = p.parse_args()
    root = Path(a.run)
    root.mkdir(exist_ok=False)
    prepare_cuda()
    torch.set_num_threads(2)
    protocol = json.loads(Path("configs/m5c_protocol.json").read_text())
    cfg = protocol["training"]
    start_all = time.perf_counter()
    net = encoder_load()
    torch.cuda.reset_peak_memory_stats()
    sources = freeze_sources(
        root / "source",
        [
            __file__,
            "models/m5b_cached_encoder.py",
            "training/m5b_change_loss.py",
            "diagnostics/m5b_metrics.py",
            "diagnostics/m5c_localization.py",
            "models/m5c_obstacle_head.py",
            "training/m5c_obstacle_loss.py",
            "configs/m5c_protocol.json",
            "datasets/m5a_training.py",
            "training/m48_evidence_loss.py",
        ],
    )
    ds = {s: VisualEvidenceDataset(a.index, s) for s in ["train", "val"]}
    items = {s: [d[i] for i in range(len(d))] for s, d in ds.items()}
    lookup = {
        s: {r["sample_id"]: i for i, r in enumerate(d.rows)} for s, d in ds.items()
    }
    pairs = {s: read_pairs(a.index, s) for s in ds}
    labels = {s: [stack([it], "targets") for it in v] for s, v in items.items()}
    caches = {
        s: [cache_rgb(net, stack([it], "inputs")) for it in v] for s, v in items.items()
    }
    with torch.no_grad():
        features = {
            s: [cached_forward(net, c).detach() for c in cs] for s, cs in caches.items()
        }
    evalrows = json.loads(Path(a.index, "m5c_surface_index.json").read_text())[
        "records"
    ]
    evalmap = {r["sample_id"]: r for r in evalrows}
    evaldata = []
    for r in ds["val"].rows:
        assert (
            file_hash(evalmap[r["sample_id"]]["path"])
            == evalmap[r["sample_id"]]["sha256"]
        )
        with np.load(evalmap[r["sample_id"]]["path"]) as z:
            evaldata.append((z["target"], z["valid"]))
    eval_labels = dict(
        obstacle_valid=np.array([v for t, v in evaldata]),
        surface_semantic=np.stack(
            [
                np.zeros_like(np.array([t for t, v in evaldata])),
                np.array([t for t, v in evaldata]),
            ],
            axis=1,
        ),
    )
    precompute = dict(
        elapsed_s=time.perf_counter() - start_all,
        peak_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    records = []
    for seed in cfg["seeds"]:
        schedule = np.random.default_rng(seed).integers(
            0, len(pairs["train"]), size=cfg["steps"]
        )
        np.save(root / f"m5c_schedule_{seed}.npy", schedule)
        for name in cfg["models"]:
            net = encoder_load()
            torch.manual_seed(seed)
            head = (
                PositionPrior()
                if name == "P_prior"
                else VisibleObstacleHead() if name == "D_direct" else EvidenceHead()
            ).cuda()
            context = name in ["B_context", "C_pair", "D_direct"]
            for v in net.context_head.parameters():
                v.requires_grad_(context)
            groups = [
                dict(
                    params=head.parameters(),
                    lr=cfg["prior_lr"] if name == "P_prior" else cfg["head_lr"],
                )
            ]
            if context:
                groups.append(
                    dict(params=net.context_head.parameters(), lr=cfg["context_lr"])
                )
            optimizer = torch.optim.Adam(groups)
            best = -1
            history = []
            gradmax = 0.0
            finite = True
            frozen_gradients_absent = True
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for step in range(cfg["steps"] + 1):
                if step % cfg["validate_every"] == 0:
                    probs = predict(
                        net, head, caches["val"], None if context else features["val"]
                    )
                    score, threshold = scene_score(
                        probs, eval_labels, ds["val"].rows, cfg["thresholds"]
                    )
                    history.append(
                        dict(step=step, val_absolute_iou=score, threshold=threshold)
                    )
                    if score > best:
                        best = score
                        selected = dict(
                            step=step, val_absolute_iou=score, threshold=threshold
                        )
                        torch.save(
                            dict(
                                head=head.state_dict(),
                                context=(
                                    net.context_head.state_dict() if context else None
                                ),
                                seed=seed,
                                model=name,
                                selection=selected,
                            ),
                            root / f"m5c_{name}_{seed}.pt",
                        )
                    print(name, seed, step, score, flush=True)
                if step == cfg["steps"]:
                    break
                r, pair = pairs["train"][schedule[step]]
                ids = [lookup["train"][r[k]] for k in ["a", "b"]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = torch.cat(
                        [
                            (
                                cached_forward(net, caches["train"][i])
                                if context
                                else features["train"][i]
                            )
                            for i in ids
                        ]
                    )
                    out = head(f)
                lab = {
                    k: torch.cat([labels["train"][i][k] for i in ids])
                    for k in labels["train"][ids[0]]
                }
                loss, parts = evidence_loss(out, lab)
                extra = loss * 0
                if name in ["C_pair", "D_direct"]:
                    extra, _ = change_loss(
                        out,
                        {
                            k: torch.from_numpy(pair[k]).cuda()
                            for k in ["appeared", "vanished", "stable"]
                        },
                    )
                    loss = loss + cfg["pair_weight"] * extra
                direct = loss * 0
                if name == "D_direct":
                    direct, _ = visible_obstacle_loss(out, lab)
                    loss = loss + cfg["direct_weight"] * direct
                loss.backward()
                grad = max(
                    float(v.grad.abs().max())
                    for v in list(head.parameters())
                    + list(net.context_head.parameters())
                    if v.grad is not None
                )
                gradmax = max(gradmax, grad)
                finite &= bool(np.isfinite(grad)) and bool(torch.isfinite(loss))
                frozen_gradients_absent &= all(
                    p.grad is None
                    for n, p in net.named_parameters()
                    if not (context and n.startswith("context_head."))
                )
                optimizer.step()
                if step % 50 == 0:
                    history.append(
                        dict(
                            step=step,
                            total=float(loss.detach()),
                            pair=float(extra.detach()),
                            direct=float(direct.detach()),
                            tasks={k: float(v.detach()) for k, v in parts.items()},
                        )
                    )
            torch.cuda.synchronize()
            ck = root / f"m5c_{name}_{seed}.pt"
            record = dict(
                name=name,
                seed=seed,
                selected=selected,
                steps=cfg["steps"],
                history=history,
                elapsed_s=time.perf_counter() - start,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                trainable_parameters=sum(v.numel() for v in head.parameters())
                + sum(v.numel() for v in net.parameters() if v.requires_grad),
                finite_gradients=finite,
                frozen_gradients_absent=frozen_gradients_absent,
                schedule_sha256=file_hash(root / f"m5c_schedule_{seed}.npy"),
                max_gradient=gradmax,
                checkpoint=str(ck),
                sha256=file_hash(ck),
            )
            records.append(record)
            atomic_json(root / "m5c_progress.json", dict(records=records))
    atomic_json(
        "reports/m5c_model_ablation.json",
        dict(
            kind="actual_RTX4090_matched_training_val_only",
            records=records,
            precompute=precompute,
            sources=sources,
            protocol_sha256=file_hash("configs/m5c_protocol.json"),
            index_sha256=file_hash(Path(a.index, "index.json")),
            gpu=torch.cuda.get_device_name(),
            encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
            test_opened=False,
            geometry_fp32=True,
            amp_bf16=True,
            notes="P lr .03; all heads .001; B/C/D context .0003. 400 same paired updates. D adds direct head and auxiliary BCE with1/3weight; C pair weight .2. Selection on heldout-pixel absolute surface val IoU. GT never in cache. Unknown not negative.",
        ),
    )


if __name__ == "__main__":
    main()
