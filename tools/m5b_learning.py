"""Matched from-scratch Head A/B/C and single-variable C+pair study; val only."""

import argparse, json, time
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from models.m48_evidence_head import EvidenceHead, PositionPrior
from models.m5b_cached_encoder import cache_rgb, cached_forward
from tools.m5a_learning import encoder_load, stack
from training.m48_evidence_loss import evidence_loss
from training.m5b_change_loss import obstacle_score, change_loss
from diagnostics.m5b_metrics import change_metrics, aggregate
from simulator.recorder import atomic_json, file_hash
from tools.m47_common import freeze_sources
from training.runtime import prepare_cuda


def read_pairs(index, split):
    rows = json.loads(Path(index, "m5b_pairs.json").read_text())["pairs"]
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
            output.append(obstacle_score(p)[0].cpu().numpy())
    return np.array(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index", default="/root/gpufree-data/hs-drive-data/processed/m5b_index"
    )
    p.add_argument("--run", default="runs/m5b_learning")
    a = p.parse_args()
    root = Path(a.run)
    root.mkdir(exist_ok=False)
    prepare_cuda()
    torch.set_num_threads(2)
    protocol = json.loads(Path("configs/m5b_protocol.json").read_text())
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
    precompute = dict(
        elapsed_s=time.perf_counter() - start_all,
        peak_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    records = []
    for seed in cfg["seeds"]:
        schedule = np.random.default_rng(seed).integers(
            0, len(pairs["train"]), size=cfg["steps"]
        )
        np.save(root / f"m5b_schedule_{seed}.npy", schedule)
        for name in cfg["models"]:
            net = encoder_load()
            torch.manual_seed(seed)
            head = (PositionPrior() if name == "A_prior" else EvidenceHead()).cuda()
            context = name.startswith("C_")
            for v in net.context_head.parameters():
                v.requires_grad_(context)
            groups = [
                dict(
                    params=head.parameters(),
                    lr=cfg["prior_lr"] if name == "A_prior" else cfg["head_lr"],
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
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for step in range(cfg["steps"] + 1):
                if step % cfg["validate_every"] == 0:
                    probs = predict(
                        net, head, caches["val"], None if context else features["val"]
                    )
                    score, threshold = choose(
                        probs,
                        pairs["val"],
                        lookup["val"],
                        protocol["evaluation"]["thresholds"],
                    )
                    history.append(
                        dict(step=step, val_change_iou=score, threshold=threshold)
                    )
                    if score > best:
                        best = score
                        selected = dict(
                            step=step, val_change_iou=score, threshold=threshold
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
                            root / f"m5b_{name}_{seed}.pt",
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
                if name == "C_pair":
                    extra, _ = change_loss(
                        out,
                        {
                            k: torch.from_numpy(pair[k]).cuda()
                            for k in ["appeared", "vanished", "stable"]
                        },
                    )
                    loss = loss + cfg["pair_weight"] * extra
                loss.backward()
                grad = max(
                    float(v.grad.abs().max())
                    for v in list(head.parameters())
                    + list(net.context_head.parameters())
                    if v.grad is not None
                )
                gradmax = max(gradmax, grad)
                finite &= bool(np.isfinite(grad)) and bool(torch.isfinite(loss))
                optimizer.step()
                if step % 50 == 0:
                    history.append(
                        dict(
                            step=step,
                            total=float(loss.detach()),
                            pair=float(extra.detach()),
                            tasks={k: float(v.detach()) for k, v in parts.items()},
                        )
                    )
            torch.cuda.synchronize()
            ck = root / f"m5b_{name}_{seed}.pt"
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
                max_gradient=gradmax,
                checkpoint=str(ck),
                sha256=file_hash(ck),
            )
            records.append(record)
            atomic_json(root / "m5b_progress.json", dict(records=records))
    atomic_json(
        "reports/m5b_context_ablation.json",
        dict(
            kind="actual_RTX4090_matched_training_val_only",
            records=records,
            precompute=precompute,
            sources=sources,
            protocol_sha256=file_hash("configs/m5b_protocol.json"),
            encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
            test_opened=False,
            geometry_fp32=True,
            amp_bf16=True,
            notes="A lr .03 vs B/C head .001. C context .0003. All400 updates same pairs. C_pair only extra loss. No validation labels in train. Cache contains frozen RGB features/depth prediction/calibration only.",
        ),
    )


if __name__ == "__main__":
    main()
