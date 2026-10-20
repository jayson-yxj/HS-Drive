"""Matched existence-loss and actual-P8 study; validation selection, sealed test excluded."""

import argparse, json, time
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import encoder_load, stack
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5d_prior import ExistencePrior
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb, cached_forward
from training.m48_evidence_loss import evidence_loss
from training.m5b_change_loss import change_loss
from training.m5c_obstacle_loss import visible_obstacle_loss
from training.m5d_existence_loss import existence_loss
from diagnostics.m5d_metrics import choose
from tools.m47_common import freeze_sources
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def read_pairs(index, split):
    result = []
    for r in json.loads(Path(index, "m5d_pairs.json").read_text())["pairs"]:
        if r["split"] != split:
            continue
        assert file_hash(r["path"]) == r["sha256"]
        with np.load(r["path"]) as z:
            result.append((r, {k: z[k] for k in z.files}))
    return result


def predict(net, head, caches):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c in caches:
            out.append(
                head(cached_forward(net, c))["visible_obstacle"][0]
                .float()
                .sigmoid()
                .cpu()
                .numpy()
            )
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index", default="/root/gpufree-data/hs-drive-data/processed/m5d_index"
    )
    p.add_argument("--run", default="runs/m5d_learning")
    a = p.parse_args()
    root = Path(a.run)
    root.mkdir(exist_ok=False)
    cfg = json.loads(Path("configs/m5d_protocol.json").read_text())["training"]
    prepare_cuda()
    torch.set_num_threads(2)
    start = time.perf_counter()
    net = encoder_load()
    torch.cuda.reset_peak_memory_stats()
    sources = freeze_sources(
        root / "source",
        [
            str(f)
            for pat in [
                "tools/m5d_learning.py",
                "training/m5d*.py",
                "models/m5d*.py",
                "diagnostics/m5d*.py",
                "datasets/m5d*.py",
                "configs/m5d_protocol.json",
            ]
            for f in Path(".").glob(pat)
        ],
    )
    ds = {s: VisualEvidenceDataset(a.index, s) for s in ["train", "val"]}
    items = {s: [d[i] for i in range(len(d))] for s, d in ds.items()}
    lookup = {
        s: {r["sample_id"]: i for i, r in enumerate(d.rows)} for s, d in ds.items()
    }
    pairs = {s: read_pairs(a.index, s) for s in ds}
    labels = {s: [stack([i], "targets") for i in its] for s, its in items.items()}
    cache = {}
    for stride, fn in [(16, cache_rgb), (8, cache_p8)]:
        cache[stride] = {
            s: [fn(net, stack([i], "inputs")) for i in its] for s, its in items.items()
        }
    precompute = dict(
        seconds=time.perf_counter() - start,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    records = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        groups = sorted(set(r["group"] for r, d in pairs["train"]))
        schedule = []
        while len(schedule) < cfg["steps"]:
            for g in rng.permutation(groups):
                indices = [
                    i for i, (r, d) in enumerate(pairs["train"]) if r["group"] == g
                ]
                schedule.append(int(rng.choice(indices)))
        schedule = np.array(schedule[: cfg["steps"]])
        np.save(root / f"m5d_schedule_{seed}.npy", schedule)
        for name in cfg["models"]:
            net = encoder_load()
            torch.manual_seed(seed)
            head = (
                ExistencePrior() if name == "P_prior" else VisibleObstacleHead()
            ).cuda()
            context = name != "P_prior"
            stride = 8 if name == "S8_stable" else 16
            for v in net.context_head.parameters():
                v.requires_grad_(context)
            params = [
                dict(
                    params=head.parameters(),
                    lr=cfg["prior_lr"] if name == "P_prior" else cfg["head_lr"],
                )
            ]
            if context:
                params.append(
                    dict(params=net.context_head.parameters(), lr=cfg["context_lr"])
                )
            opt = torch.optim.Adam(params)
            history = []
            best = -1
            maxgrad = 0.0
            finite = True
            frozen = True
            direct_gradmax = 0.0
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for step in range(cfg["steps"] + 1):
                if step % cfg["validate_every"] == 0:
                    prob = predict(net, head, cache[stride]["val"])
                    score, th = choose(
                        prob, pairs["val"], lookup["val"], cfg["thresholds"]
                    )
                    h = dict(
                        step=step, val_presence_absence_harmonic=score, threshold=th
                    )
                    history.append(h)
                    if score > best:
                        best = score
                        selected = h
                        torch.save(
                            dict(
                                head=head.state_dict(),
                                context=(
                                    net.context_head.state_dict() if context else None
                                ),
                                seed=seed,
                                model=name,
                                stride=stride,
                                selection=selected,
                            ),
                            root / f"m5d_{name}_{seed}.pt",
                        )
                    print(name, seed, step, score, th, flush=True)
                if step == cfg["steps"]:
                    break
                r, pair = pairs["train"][schedule[step]]
                ids = [lookup["train"][r[k]] for k in ["a", "b"]]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = head(
                        torch.cat(
                            [
                                cached_forward(net, cache[stride]["train"][i])
                                for i in ids
                            ]
                        )
                    )
                lab = {
                    k: torch.cat([labels["train"][i][k] for i in ids])
                    for k in labels["train"][ids[0]]
                }
                loss, parts = evidence_loss(out, lab)
                oldpair = {
                    k: torch.from_numpy(pair["legacy_" + k]).cuda()
                    for k in ["appeared", "vanished", "stable"]
                }
                pairloss, _ = change_loss(out, oldpair)
                direct, _ = visible_obstacle_loss(out, lab)
                loss = loss + 0.2 * pairloss + direct / 3
                extra = loss * 0
                eparts = {}
                if name != "A_matched":
                    newpair = {
                        k: torch.from_numpy(pair[k]).cuda()
                        for k in ["appeared", "vanished", "stable"]
                    }
                    extra, eparts = existence_loss(
                        out,
                        newpair,
                        stable=name in ["C_stable", "S8_stable", "P_prior"],
                    )
                    loss = loss + extra
                loss.backward()
                grad = max(
                    float(v.grad.abs().max())
                    for v in list(head.parameters())
                    + list(net.context_head.parameters())
                    if v.grad is not None
                )
                maxgrad = max(maxgrad, grad)
                finite &= bool(torch.isfinite(loss)) and np.isfinite(grad)
                frozen &= all(
                    v.grad is None
                    for n, v in net.named_parameters()
                    if not (context and n.startswith("context_head."))
                )
                if name != "P_prior":
                    direct_gradmax = max(
                        direct_gradmax,
                        float(head.visible_obstacle.weight.grad.abs().max()),
                    )
                opt.step()
                if step % 50 == 0:
                    history.append(
                        dict(
                            step=step,
                            total=float(loss.detach()),
                            old_pair=float(pairloss.detach()),
                            direct_surface=float(direct.detach()),
                            direct_existence=float(extra.detach()),
                            tasks={k: float(v.detach()) for k, v in parts.items()},
                            existence={
                                k: float(v.detach())
                                for k, v in eparts.items()
                                if k != "counts"
                            },
                        )
                    )
            torch.cuda.synchronize()
            ck = root / f"m5d_{name}_{seed}.pt"
            record = dict(
                name=name,
                seed=seed,
                stride=stride,
                selected=selected,
                steps=cfg["steps"],
                history=history,
                elapsed_s=time.perf_counter() - start,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                trainable_parameters=sum(v.numel() for v in head.parameters())
                + sum(v.numel() for v in net.parameters() if v.requires_grad),
                finite_gradients=bool(finite),
                frozen_gradients_absent=frozen,
                max_gradient=maxgrad,
                direct_output_max_gradient=direct_gradmax,
                checkpoint=str(ck),
                sha256=file_hash(ck),
                schedule_sha256=file_hash(root / f"m5d_schedule_{seed}.npy"),
            )
            records.append(record)
            atomic_json(root / "m5d_progress.json", dict(records=records))
    atomic_json(
        "reports/m5d_model_ablation.json",
        dict(
            kind="actual_RTX4090_matched_400step_train_val_only",
            records=records,
            precompute=precompute,
            sources=sources,
            protocol_sha256=file_hash("configs/m5d_protocol.json"),
            index_sha256=file_hash(Path(a.index, "index.json")),
            encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
            test_opened=False,
            gpu=torch.cuda.get_device_name(),
            geometry_fp32=True,
            amp_bf16=True,
            notes="A/B/C/S8 identical head seed/init,groupbalanced same pair sequence. A matches old base losses; B adds direct existence BCE+.1ranking; C adds .2stableconsistency. P8 true layer2+lateral fusion, existing frozen depth head evaluated at8. No geometry auxiliary D approved.",
        ),
    )


if __name__ == "__main__":
    main()
