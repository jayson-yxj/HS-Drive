"""Fixed grouped A/B study; conditional context-only C. No test selection."""

import argparse, json, time, copy
from pathlib import Path
import numpy as np
import torch
from datasets.m5a_training import VisualEvidenceDataset
from models.m48_evidence_head import EvidenceHead, PositionPrior
from training.m48_evidence_loss import evidence_loss, evidence_targets
from training.runner import initialize, load_checkpoint
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash
from tools.m47_common import freeze_sources


def encoder_load():
    cfg = json.loads(Path("runs/m4_expanded_joint/manifest.json").read_text())["config"]
    net = initialize(cfg, pretrained=False).cuda().eval()
    load_checkpoint(Path("runs/m4_expanded_joint/best.pt"), net)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def stack(items, field, device="cuda"):
    return {
        k: torch.stack([getattr(x, field)[k] for x in items]).to(device)
        for k in getattr(items[0], field)
    }


def task_arrays(pred, labels):
    targets = evidence_targets(labels)
    return dict(
        surface=(pred["surface"], *targets["surface"]),
        free=(pred["free"], *targets["free"]),
        ground=(
            pred["semantic"][:, 0],
            targets["semantic"][0][:, 0],
            targets["semantic"][1][:, 0],
        ),
        obstacle=(
            pred["semantic"][:, 1],
            targets["semantic"][0][:, 1],
            targets["semantic"][1][:, 1],
        ),
        other=(
            pred["semantic"][:, 2],
            targets["semantic"][0][:, 2],
            targets["semantic"][1][:, 2],
        ),
    )


def selection(head, features, labels, rows):
    values = []
    by_task = {k: [] for k in ["surface", "obstacle"]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        preds = {k: [] for k in ["surface", "free", "semantic"]}
        for i in range(0, len(features), 2):
            for k, v in head(features[i : i + 2]).items():
                preds[k].append(v.float())
        preds = {k: torch.cat(v) for k, v in preds.items()}
        for group in sorted(set(r["group"] for r in rows)):
            ids = [i for i, r in enumerate(rows) if r["group"] == group]
            for key, (p, t, m) in task_arrays(preds, labels).items():
                if key not in by_task:
                    continue
                p = p[ids] > 0
                t = t[ids].bool()
                m = m[ids]
                tp = (p & t & m).sum()
                union = ((p | t) & m).sum()
                by_task[key].append(float(tp / union.clamp_min(1)))
    means = {k: float(np.mean(v)) for k, v in by_task.items()}
    return float(np.mean(list(means.values()))), means


def encode(net, items):
    fs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in items:
            fs.append(net(**stack([item], "inputs"))["bev_current"].detach().half())
    return torch.cat(fs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index", default="/root/gpufree-data/hs-drive-data/processed/m5a_index"
    )
    p.add_argument("--run", default="runs/m5a_learning")
    p.add_argument("--output", default="reports/m5a_training.json")
    a = p.parse_args()
    root = Path(a.run)
    if root.exists() or Path(a.output).exists():
        raise FileExistsError("new run required")
    root.mkdir()
    prepare_cuda()
    torch.set_num_threads(2)
    protocol = json.loads(Path("configs/m5a_protocol.json").read_text())
    cfg = protocol["training"]
    sources = freeze_sources(
        root / "source",
        [
            __file__,
            "datasets/m5a_training.py",
            "training/m48_evidence_loss.py",
            "models/m48_evidence_head.py",
            "models/depth_bev.py",
            "models/view_transform.py",
        ],
    )
    datasets = {s: VisualEvidenceDataset(a.index, s) for s in ["train", "val"]}
    items = {s: [d[i] for i in range(len(d))] for s, d in datasets.items()}
    labels = {s: stack(it, "targets") for s, it in items.items()}
    net = encoder_load()
    dtypes = []
    hook = net.splat.register_forward_hook(
        lambda module, args, out: dtypes.append([str(x.dtype) for x in out])
    )
    features = {s: encode(net, it) for s, it in items.items()}
    hook.remove()
    assert all(t == ["torch.float32", "torch.float32"] for t in dtypes)
    records = []
    chosen = {}
    for seed in cfg["seeds"]:
        schedule = np.random.default_rng(seed).integers(
            0, len(items["train"]), size=(cfg["steps"], cfg["batch_size"])
        )
        np.save(root / f"m5a_schedule_{seed}.npy", schedule)
        for name in ["A_prior", "B_frozen"]:
            torch.manual_seed(seed)
            head = (PositionPrior() if name == "A_prior" else EvidenceHead()).cuda()
            lr = 0.05 if name == "A_prior" else 0.003
            opt = torch.optim.Adam(head.parameters(), lr=lr)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            history = []
            best = -1
            finite = True
            max_grad = 0
            for step in range(cfg["steps"] + 1):
                if step % cfg["validation_every"] == 0:
                    score, per = selection(
                        head, features["val"], labels["val"], datasets["val"].rows
                    )
                    history.append(dict(step=step, selection=score, val=per))
                    if score > best:
                        best = score
                        chosen[name, seed] = dict(score=score, per=per, step=step)
                        torch.save(
                            dict(
                                head=head.state_dict(), seed=seed, step=step, model=name
                            ),
                            root / f"m5a_{name}_{seed}.pt",
                        )
                if step == cfg["steps"]:
                    break
                ix = schedule[step]
                lab = {k: v[ix] for k, v in labels["train"].items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = head(features["train"][ix])
                loss, parts = evidence_loss(out, lab)
                loss.backward()
                grad = max(
                    float(v.grad.abs().max())
                    for v in head.parameters()
                    if v.grad is not None
                )
                finite &= np.isfinite(grad) and bool(torch.isfinite(loss))
                max_grad = max(grad, max_grad)
                opt.step()
                if step % 100 == 0:
                    print(name, seed, step, float(loss.detach()), flush=True)
                    history.append(
                        dict(
                            step=step,
                            loss=float(loss.detach()),
                            tasks={k: float(v.detach()) for k, v in parts.items()},
                        )
                    )
            torch.cuda.synchronize()
            ck = root / f"m5a_{name}_{seed}.pt"
            records.append(
                dict(
                    name=name,
                    seed=seed,
                    steps=cfg["steps"],
                    lr=lr,
                    selected=chosen[name, seed],
                    history=history,
                    parameters=sum(p.numel() for p in head.parameters()),
                    finite_gradients=bool(finite),
                    max_gradient=max_grad,
                    elapsed_s=time.perf_counter() - start,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    checkpoint=str(ck),
                    sha256=file_hash(ck),
                )
            )
    need_c = any(
        chosen["B_frozen", s]["per"][k] - chosen["A_prior", s]["per"][k] < 0.02
        for s in cfg["seeds"]
        for k in ["surface", "obstacle"]
    )
    atomic_json(
        root / "m5a_ab_decision.json",
        dict(need_context=need_c, records=records, test_used=False),
    )
    if need_c:
        for seed in cfg["seeds"]:
            net = encoder_load()
            torch.manual_seed(seed)
            head = EvidenceHead().cuda()
            head.load_state_dict(
                torch.load(root / f"m5a_B_frozen_{seed}.pt", weights_only=True)["head"]
            )
            for v in net.context_head.parameters():
                v.requires_grad_(True)
            parameters = list(head.parameters()) + list(net.context_head.parameters())
            opt = torch.optim.Adam(parameters, lr=0.0003)
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            best = -1
            history = []
            max_grad = 0
            finite = True
            schedule = np.load(root / f"m5a_schedule_{seed}.npy")[:200]
            for step in range(201):
                if step % 50 == 0:
                    vf = encode(net, items["val"])
                    score, per = selection(
                        head, vf, labels["val"], datasets["val"].rows
                    )
                    history.append(dict(step=step, selection=score, val=per))
                    del vf
                    if score > best:
                        best = score
                        chosen["C_context", seed] = dict(
                            score=score, per=per, step=step
                        )
                        torch.save(
                            dict(
                                head=head.state_dict(),
                                context=net.context_head.state_dict(),
                                seed=seed,
                                step=step,
                                model="C_context",
                            ),
                            root / f"m5a_C_context_{seed}.pt",
                        )
                if step == 200:
                    break
                ix = schedule[step]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = head(
                        net(**stack([items["train"][i] for i in ix], "inputs"))[
                            "bev_current"
                        ]
                    )
                loss, parts = evidence_loss(
                    out, {k: v[ix] for k, v in labels["train"].items()}
                )
                loss.backward()
                grad = max(
                    float(v.grad.abs().max()) for v in parameters if v.grad is not None
                )
                finite &= np.isfinite(grad) and bool(torch.isfinite(loss))
                max_grad = max(max_grad, grad)
                opt.step()
                if step % 50 == 0:
                    print("C_context", seed, step, float(loss.detach()), flush=True)
                    history.append(
                        dict(
                            step=step,
                            loss=float(loss.detach()),
                            tasks={k: float(v.detach()) for k, v in parts.items()},
                        )
                    )
            assert all(
                p.grad is None
                for n, p in net.named_parameters()
                if not n.startswith("context_head.")
            )
            torch.cuda.synchronize()
            ck = root / f"m5a_C_context_{seed}.pt"
            records.append(
                dict(
                    name="C_context",
                    seed=seed,
                    steps=200,
                    lr=0.0003,
                    selected=chosen["C_context", seed],
                    history=history,
                    parameters=sum(v.numel() for v in parameters),
                    finite_gradients=bool(finite),
                    max_gradient=max_grad,
                    elapsed_s=time.perf_counter() - start,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    checkpoint=str(ck),
                    sha256=file_hash(ck),
                    context_only=True,
                )
            )
    atomic_json(
        a.output,
        dict(
            kind="actual_RTX4090_grouped_small_research",
            records=records,
            conditional_c_executed=need_c,
            sources=sources,
            encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
            protocol_sha256=file_hash("configs/m5a_protocol.json"),
            index_manifest=json.loads(Path(a.index, "manifest.json").read_text()),
            geometry_dtypes=sorted(set(tuple(t) for t in dtypes)),
            test_used_for_selection=False,
            limits="C starts selected B and adds 200 steps; feature fine-tuning evidence, not equal-budget single-variable ablation.",
        ),
    )


if __name__ == "__main__":
    main()
