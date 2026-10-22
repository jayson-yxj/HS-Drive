"""Fixed-budget nested location scaling; same RGB candidates/order for Direct/Prior."""

import json, time, hashlib
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from models.m5n_direct import DirectCenterHead, reconstruct
from tools.m5o_common import write, asset, provenance
from datasets.m5o_benchmark import support, unique_target_key, summary
from training.runtime import prepare_cuda


def read(split):
    p = Path(f"runs/m5o_cache_{split}")
    return (
        torch.load(p / "features.pt", weights_only=True)["rgb_features"],
        json.loads((p / "candidates.json").read_text()),
        json.loads((p / "supervision.json").read_text()),
    )


def pack(data, seed, groups=None):
    X, cc, sup = data
    labels = []
    seen = set()
    for l in sup["labels"]:
        c = cc[l["feature_index"]]
        if (
            c["seed"] != seed
            or not l["train_eligible"]
            or (groups is not None and c["group"] not in groups)
        ):
            continue
        key = unique_target_key(l["reference"])
        if key in seen:
            continue
        seen.add(key)
        labels.append(l)
    ids = [l["feature_index"] for l in labels]
    cs = [cc[i] for i in ids]
    return dict(
        x=X[ids].cuda(),
        target=torch.tensor([l["target"] for l in labels], device="cuda"),
        regions=torch.tensor([c["region"] for c in cs], device="cuda"),
        K=torch.tensor([c["K"] for c in cs], device="cuda"),
        T=torch.tensor([c["T"] for c in cs], device="cuda"),
        xyz=torch.tensor(
            [l["reference"]["box_center_ego"] for l in labels], device="cuda"
        ),
        groups=[c["group"] for c in cs],
        ids=[c["candidate_id"] for c in cs],
    )


def main():
    protocol = json.loads(Path("configs/m5o_protocol.json").read_text())
    data = {s: read(s) for s in ["train", "val", "calibration"]}
    checks = []
    sets = {s: {c["group"] for c in d[1]} for s, d in data.items()}
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise ValueError("cross-split group leakage")
    for split, (X, cc, sup) in data.items():
        for seed in protocol["seeds"]:
            s = support(cc, sup["labels"], seed)
            checks.append(
                dict(
                    split=split,
                    seed=seed,
                    **s,
                    passed=s["unique_candidates"]
                    >= protocol["support"]["unique_candidates"][split]
                    and s["groups"] >= protocol["support"]["groups"][split],
                )
            )
    write(
        "reports/m5o_pretraining_support.json",
        dict(
            records=checks,
            passed=all(r["passed"] for r in checks),
            provenance=provenance(),
        ),
    )
    if not all(r["passed"] for r in checks):
        write(
            "reports/m5o_training.json",
            dict(status="NotRun_support_gate", checks=checks),
        )
        return
    prepare_cuda()
    torch.set_num_threads(2)
    out = Path("runs/m5o_training")
    out.mkdir(exist_ok=False)
    results = []
    cfg = protocol["training"]
    for seed in protocol["seeds"]:
        allgroups = sorted(
            {c["group"] for c in data["train"][1]},
            key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
        )
        val = pack(data["val"], seed)
        for fraction in cfg["fractions"]:
            groups = allgroups[: int(len(allgroups) * fraction)]
            tr = pack(data["train"], seed, groups)
            rng = np.random.default_rng(seed)
            bygroup = {
                g: np.flatnonzero(np.array(tr["groups"]) == g)
                for g in sorted(set(tr["groups"]))
            }
            order = [
                [
                    int(rng.choice(bygroup[str(rng.choice(list(bygroup)))]))
                    for _ in range(cfg["batch"])
                ]
                for step in range(cfg["steps"])
            ]
            tag = f"{seed}_f{int(fraction*100)}"
            write(
                out / f"order_{tag}.json",
                dict(groups=groups, candidate_ids=tr["ids"], order=order),
            )
            for arm in cfg["arms"]:
                torch.manual_seed(seed)
                dim = 600 if arm == "direct" else 24
                model = DirectCenterHead(dim).cuda()
                x = tr["x"] if dim == 600 else tr["x"][:, -24:]
                vx = val["x"] if dim == 600 else val["x"][:, -24:]
                model.mean.copy_(x.mean(0))
                model.scale.copy_(x.std(0).clamp_min(0.05))
                opt = torch.optim.AdamW(
                    model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
                )
                history = []
                best = None
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for step in range(cfg["steps"] + 1):
                    if step in cfg["eval_steps"]:
                        with torch.no_grad():
                            pred = model(vx)
                            xyz = reconstruct(pred, val["regions"], val["K"], val["T"])
                            errors = (
                                (xyz[:, :2] - val["xyz"][:, :2])
                                .norm(dim=1)
                                .cpu()
                                .numpy()
                            )
                            macro = float(
                                np.mean(
                                    [
                                        errors[np.array(val["groups"]) == g].mean()
                                        for g in sorted(set(val["groups"]))
                                    ]
                                )
                            )
                            path = out / f"m5o_{arm}_{tag}_step{step}.pt"
                            torch.save(
                                dict(
                                    model=model.state_dict(),
                                    input_dim=dim,
                                    seed=seed,
                                    arm=arm,
                                    fraction=fraction,
                                    step=step,
                                ),
                                path,
                            )
                            r = dict(
                                step=step,
                                val_macro_xy=macro,
                                val_xy=summary(errors),
                                **asset(path),
                            )
                            history.append(r)
                            if best is None or macro < best["val_macro_xy"]:
                                best = r
                        print(tag, arm, step, macro, flush=True)
                    if step == cfg["steps"]:
                        break
                    ids = order[step]
                    opt.zero_grad(set_to_none=True)
                    pred = model(x[ids])
                    target = tr["target"][ids]
                    depth = F.smooth_l1_loss(pred[:, :1], target[:, :1])
                    offset = F.smooth_l1_loss(pred[:, 1:], target[:, 1:])
                    loss = depth + offset
                    if not torch.isfinite(loss):
                        raise FloatingPointError("loss")
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if not torch.isfinite(grad):
                        raise FloatingPointError("gradient")
                    opt.step()
                    if step % 50 == 0:
                        history.append(
                            dict(
                                step=step + 1,
                                loss=float(loss.detach()),
                                depth_loss=float(depth.detach()),
                                offset_loss=float(offset.detach()),
                                gradient_norm=float(grad),
                            )
                        )
                torch.cuda.synchronize()
                results.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        fraction=fraction,
                        groups=groups,
                        unique_candidates=len(tr["ids"]),
                        selected=best,
                        history=history,
                        parameters=sum(p.numel() for p in model.parameters()),
                        elapsed_s=time.perf_counter() - start,
                        peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                        order=asset(out / f"order_{tag}.json"),
                    )
                )
    write(
        "reports/m5o_training.json",
        dict(
            status="Completed",
            kind="actual_RTX4090_fixed_M5N_heads",
            records=results,
            provenance=provenance(),
            protocol=asset("configs/m5o_protocol.json"),
        ),
    )


if __name__ == "__main__":
    main()
