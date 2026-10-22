"""Finite matched two-seed DirectCenter and geometry-only prior training."""

import json, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from tools.m5n_common import write, asset, stats
from models.m5n_direct import DirectCenterHead, reconstruct
from training.runtime import prepare_cuda
from simulator.recorder import file_hash


def read(split):
    root = Path(f"runs/m5n_cache_{split}")
    X = torch.load(root / "features.pt", weights_only=True)["rgb_features"]
    cand = json.loads((root / "candidates.json").read_text())
    sup = json.loads((root / "supervision.json").read_text())
    return X, cand, sup


def pack(X, candidates, labels, seed):
    ll = [
        r
        for r in labels
        if r["train_eligible"] and candidates[r["feature_index"]]["seed"] == seed
    ]
    ii = [r["feature_index"] for r in ll]
    cc = [candidates[i] for i in ii]
    return dict(
        x=X[ii].cuda(),
        target=torch.tensor([r["target"] for r in ll], device="cuda"),
        regions=torch.tensor([r["region"] for r in cc], device="cuda"),
        K=torch.tensor([r["K"] for r in cc], device="cuda"),
        T=torch.tensor([r["T"] for r in cc], device="cuda"),
        xyz=torch.tensor([r["reference"]["box_center_ego"] for r in ll], device="cuda"),
        groups=[r["group"] for r in cc],
        types=[r["reference"]["type"] for r in ll],
        ranges=[r["reference"]["ego_xy_range_m"] for r in ll],
        indices=ii,
    )


def main():
    prepare_cuda()
    torch.set_num_threads(2)
    protocol = json.loads(Path("configs/m5n_protocol.json").read_text())
    steps = protocol["training"]["steps"]
    tr, tc, ts = read("train")
    va, vc, vs = read("val")
    train_groups = {r["group"] for r in tc}
    val_groups = {r["group"] for r in vc}
    assert not train_groups & val_groups
    support = []
    for seed in [101, 102]:
        for split, X, cc, ss in [("train", tr, tc, ts), ("val", va, vc, vs)]:
            data = pack(X, cc, ss["labels"], seed)
            n = len(data["indices"])
            groups = len(set(data["groups"]))
            bands = len({0 if d < 10 else 1 if d < 20 else 2 for d in data["ranges"]})
            passed = (
                n >= (40 if split == "train" else 30) and groups >= 4 and bands >= 2
            )
            support.append(
                dict(
                    seed=seed,
                    split=split,
                    n=n,
                    positions=groups,
                    distance_bands=bands,
                    by_class={
                        t: data["types"].count(t) for t in ["vehicle", "person", "cone"]
                    },
                    passed=passed,
                )
            )
    write(
        "reports/m5n_training_support.json",
        dict(
            records=support,
            passed=all(r["passed"] for r in support),
            limits="Actor-views correlated; independent units are position groups. Class claims withheld under20 unique views. Historical actor IDs repeat across groups/world reload; group is split identity.",
        ),
    )
    if not all(r["passed"] for r in support):
        write(
            "reports/m5n_direct_range_training.json",
            dict(status="NotRun", reason="support gate failed"),
        )
        return
    out = Path("runs/m5n_training")
    out.mkdir(exist_ok=False)
    results = []
    for seed in [101, 102]:
        train = pack(tr, tc, ts["labels"], seed)
        val = pack(va, vc, vs["labels"], seed)
        rng = np.random.default_rng(seed)
        groups = sorted(set(train["groups"]))
        bygroup = {
            g: np.array([i for i, v in enumerate(train["groups"]) if v == g])
            for g in groups
        }
        order = []
        for step in range(steps):
            order.append(
                [int(rng.choice(bygroup[str(rng.choice(groups))])) for _ in range(32)]
            )
        write(out / f"order_{seed}.json", order)
        for arm in ["direct", "geometry_prior"]:
            torch.manual_seed(seed)
            dim = 600 if arm == "direct" else 24
            model = DirectCenterHead(dim).cuda()
            x = train["x"] if arm == "direct" else train["x"][:, -24:]
            vx = val["x"] if arm == "direct" else val["x"][:, -24:]
            model.mean.copy_(x.mean(0))
            model.scale.copy_(x.std(0).clamp_min(0.05))
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
            history = []
            best = None
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for step in range(steps + 1):
                if step in protocol["training"]["eval_steps"]:
                    with torch.no_grad():
                        pred = model(vx)
                        xyz = reconstruct(pred, val["regions"], val["K"], val["T"])
                        errors = (
                            (xyz[:, :2] - val["xyz"][:, :2]).norm(dim=1).cpu().numpy()
                        )
                        macro = float(
                            np.mean(
                                [
                                    errors[np.array(val["groups"]) == g].mean()
                                    for g in set(val["groups"])
                                ]
                            )
                        )
                        loss = float(
                            F.smooth_l1_loss(pred[:, :1], val["target"][:, :1])
                            + F.smooth_l1_loss(pred[:, 1:], val["target"][:, 1:])
                        )
                        path = out / f"m5n_{arm}_{seed}_step{step}.pt"
                        torch.save(
                            dict(
                                model=model.state_dict(),
                                seed=seed,
                                arm=arm,
                                step=step,
                                input_dim=dim,
                            ),
                            path,
                        )
                        record = dict(
                            step=step,
                            val_macro_xy=macro,
                            val_loss=loss,
                            val_xy=stats(errors.tolist()),
                            **asset(path),
                        )
                        history.append(record)
                        if best is None or macro < best["val_macro_xy"]:
                            best = record
                    print(arm, seed, step, macro, flush=True)
                if step == steps:
                    break
                ii = order[step]
                opt.zero_grad(set_to_none=True)
                pred = model(x[ii])
                target = train["target"][ii]
                depth = F.smooth_l1_loss(pred[:, :1], target[:, :1])
                offset = F.smooth_l1_loss(pred[:, 1:], target[:, 1:])
                loss = depth + offset
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                assert torch.isfinite(grad)
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
            results.append(
                dict(
                    seed=seed,
                    arm=arm,
                    selected=best,
                    history=history,
                    parameters=sum(p.numel() for p in model.parameters()),
                    steps=steps,
                    elapsed_s=time.perf_counter() - start,
                    peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                    order=asset(out / f"order_{seed}.json"),
                )
            )
    write(
        "reports/m5n_direct_range_training.json",
        dict(
            kind="actual_RTX4090_frozen_Mobile_600step_2seed_small_heads",
            status="Completed",
            records=results,
            source_sha256=file_hash(__file__),
            limits="Head only; original Depth/Mobile frozen. Different fixed Mobile checkpoints perseed; same-seed direct/prior matched exact samples and steps. GeometryPrior different input dimension, not equal parameter count. No NLL/no uncertainty training/no budget extension. Validation reused for checkpoint+calibration, not independent guarantee.",
        ),
    )


if __name__ == "__main__":
    main()
