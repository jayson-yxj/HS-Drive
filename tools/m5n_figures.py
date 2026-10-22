"""Standalone training/generalization/risk/instance visualizations."""

import json
from pathlib import Path
import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    dest = Path("reports/m5n_figures")
    dest.mkdir(exist_ok=True)
    train = json.loads(Path("reports/m5n_direct_range_training.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for r in train["records"]:
        h = [v for v in r["history"] if "loss" in v]
        axs[0].semilogy(
            [v["step"] for v in h],
            [v["loss"] for v in h],
            label=f"{r['arm']}{r['seed']}",
        )
        h = [v for v in r["history"] if "val_macro_xy" in v]
        axs[1].plot(
            [v["step"] for v in h],
            [v["val_macro_xy"] for v in h],
            "-o",
            label=f"{r['arm']}{r['seed']}",
        )
    axs[0].set_ylabel("Train Huber loss")
    axs[1].set_ylabel("Validation position-macro XY MAE (m)")
    for ax in axs:
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(dest / "m5n_training.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for split in ["val", "regression", "sealed"]:
        path = Path(f"reports/m5n_eval_{split}.json")
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        for seed in [101, 102]:
            curve = [
                v
                for v in r["curves"]
                if v["seed"] == seed and v["metrics"]["localized"]
            ]
            x = [v["metrics"]["coverage"] for v in curve]
            axs[0].plot(
                x,
                [v["metrics"]["xy"]["mean"] for v in curve],
                "-o",
                label=f"{split}{seed}",
            )
            axs[1].plot(x, [v["metrics"]["xy"]["p95"] for v in curve], "-o")
            axs[2].plot(x, [v["metrics"]["severe5m_rate"] for v in curve], "-o")
    for ax, label in zip(axs, ["XY mean (m)", "XY P95 (m)", "Severe >5m fraction"]):
        ax.set_xlabel("All visible actor-view coverage")
        ax.set_ylabel(label)
        ax.axvline(0.3, color="gray", ls="--")
        ax.grid(alpha=0.2)
    axs[1].axhline(5, color="red", ls="--")
    axs[2].axhline(0.05, color="red", ls="--")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest / "m5n_risk_coverage.png", dpi=160)
    plt.close(fig)
    r = json.loads(Path("reports/m5n_eval_val.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    methods = ["plane", "dense_expected", "dense_mode", "direct", "geometry_prior"]
    for ax, seed in zip(axs, [101, 102]):
        rr = [
            next(v for v in r["aggregate"] if v["seed"] == seed and v["method"] == m)
            for m in methods
        ]
        x = np.arange(5)
        ax.bar(x - 0.18, [v["xy"]["mean"] for v in rr], 0.35, label="MAE")
        ax.bar(x + 0.18, [v["xy"]["p95"] for v in rr], 0.35, label="P95")
        ax.set_xticks(x, methods, rotation=20)
        ax.set_title(f"Validation seed{seed}; different method coverage")
        ax.set_ylabel("XY error m")
        ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "m5n_localization.png", dpi=160)
    plt.close(fig)
    # Same actor-center reference, three-dimensional origin/visible-bottom distinction.
    root = Path("runs/m5n_cache_val")
    s = json.loads((root / "supervision.json").read_text())
    rr = [a for a in s["actors"] if a["seed"] == 101]
    fig = plt.figure(figsize=(12, 4))
    for i, typ in enumerate(["vehicle", "person", "cone"]):
        r = next(a for a in rr if a["type"] == typ)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for key, label in [
            ("box_center_ego", "simulator BoxCenter"),
            ("actor_origin_ego", "Actor origin"),
            ("visible_surface_center_ego", "visible surface mean"),
            ("visible_bottom_ego", "visible bottom NOT contact"),
        ]:
            p = r[key]
            ax.scatter(*p, label=label, s=35)
        ax.set_title(typ)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(dest / "m5n_reference_semantics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
