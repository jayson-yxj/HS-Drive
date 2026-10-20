"""Research figures from frozen predictions; no threshold or model selection."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


def read(name):
    return json.loads(Path("reports", name + ".json").read_text())


def main():
    dest = Path("reports/m5d_figures")
    dest.mkdir(exist_ok=False)
    report = read("m5d_presence_absence")
    recs = report["records"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for r in recs:
        if r["seed"] not in [81, 71]:
            continue
        curves = r["threshold_curve"]
        rec = []
        fp = []
        prec = []
        for c in curves:
            vals = list(c["by_scene"].values())
            n = sum(v["n"] for v in vals)
            tp = sum(v["presence_hits"] for v in vals)
            f = sum(v["absence_false_positives"] for v in vals)
            rec.append(tp / max(n, 1))
            fp.append(f / max(n, 1))
            prec.append(tp / max(tp + f, 1))
        ax[0].plot(fp, rec, label=r["name"])
        ax[1].plot(rec, prec, label=r["name"])
    ax[0].set(
        xlabel="Certified absence FPR",
        ylabel="Presence recall",
        title="Frozen-test tradeoff (not threshold reselection)",
    )
    ax[1].set(
        xlabel="Presence recall",
        ylabel="Conditional precision",
        title="Balanced positive/negative point-witness PR",
    )
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.03)
    fig.tight_layout()
    fig.savefig(dest / "m5d_presence_absence_tradeoff.png", dpi=150)
    plt.close(fig)
    index = Path("/root/gpufree-data/hs-drive-data/processed/m5d_test_index")
    rows = json.loads((index / "index.json").read_text())["samples"]
    lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
    predictions = {
        r["name"]: np.load(r["predictions"])["score"]
        for r in recs
        if r["seed"] in [81, 71]
    }
    groups = sorted({r["group"] for r in rows})
    names = ["Historical_D_direct", "A_matched", "B_existence", "C_stable", "S8_stable"]
    for group in groups:
        selected = [
            next(
                (i, r)
                for i, r in enumerate(rows)
                if r["group"] == group and r["configuration"] == c
            )
            for c in [0, 1, 3, 6, 15]
        ]
        fig, axes = plt.subplots(len(names) + 1, 5, figsize=(15, 15))
        for col, (i, row) in enumerate(selected):
            raw = Path(row["raw_frame"]).parent.parent
            frame = json.loads(Path(row["raw_frame"]).read_text())
            rgb = cv2.cvtColor(
                cv2.imread(str(raw / frame["cameras"]["front"]["rgb"])),
                cv2.COLOR_BGR2RGB,
            )
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f"cfg {row['configuration']} front RGB")
            axes[0, col].axis("off")
            for rr, name in enumerate(names, 1):
                im = axes[rr, col].imshow(
                    predictions[name][i, 2],
                    origin="lower",
                    vmin=0,
                    vmax=1,
                    cmap="magma",
                    extent=(-30, 30, -20, 60),
                )
                axes[rr, col].set(
                    xlim=(-8, 8),
                    ylim=(-3, 38),
                    xlabel="Y left (m)",
                    ylabel=f"{name}\nX forward (m)",
                )
        fig.suptitle(
            f"{group}: fixed height [1,2.5)m, no GT mask applied; cfg0/15 empty",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(dest / f"m5d_{group}_presence_maps.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    geometry = read("m5d_geometry_resolution")["summary"]
    bands = ["10_15", "15_20", "20_35"]
    for col, stride in enumerate([16, 8]):
        for mode in ["nearest_lid", "learned_probability", "expected_depth"]:
            rr = [
                next(
                    r
                    for r in geometry
                    if r["stride"] == stride
                    and r["mode"] == mode
                    and r["distance"] == b
                )
                for b in bands
            ]
            axes[0, col].plot(
                bands, [r["xy_error_m"] for r in rr], marker="o", label=mode
            )
            axes[1, col].plot(
                bands, [r["same_layer_1m_mass"] for r in rr], marker="o", label=mode
            )
        axes[0, col].set(
            title=f"Stride {stride}: matched actual feature rays",
            ylabel="Expected XY absolute error (m)",
        )
        axes[1, col].set(
            ylabel="Same-layer <=1m probability mass", xlabel="Distance band (m)"
        )
        for a in axes[:, col]:
            a.legend(fontsize=8)
            a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(dest / "m5d_geometry_errors.png", dpi=150)
    plt.close(fig)
    study = read("m5d_model_ablation")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in study["records"]:
        if r["seed"] != 81:
            continue
        h = [h for h in r["history"] if "total" in h]
        v = [h for h in r["history"] if "val_presence_absence_harmonic" in h]
        axes[0].plot([x["step"] for x in h], [x["total"] for x in h], label=r["name"])
        axes[1].plot(
            [x["step"] for x in v],
            [x["val_presence_absence_harmonic"] for x in v],
            label=r["name"],
        )
    axes[0].set(title="Training total loss (different objectives)", xlabel="step")
    axes[1].set(title="Validation grouped presence/specificity harmonic", xlabel="step")
    for a in axes:
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(dest / "m5d_training.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
