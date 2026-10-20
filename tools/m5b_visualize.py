"""Predetermined test config plots, failures retained; no model selection."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json, file_hash


def main():
    art = Path("reports/m5b_figures")
    art.mkdir(exist_ok=False)
    r = json.loads(Path("reports/m5b_counterfactual_detection.json").read_text())
    index = Path("/root/gpufree-data/hs-drive-data/processed/m5b_sealed_index")
    rows = [
        x
        for x in json.loads((index / "index.json").read_text())["samples"]
        if x["split"] == "test"
    ]
    pairs = json.loads((index / "m5b_pairs.json").read_text())["pairs"]
    figs = []
    for model in r["records"]:
        with np.load(model["predictions"]) as z:
            p = z["score"]
        for group in sorted(set(x["group"] for x in rows)):
            ia = next(
                i
                for i, x in enumerate(rows)
                if x["group"] == group and x["configuration"] == 0
            )
            ib = next(
                i
                for i, x in enumerate(rows)
                if x["group"] == group and x["configuration"] == 2
            )
            pair = next(
                x
                for x in pairs
                if x["group"] == group
                and x["a_configuration"] == 0
                and x["b_configuration"] == 2
            )
            with np.load(pair["path"]) as z:
                gt = z["appeared"].any(0)
                support = (z["appeared"] | z["vanished"] | z["stable"]).any(0)
            fig, axes = plt.subplots(1, 4, figsize=(14, 5))
            delta = p[ib] - p[ia]
            values = [
                np.where(support, gt, np.nan),
                p[ia].max(0),
                p[ib].max(0),
                np.max(delta, axis=0),
            ]
            titles = [
                "Supported appearance GT",
                "Empty score max Z",
                "Obstacle score max Z",
                "Max Z signed change",
            ]
            for ax, v, title in zip(axes, values, titles):
                ax.set_facecolor(".7")
                ax.imshow(
                    v[::-1, ::-1],
                    extent=[30, -30, -20, 60],
                    vmin=-0.3 if title.startswith("Max") else 0,
                    vmax=0.3 if title.startswith("Max") else 1,
                    cmap="coolwarm" if title.startswith("Max") else "viridis",
                )
                ax.set_title(title)
                ax.set_xlabel("ego y left m")
            axes[0].set_ylabel("ego x forward m")
            fig.suptitle(
                f"{model['name']} seed{model['seed']} {group} shifted11m, no volume-free claim"
            )
            fig.tight_layout()
            path = art / f"m5b_{model['name']}_{model['seed']}_{group}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            figs.append(str(path))
    g = json.loads(Path("reports/m5b_geometry_diagnostics.json").read_text())
    p = Path("reports/m5b_geometry_artifacts/m5b_projection_example.npz")
    with np.load(p) as z:
        fig, ax = plt.subplots(1, 4, figsize=(14, 5))
        for a, k in zip(
            ax, ["learned_mass", "oracle_mass", "learned_score", "oracle_score"]
        ):
            v = z[k]
            v = v[0] if v.ndim == 4 else v
            a.imshow(v.max(0)[::-1, ::-1], cmap="viridis")
            a.set_title(k)
        fig.tight_layout()
        fig.savefig(art / "m5b_oracle_learned_projection.png", dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [f"{x['name']} {x['seed']}" for x in r["records"]]
    ax.bar(names, [x["summary"]["mean_scene_iou"] for x in r["records"]])
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("scene mean signed-change IoU")
    ax.set_title("3 sealed pose groups; validation-only thresholds")
    fig.tight_layout()
    fig.savefig(art / "m5b_change_comparison.png", dpi=140)
    plt.close(fig)
    atomic_json(
        "reports/m5b_visualizations.json",
        dict(
            kind="offline_diagnostic_plots",
            paths=figs,
            selection="Every model/seed/group config2, predetermined; no cherry-picked successes.",
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
