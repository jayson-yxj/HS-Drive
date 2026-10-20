"""Research figures from actual reports and predictions; no generated imagery."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from simulator.recorder import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path("reports/m5c_figures")
    root.mkdir(exist_ok=False)
    plots = []
    geometry = read("reports/m5c_corrected_geometry_resolution.json")
    bands = ["0_10", "10_15", "15_20", "20_35", "35_80"]
    modes = ["nearest_lid", "learned_distribution", "expected_depth"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for mode in modes:
        r = [
            next(
                a
                for a in geometry["summary"]
                if a["mode"] == mode and a["distance"] == d
            )
            for d in bands
        ]
        for ax, key in zip(
            axes, ["correct_voxel", "xy_error_m", "same_height_within1m_mass"]
        ):
            ax.plot(
                bands,
                [a[key] if a[key] is not None else np.nan for a in r],
                marker="o",
                label=mode,
            )
            ax.set_title(key)
            ax.set_xlabel("surface radial distance(m)")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = root / "m5c_geometry.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))
    sampling = read("reports/m5c_corrected_feature_sampling.json")["records"]
    fig, ax = plt.subplots(figsize=(9, 4))
    configs = [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14]
    for offset, stride in [(-0.18, "16"), (0.18, "8")]:
        values = [
            sum(
                r["stride"][stride]["reliable"]
                for r in sampling
                if r["configuration"] == c
            )
            for c in configs
        ]
        ax.bar(np.arange(len(configs)) + offset, values, 0.36, label=f"stride{stride}")
    ax.set_xticks(
        np.arange(len(configs)),
        [
            "car5",
            "car10",
            "car15",
            "car20",
            "car25",
            "car30",
            "truck20",
            "ped15",
            "ped25",
            "cone10",
            "raised20",
        ],
    )
    ax.set_ylabel("reliable target rays (4 val groups)")
    ax.legend()
    fig.tight_layout()
    path = root / "m5c_sampling.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))
    report = read("reports/m5c_absolute_obstacle_detection.json")
    fig, ax = plt.subplots(figsize=(10, 4))
    names = ["P_prior", "A_frozen", "B_context", "C_pair", "D_direct"]
    for shift, seed in [(-0.18, 71), (0.18, 72)]:
        ax.bar(
            np.arange(5) + shift,
            [
                next(
                    r["summary"]["mean_scene_iou"]
                    for r in report["records"]
                    if r["name"] == n and r["seed"] == seed
                )
                for n in names
            ],
            0.36,
            label=f"seed{seed}",
        )
    ax.set_xticks(np.arange(5), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean scene singleframe known-surface IoU")
    ax.legend()
    fig.tight_layout()
    path = root / "m5c_ablation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))
    index = Path("/root/gpufree-data/hs-drive-data/processed/m5c_test_index")
    rows = read(index / "index.json")["samples"]
    surf = {
        r["sample_id"]: r for r in read(index / "m5c_surface_index.json")["records"]
    }
    selected = [
        r
        for r in report["records"]
        if r["name"] in ["P_prior", "C_pair", "D_direct"] and r["seed"] == 71
    ]
    scores = {r["name"]: np.load(r["predictions"])["score"] for r in selected}
    # Fixed first test group, fixed near/medium/far/ped configurations, not best-case selection.
    for configuration in [1, 3, 5, 6, 11]:
        i = next(
            i
            for i, r in enumerate(rows)
            if r["group"] == "Town01_c3" and r["configuration"] == configuration
        )
        r = rows[i]
        with np.load(surf[r["sample_id"]]["path"]) as z:
            gt = z["target"]
            valid = z["valid"]
            actor = z["actor_masks"].any(0)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        inp = read(index / r["input"])
        axes[0, 0].imshow(Image.open(inp["images"][0]["path"]))
        axes[0, 0].set_title("front RGB")
        img = np.zeros((160, 120, 3))
        img[valid.any(0)] = [0.3, 0.3, 0.3]
        img[(gt & valid).any(0)] = [0.8, 0.2, 0.1]
        img[actor.any(0)] = [1, 1, 0]
        axes[0, 1].imshow(img, origin="lower", extent=[-30, 30, -20, 60])
        axes[0, 1].set_title("known ground gray / obstacle red / actor yellow")
        axes[0, 2].axis("off")
        axes[0, 2].text(
            0.05,
            0.9,
            f'{r["group"]} config {configuration}\nUnknown black is NOT safe free\nXY max over height for display only\nMetrics enforce height layer',
            va="top",
        )
        for ax, rr in zip(axes[1], selected):
            ax.imshow(
                scores[rr["name"]][i].max(0),
                origin="lower",
                extent=[-30, 30, -20, 60],
                vmin=0,
                vmax=1,
                cmap="magma",
            )
            ax.contour(
                (gt & valid).any(0),
                levels=[0.5],
                colors="cyan",
                linewidths=0.5,
                origin="lower",
                extent=[-30, 30, -20, 60],
            )
            ax.set_title(f"{rr['name']} raw score; th={rr['threshold']}")
        fig.tight_layout()
        path = root / f"m5c_singleframe_config{configuration}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(str(path))
    runtime = read("reports/m5c_runtime_stress.json")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for r in runtime["records"]:
        if "pressure" not in r["report"]:
            continue
        logs = [json.loads(v) for v in Path(r["log"]).read_text().splitlines()]
        t = np.array([v["control_apply_wall"] for v in logs])
        axes[0].plot(t[1:] - t[0], np.diff(t) * 1000, linewidth=0.5)
        axes[1].plot(
            t - t[0], [v["profile"]["world_tick_ms"] for v in logs], linewidth=0.5
        )
    axes[0].axhline(50, color="red", ls="--")
    axes[0].set_ylabel("control interval ms")
    axes[1].set_ylabel("world.tick blocking ms")
    axes[1].set_xlabel("wall elapsed seconds")
    fig.tight_layout()
    path = root / "m5c_runtime_pressure.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))
    atomic_json(
        "reports/m5c_visualizations.json",
        dict(
            kind="plots_from_actual_experiments",
            figures={f: file_hash(f) for f in plots},
            source_sha256=file_hash(__file__),
            selection="fixed first test group and configs1/3/5/6/11; no bestcase selection",
        ),
    )


if __name__ == "__main__":
    main()
