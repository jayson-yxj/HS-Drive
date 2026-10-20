"""Post-hoc plots/local diagnostics only. Never select or train a checkpoint."""

import argparse
import json
from pathlib import Path
import numpy as np, torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import stack, task_arrays
from diagnostics.m5a_metrics import binary_metrics
from simulator.recorder import atomic_json, file_hash


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="reports/m5a_visual_generalization.json")
    p.add_argument("--output", default="reports/m5a_figures")
    p.add_argument("--diagnostics", default="reports/m5a_local_change_diagnostics.json")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    report = json.loads(Path(args.report).read_text())
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5a_index", "test"
    )
    items = [ds[i] for i in range(len(ds))]
    labels = stack(items, "targets", "cpu")
    records = []
    for r in report["records"]:
        with np.load(r["predictions"]) as z:
            pred = {k: torch.from_numpy(z[k]) for k in z.files}
        arr = {
            k: tuple(x.numpy() for x in val)
            for k, val in task_arrays(pred, labels).items()
        }
        pairs = []
        for group in sorted(set(x["group"] for x in ds.rows)):
            zero = next(
                i
                for i, x in enumerate(ds.rows)
                if x["group"] == group and x["configuration"] == 0
            )
            for i, row in enumerate(ds.rows):
                if row["group"] != group or row["configuration"] == 0:
                    continue
                tasks = {}
                for key in ["surface", "obstacle"]:
                    logit, t, m = arr[key]
                    p = sig(logit)
                    changed = (t[i] != t[zero]) & m[i] & m[zero]
                    d = p[i] - p[zero]
                    direction = t[i].astype(float) - t[zero].astype(float)
                    threshold = r["val_selected_thresholds"][key]
                    tasks[key] = dict(
                        n=int(changed.sum()),
                        correct_rgb=binary_metrics(p[i], t[i], changed, threshold),
                        empty_rgb_wrong_pair=binary_metrics(
                            p[zero], t[i], changed, threshold
                        ),
                        signed_delta_mean=(
                            float((d[changed] * direction[changed]).mean())
                            if changed.any()
                            else None
                        ),
                        direction_over_005=(
                            float((d[changed] * direction[changed] > 0.05).mean())
                            if changed.any()
                            else None
                        ),
                    )
                pairs.append(
                    dict(group=group, configuration=row["configuration"], tasks=tasks)
                )
            # Predetermined config1, all three groups; zero cherry-picking.
            i = next(
                i
                for i, x in enumerate(ds.rows)
                if x["group"] == group and x["configuration"] == 1
            )
            fig, axes = plt.subplots(2, 4, figsize=(14, 8))
            extent = (-30, 30, 60, -20)
            for rr, key in enumerate(["surface", "obstacle"]):
                l, t, m = arr[key]
                p = sig(l)
                th = r["val_selected_thresholds"][key]
                truth = t[i].any(0)
                error = ((p[i] > th) & ~t[i] & m[i]).any(0).astype(float) - (
                    (p[i] <= th) & t[i] & m[i]
                ).any(0).astype(float)
                maps = [
                    np.where(m[i].any(0), truth, np.nan),
                    p[i].max(0),
                    np.max(abs(p[i] - p[zero]), axis=0),
                    error,
                ]
                titles = [
                    "GT events (gray unknown)",
                    "prediction max layer",
                    "RGB layout change |delta|",
                    "FP red / FN blue on mask",
                ]
                for ax, im, title in zip(axes[rr], maps, titles):
                    cm = "coolwarm" if title.startswith("FP") else "viridis"
                    ax.set_facecolor(".75")
                    ax.imshow(
                        im,
                        extent=extent,
                        cmap=cm,
                        vmin=-1 if cm == "coolwarm" else 0,
                        vmax=1,
                    )
                    ax.set_title(key + " " + title, fontsize=9)
                    ax.set_xlabel("ego y left m")
                    ax.set_ylabel("ego x forward m")
            fig.suptitle(
                f"{r['name']} seed {r['seed']} {group} config1 vs empty; no volume-free claim"
            )
            fig.tight_layout()
            fig.savefig(root / f"m5a_{r['name']}_{r['seed']}_{group}.png", dpi=130)
            plt.close(fig)
        records.append(dict(name=r["name"], seed=r["seed"], pairs=pairs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["surface", "obstacle"]):
        names = [f"{r['name']} {r['seed']}" for r in report["records"]]
        val = [r["test"][key]["scene_iou_mean"] for r in report["records"]]
        err = [r["test"][key]["scene_iou_std"] for r in report["records"]]
        ax.bar(names, val, yerr=err)
        ax.set_title(key + " scene mean IoU (std, n=3)")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(root / "m5a_generalization.png", dpi=150)
    plt.close(fig)
    # Six RGB views at fixed first test group/config1 (inputs, no privileged overlays).
    i = next(i for i, r in enumerate(ds.rows) if r["configuration"] == 1)
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))
    for a, im, name in zip(
        ax.flat, items[i].inputs["images"], ds.rows[i]["camera_names"]
    ):
        a.imshow(im.permute(1, 2, 0))
        a.set_title(name)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(root / "m5a_six_camera_example.png", dpi=130)
    plt.close(fig)
    atomic_json(
        args.diagnostics,
        dict(
            kind="offline_same_pose_empty_vs_obstacle_rgb_counterfactual_on_common_changed_event_masks",
            records=records,
            threshold_source="validation selected",
            selection=False,
            source_sha256=file_hash(__file__),
            limits="Changed-mask evaluation is diagnostic with GT masks; sparse event changes can reflect occlusion/rasterization, not complete mesh obstacle changes.",
        ),
    )


if __name__ == "__main__":
    main()
