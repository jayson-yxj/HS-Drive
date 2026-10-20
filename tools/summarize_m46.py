"""Render M4.6 measured labels, temporal masks and closed-loop comparisons."""

import argparse, json, shutil
from pathlib import Path
import cv2, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from simulator.recorder import atomic_json, file_hash


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", default="reports/m46_figures")
    p.add_argument("--driving", default="reports/m46_closedloop_final.json")
    p.add_argument("--output", default="reports/m46_comparison_metrics.json")
    a = p.parse_args()
    root = Path(a.figures)
    if root.exists() or Path(a.output).exists():
        raise FileExistsError("New paths required")
    root.mkdir()
    raw = read("reports/m46_occupancy_validation.json")
    dense = read("reports/m46_occupancy_dense.json")
    drive = read(a.driving)
    metrics = {}
    means = {}
    for sid in sorted(set(r["scene"] for r in dense["records"])):
        rows = [r for r in dense["records"] if r["scene"] == sid]
        baseline = [r for r in raw["records"] if r["scene"] == sid]
        sample = rows[0]
        means[sid] = dict(
            frames=len(rows),
            old_ground=float(np.mean([r["old"]["ground"] for r in baseline])),
            visibility_ground=float(np.mean([r["new"]["ground"] for r in baseline])),
            dense_counts={
                k: float(np.mean([r["counts"][k] for r in rows]))
                for k in sample["counts"]
            },
            recovered_rays_mean=float(
                np.mean(
                    [
                        sum(c["recovered_box_rays"] for c in r["cameras"].values())
                        for r in baseline
                    ]
                )
            ),
            heldout_false_free=sum(
                r["audit"]["heldout_surface_marked_free"] for r in rows
            ),
            heldout_surfaces=sum(r["audit"]["heldout_surface_voxels"] for r in rows),
        )
        with np.load(sample["path"]) as arrays:
            labels = {k: arrays[k] for k in arrays.files}
        shutil.copy2(sample["path"], root / (sid + "_labels.npz"))
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))
        for z in range(4):
            axes[0, z].imshow(
                labels["occupancy_state"][z][::-1, ::-1],
                cmap=ListedColormap(["#333333", "#70c1b3", "#e76f51"]),
                vmin=0,
                vmax=2,
            )
            axes[0, z].set_title(f"z{z}: Unknown / Free / Surface")
            axes[1, z].imshow(
                labels["surface_semantic"][z][::-1, ::-1],
                cmap=ListedColormap(
                    ["#333333", "#d6cb67", "#6290c8", "#d9598c", "#aaa"]
                ),
                vmin=0,
                vmax=4,
            )
            axes[1, z].set_title("Ground / Static / Dynamic")
        for ax in axes.flat:
            ax.set_xlabel("Y: left to right (decreasing index)")
            ax.set_ylabel("X front to rear")
        fig.suptitle(sid + " | observed surface, never solid semantic occupancy")
        fig.tight_layout()
        fig.savefig(root / (sid + "_occupancy.png"))
        plt.close(fig)
    metrics["occupancy"] = means
    sid = list(means)
    x = np.arange(len(sid))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.25, [means[s]["old_ground"] for s in sid], width=0.25, label="M45 box")
    ax.bar(
        x,
        [means[s]["visibility_ground"] for s in sid],
        width=0.25,
        label="Visibility correction only",
    )
    ax.bar(
        x + 0.25,
        [means[s]["dense_counts"]["ground"] for s in sid],
        width=0.25,
        label="Additional dense surfaces",
    )
    ax.set_xticks(x, sid, rotation=35, ha="right")
    ax.set_ylabel("Ground voxels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "ground_coverage.png")
    plt.close(fig)
    driving = {}
    for name in sorted(set(r["scenario"] for r in drive["results"])):
        rows = {r["version"]: r for r in drive["results"] if r["scenario"] == name}
        if set(rows) != {"m45", "m46"}:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for version, r in rows.items():
            data = [
                json.loads(line) for line in Path(r["log"]).read_text().splitlines()
            ]
            t = np.array([x["t"] for x in data])
            v = np.array([x["v"] for x in data])
            acc = np.diff(v) / 0.05
            jerk = np.diff(acc) / 0.05
            axes[0, 0].plot(t, v, label=version)
            axes[0, 1].plot(
                t,
                [np.nan if x["gap"] is None else x["gap"] for x in data],
                label=version,
            )
            axes[0, 2].plot(t[:-1], acc, label=version)
            axes[1, 0].plot(t[:-2], jerk, label=version)
            axes[1, 1].plot(t, [x["command"]["brake"] for x in data], label=version)
            axes[1, 2].plot(t, [bool(x["safety_reason"]) for x in data], label=version)
        for ax, title in zip(
            axes.flat,
            [
                "Speed m/s",
                "Bumper gap m",
                "Raw acceleration m/s2",
                "Raw jerk m/s3",
                "Requested brake",
                "Safety forced brake",
            ],
        ):
            ax.set_title(title)
            ax.set_xlabel("Simulation time s")
            ax.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(name + " | same setup, whole-stack comparison")
        fig.tight_layout()
        fig.savefig(root / (name + "_driving.png"))
        plt.close(fig)
        driving[name] = {
            v: {
                k: r[k]
                for k in [
                    "status",
                    "collision_count",
                    "lane_invasions",
                    "final_gap_m",
                    "min_gap_m",
                    "max_decel_raw_mps2",
                    "max_decel_250ms_mps2",
                    "jerk_abs_p95_raw_mps3",
                    "jerk_abs_p95_250ms_mps3",
                    "safety_forced_braking_ticks",
                    "safety_forced_braking_events",
                    "lateral_mean_m",
                    "lateral_max_m",
                    "kinematic_violation_ticks",
                ]
            }
            for v, r in rows.items()
        }
    metrics["driving"] = driving
    # Actual dynamic occupancy at per-height support; distinguish unknown from zero.
    dynamic = [r for r in raw["records"] if r["scene"] == "dynamic_a"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for col, r in enumerate(dynamic):
        with np.load(r["path"]) as d:
            axes[0, col].imshow(
                d["occupancy_state"][1][::-1, ::-1],
                vmin=0,
                vmax=2,
                cmap=ListedColormap(["#333333", "#70c1b3", "#e76f51"]),
            )
            axes[1, col].imshow((d["surface_semantic"][1] == 3)[::-1, ::-1])
            axes[0, col].set_title(f"Actor scripted frame {r['frame_id']}")
    fig.suptitle("Fixed ego, moved actor: dynamic/occlusion diagnostic only")
    fig.tight_layout()
    fig.savefig(root / "temporal_dynamic.png")
    plt.close(fig)
    atomic_json(
        a.output,
        dict(
            schema_version="hsd-m46-summary-metrics-v1",
            results=metrics,
            sources_sha256={
                p: file_hash(p)
                for p in [
                    "reports/m46_occupancy_validation.json",
                    "reports/m46_occupancy_dense.json",
                    a.driving,
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
