"""Research plots with raw references and explicit conditional/proxy labels."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tools.m5l_common import rows
import cv2


def main():
    out = Path("reports/m5m_figures")
    out.mkdir(exist_ok=True)
    r = json.loads(Path("reports/m5m_risk_coverage.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for seed in [101, 102]:
        rr = [
            v
            for v in r["curves"]
            if v["split"] == "val" and v["seed"] == seed and v["conditional"]
        ]
        x = [v["coverage"] for v in rr]
        for ax, k in zip(axs, ["mean", "p95", "severe5m_rate"]):
            ax.plot(
                x,
                [v["xy"][k] if k != "severe5m_rate" else v[k] for v in rr],
                "o-",
                label=str(seed),
            )
            ax.axvline(0.3, color="gray", ls="--")
            ax.set_xlabel("All native visible actor-view coverage")
            ax.set_ylabel(k + " (XY center proxy)")
            ax.grid(alpha=0.3)
    axs[1].axhline(5, color="red", ls="--")
    axs[2].axhline(0.05, color="red", ls="--")
    axs[0].legend()
    fig.suptitle(
        "Validation RGB-only selective rules: coverage and tail gates separate"
    )
    fig.tight_layout()
    fig.savefig(out / "m5m_risk_coverage.png", dpi=160)
    plt.close(fig)
    r = json.loads(Path("reports/m5m_gating_val.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for seed in [101, 102]:
        names = [
            f"{n}_{seed}"
            for n in [
                "original",
                "no_mobile_unconditional",
                "no_mobile_rgb0.9",
                "no_mobile_oracle_vehicle",
            ]
        ]
        for ax, band in zip(axs, ["20_35", "35_80"]):
            ax.plot(
                range(4),
                [
                    next(
                        v["bias"]
                        for v in r["depth"]
                        if v["method"] == n
                        and v["scope"] == "all"
                        and v["band"] == band
                    )
                    for n in names
                ],
                "o-",
                label=str(seed),
            )
            ax.set_xticks(
                range(4),
                ["Original", "Unconditional", "RGB q>=.9", "Oracle vehicle"],
                rotation=15,
            )
            ax.set_title(band + " m / all reliable GT-in-BEV surfaces")
            ax.set_ylabel("Expected depth signed bias (m)")
            ax.grid(alpha=0.3)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out / "m5m_residual_bias.png", dpi=160)
    plt.close(fig)
    data = json.loads(Path("runs/m5m_contact_val.json").read_text())
    rs = {r["sample_id"]: r for r in rows("val")}
    cases = sorted(
        [r for r in data["actors"] if r["seed"] == 102 and r["baseline_valid"]],
        key=lambda r: r["plane_center"]["xy_m"],
        reverse=True,
    )[:4]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, r in zip(axs.ravel(), cases):
        row = rs[r["sample_id"]]
        frame = json.loads(Path(row["raw_frame"]).read_text())
        root = Path(row["raw_frame"]).parent.parent
        im = cv2.imread(str(root / frame["cameras"][r["camera"]]["rgb"]))[:, :, ::-1]
        ax.imshow(im)
        x, y, w, h = r["output"]["image_region"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="red"))
        uv = r["native_bottom_uv"]
        ax.plot(*uv, "go", label="native visible bottom (not Gold)")
        ax.set_title(
            f"{r['group']} {r['type']} XY={r['plane_center']['xy_m']:.1f}m; Unknown"
        )
        ax.set_xlim(max(0, x - 60), min(640, x + w + 60))
        ax.set_ylim(min(384, y + h + 60), max(0, y - 40))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "m5m_contact_failures.png", dpi=160)
    plt.close(fig)
    # 3D actor/road relation: AABB proxy, not solid mesh.
    refs = [
        r
        for r in data["references"]
        if r["rendered_local_road"] and abs(r["aabb_bottom_road_gap_m"]) > 0.25
    ]
    fig = plt.figure(figsize=(12, 5))
    for i, r in enumerate(
        sorted(refs, key=lambda r: abs(r["aabb_bottom_road_gap_m"]), reverse=True)[:2]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        row = rs[r["sample_id"]]
        f = json.loads(Path(row["raw_frame"]).read_text())
        a = next(a for a in f["actors"] if a["actor_id"] == r["actor_id"])
        box = np.array(a["corners_hs_ego_m"])
        ax.scatter(*box.T, label="AABB corners")
        xy = np.array(r["center"][:2])
        xx, yy = np.meshgrid(
            np.linspace(xy[0] - 3, xy[0] + 3, 12), np.linspace(xy[1] - 3, xy[1] + 3, 12)
        )
        p = r["rendered_local_road"]["coefficients"]
        ax.plot_surface(xx, yy, p[0] * xx + p[1] * yy + p[2], alpha=0.35)
        b = r["native_bottom_xyz"]
        ax.scatter(*b, c="red", s=50, label="visible bottom")
        ax.set_title(f"{r['type']} possible gap {r['aabb_bottom_road_gap_m']:.2f} m")
        ax.set_xlabel("HS x")
        ax.set_ylabel("HS y")
        ax.set_zlabel("HS z")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "m5m_contact_geometry.png", dpi=160)
    plt.close(fig)
    p = Path("runs/m5m_logs/shadow_raw.json")
    if p.exists():
        rr = json.loads(p.read_text())
        fig, ax = plt.subplots(figsize=(13, 4))
        offset = 0
        for phase in rr["phases"]:
            t = phase["raw_tick_ms"]
            ax.plot(
                np.arange(len(t)) + offset,
                t,
                label=f"r{phase['repeat']} {phase['phase']}",
            )
            offset += len(t)
        ax.axhline(50, color="red", ls="--")
        ax.set_ylabel("Raw CARLA tick ms")
        ax.set_xlabel("Sequential measured tick")
        ax.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(out / "m5m_shadow_ticks.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
