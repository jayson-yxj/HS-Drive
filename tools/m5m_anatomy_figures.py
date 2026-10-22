"""Anatomy projections: candidate physical geometry, visibly not Gold contacts."""

import json
from pathlib import Path
import numpy as np, cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    records = json.loads(Path("runs/m5m_anatomy_dev.json").read_text())
    rows = json.loads(
        Path(
            "/root/gpufree-data/hs-drive-data/processed/m5m_dev_pixels/index.json"
        ).read_text()
    )["records"]
    ix = {r["sample_id"]: r for r in rows}
    cases = []
    for typ in [
        "vehicle.tesla.model3",
        "vehicle.mercedes.sprinter",
        "walker.pedestrian.0001",
    ]:
        available = [
            r
            for r in records
            if r["type_id"] == typ
            and any(v["inside"] and v["associated_instance_near"] for v in r["views"])
        ]
        if available:
            cases.append(available[0])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, r in zip(axs, cases):
        row = ix[r["sample_id"]]
        root = Path(row["raw_frame"]).parent.parent
        frame = json.loads(Path(row["raw_frame"]).read_text())
        candidates = [v for v in r["views"] if v["inside"]]
        camera = max(
            set(v["camera"] for v in candidates),
            key=lambda c: sum(
                v["associated_instance_near"] for v in candidates if v["camera"] == c
            ),
        )
        vv = [v for v in candidates if v["camera"] == camera]
        im = cv2.imread(str(root / frame["cameras"][camera]["rgb"]))[:, :, ::-1]
        ax.imshow(im)
        for v in vv:
            ax.scatter(
                *v["uv"], s=25, c="lime" if v["associated_instance_near"] else "red"
            )
            ax.annotate(
                v["name"].replace("physics_", ""), v["uv"], fontsize=7, color="yellow"
            )
        uv = np.array([v["uv"] for v in vv])
        lo = uv.min(0)
        hi = uv.max(0)
        ax.set_xlim(max(0, lo[0] - 65), min(640, hi[0] + 65))
        ax.set_ylim(min(384, hi[1] + 45), max(0, lo[1] - 110))
        ax.set_title(
            r["type_id"] + "\nCandidate geometry, NOT Gold contact", fontsize=10
        )
    fig.tight_layout()
    fig.savefig("reports/m5m_figures/m5m_anatomy_candidates.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
