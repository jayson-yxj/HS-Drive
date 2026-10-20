"""Six-camera render evidence for restored box rays and rejected body pixels."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.occupancy_quality import visibility_rays
from simulator.recorder import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="/root/gpufree-data/hs-drive-data/raw/m46_occupancy"
    )
    parser.add_argument("--output", default="reports/m46_visibility_figures")
    parser.add_argument(
        "--scenes", nargs="+", default=["road_a", "occlusion_b", "slope_a"]
    )
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    evidence = []
    for scene in args.scenes:
        root = Path(args.raw) / scene
        man = json.loads((root / "manifest.json").read_text())
        cal = json.loads((root / "calibration.json").read_text())["cameras"]
        frame = json.loads(
            (root / "frames" / f"{man['frames'][0]:08d}.json").read_text()
        )
        ref = json.loads(
            (root / "frames" / f"{man['counterfactual_frame']:08d}.json").read_text()
        )
        corners = np.array(
            next(o for o in frame["actors"] if o["is_ego"])["corners_hs_ego_m"]
        )
        box = (corners.min(0) - 0.05, corners.max(0) + 0.05)
        fig, axes = plt.subplots(6, 3, figsize=(14, 17))
        for row, (name, c) in enumerate(cal.items()):

            def load(f):
                d = f["cameras"][name]
                for key in ("depth", "instance", "rgb"):
                    assert file_hash(root / d[key]) == d[key + "_sha256"]
                with np.load(root / d["depth"]) as x:
                    depth = x["depth_m"]
                with np.load(root / d["instance"]) as x:
                    sem = x["semantic"]
                    ins = x["instance"]
                return depth, sem, ins, cv2.imread(str(root / d["rgb"]))[:, :, ::-1]

            d, s, i, rgb = load(frame)
            rd, rs, ri, rrgb = load(ref)
            rays = visibility_rays(
                d,
                s,
                i,
                rd,
                rs,
                ri,
                c["K"],
                c["T_hs_ego_from_opt"],
                box,
                man["semantic_groups"],
            )
            recovered = rays["reliable"] & rays["body_crossing"]
            axes[row, 0].imshow(rgb)
            axes[row, 1].imshow(rgb)
            overlay = np.zeros((*d.shape, 4))
            overlay[rays["self_mask"]] = [1, 0, 0, 0.8]
            axes[row, 1].imshow(overlay)
            uv = rays["uv"][recovered]
            axes[row, 1].scatter(uv[:, 0], uv[:, 1], s=3, c="#ffba08")
            axes[row, 2].imshow(rrgb)
            axes[row, 0].set_title(name + " ego present")
            axes[row, 1].set_title("Red: self mask; yellow: certified box-crossing ray")
            axes[row, 2].set_title("Same camera, ego removed (label-only)")
            for ax in axes[row]:
                ax.axis("off")
            evidence.append(dict(scene=scene, camera=name, **rays["statistics"]))
        fig.suptitle(scene + " | counterfactual is never a learned input")
        fig.tight_layout()
        fig.savefig(out / (scene + ".png"), dpi=100)
        plt.close(fig)
    atomic_json(
        out / "manifest.json",
        dict(records=evidence, raw=args.raw, source_sha256=file_hash(__file__)),
    )


if __name__ == "__main__":
    main()
