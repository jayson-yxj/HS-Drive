"""Fixed validation examples: matched-ray depth distributions and spatial projection."""

import json
from pathlib import Path
import numpy as np, torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tools.m5a_learning import encoder_load, stack
from datasets.m5a_training import VisualEvidenceDataset
from datasets.m5b_pairs import camera_rays
from models.m5b_cached_encoder import cache_rgb
from models.view_transform import feature_uv, unproject
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_probability_examples.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5c_index", "val", supervision=False
    )
    records = []
    for cfg in [1, 3, 4, 5, 6, 11]:
        i = next(
            i
            for i, r in enumerate(ds.rows)
            if r["group"] == "Town01_c2" and r["configuration"] == cfg
        )
        row = ds.rows[i]
        cache = cache_rgb(net, stack([ds[i]], "inputs"))
        _, n, _, h, w = cache["features"].shape
        uv = feature_uv(h, w, cache["image_hw"], "cuda").round()
        u = uv[..., 0].long().cpu().numpy()
        v = uv[..., 1].long().cpu().numpy()
        frame = json.loads(Path(row["raw_frame"]).read_text())
        root = Path(row["raw_frame"]).parent.parent
        man = json.loads((root / "manifest.json").read_text())
        dep = []
        ok = []
        for name in row["camera_names"]:
            rays, data = camera_rays(root, man, frame, name)
            m = np.zeros(data[0].shape, bool)
            candidate = np.zeros(len(rays["endpoints"]), bool)
            for actor in frame["actors"]:
                if actor["is_ego"]:
                    continue
                box = np.array(actor["corners_hs_ego_m"])
                candidate |= (
                    (rays["endpoints"] >= box.min(0) - 0.05)
                    & (rays["endpoints"] <= box.max(0) + 0.05)
                ).all(-1)
            rmask = rays["reliable"] & candidate & np.isin(rays["semantics"], [2, 3])
            xy = rays["uv"][rmask]
            m[xy[:, 1], xy[:, 0]] = True
            dep.append(data[0][v, u])
            ok.append(m[v, u])
        ids = np.argwhere(np.array(ok))
        r = dict(group=row["group"], configuration=cfg, n_target_feature_rays=len(ids))
        if len(ids):
            camera, iy, ix = ids[0]
            d = float(dep[camera][iy, ix])
            p = cache["probs"][0, camera, :, iy, ix].cpu().numpy()
            centers = net.splat.depth_centers.cpu().numpy()
            j = abs(centers - d).argmin()
            uvone = uv[iy, ix].reshape(1, 1, 1, 2)
            K = cache["K"][:, camera : camera + 1]
            T = cache["T"][:, camera : camera + 1]
            gt = (
                unproject(uvone, torch.tensor([[[d]]], device="cuda"), K, T)[0, 0, 0]
                .cpu()
                .numpy()
            )
            points = (
                unproject(
                    uvone.expand(1, 1, 48, 2),
                    net.splat.depth_centers.reshape(1, 1, 48),
                    K,
                    T,
                )[0, 0]
                .cpu()
                .numpy()
            )
            r.update(
                camera=row["camera_names"][camera],
                uv=uvone.cpu().tolist(),
                gt_depth=d,
                centers=centers.tolist(),
                probabilities=p.tolist(),
                gt_hs=gt.tolist(),
                bin_points_hs=points.tolist(),
            )
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].bar(
                centers,
                p,
                width=np.diff(np.r_[centers, 80]),
                align="center",
                alpha=0.65,
            )
            axes[0].axvline(d, color="red", label="GT continuous")
            axes[0].axvline(
                centers[j], color="green", ls="--", label="oracle nearest bin"
            )
            axes[0].set_xlabel("Optical Z (m)")
            axes[0].set_ylabel("probability mass")
            axes[0].legend(fontsize=8)
            axes[1].scatter(
                points[:, 0], points[:, 2], c=p, s=25 + 800 * p, cmap="viridis"
            )
            axes[1].scatter(gt[0], gt[2], marker="x", s=80, c="red", label="GT")
            axes[1].scatter(
                points[j, 0],
                points[j, 2],
                marker="+",
                s=100,
                c="green",
                label="nearest bin",
            )
            axes[1].set_xlabel("ego X(m)")
            axes[1].set_ylabel("ego Z(m)")
            axes[1].set_ylim(-2, 4)
            axes[1].set_xlim(0, 60)
            axes[1].legend(fontsize=8)
            fig.suptitle(f"Town01_c2 config{cfg}, first reliable feature ray")
            fig.tight_layout()
            path = Path(f"reports/m5c_figures/m5c_probability_config{cfg}.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            r["figure"] = str(path)
        records.append(r)
    atomic_json(
        dest,
        dict(
            kind="actual_GPU_validation_fixed_first_target_ray_diagnostic",
            records=records,
            source_sha256=file_hash(__file__),
            selection="Fixed Town01_c2 configs1/3/4/5/6/11, lexicographically first valid ray; zero-ray cases retained. No test selection.",
        ),
    )


if __name__ == "__main__":
    main()
