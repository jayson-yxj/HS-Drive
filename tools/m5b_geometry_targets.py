"""Corrected actor-only validation geometry audit: excludes ego and background actors."""

import json
from pathlib import Path
import numpy as np, torch
from tools.m5a_learning import encoder_load, stack
from models.m5b_cached_encoder import cache_rgb
from models.view_transform import feature_uv, unproject
from datasets.m5a_training import VisualEvidenceDataset
from datasets.m5b_pairs import camera_rays
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    out = Path("reports/m5b_target_geometry.json")
    if out.exists():
        raise FileExistsError(out)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_index", "val"
    )
    records = []
    for i, row in enumerate(ds.rows):
        if row["configuration"] not in [1, 2, 3, 4, 7, 9]:
            continue
        item = ds[i]
        cache = cache_rgb(net, stack([item], "inputs"))
        b, n, c, h, w = cache["features"].shape
        uv = feature_uv(h, w, cache["image_hw"], "cuda")
        u = uv[..., 0].round().long().cpu().numpy()
        v = uv[..., 1].round().long().cpu().numpy()
        frame = json.loads(Path(row["raw_frame"]).read_text())
        root = Path(row["raw_frame"]).parent.parent
        man = json.loads((root / "manifest.json").read_text())
        depth = []
        valid = []
        for name in row["camera_names"]:
            rays, data = camera_rays(root, man, frame, name)
            m = np.zeros(data[0].shape, bool)
            m[rays["uv"][:, 1], rays["uv"][:, 0]] = rays["reliable"]
            depth.append(data[0][v, u])
            valid.append(m[v, u] & np.isin(data[1][v, u], [12, 13, 14, 15, 16, 18, 19]))
        d = torch.tensor(np.array(depth)[None], device="cuda")
        mask = (
            torch.tensor(np.array(valid)[None], device="cuda") & (d >= 1.5) & (d < 80)
        )
        p = unproject(uv.expand(1, n, h, w, 2), d, cache["K"], cache["T"])
        rank, inside = net.splat.ranks(p)
        target = torch.zeros_like(mask)
        for actor in frame["actors"]:
            if actor["is_ego"]:
                continue
            box = torch.tensor(actor["corners_hs_ego_m"], device="cuda")
            target |= (
                (p >= box.min(0).values - 0.05) & (p <= box.max(0).values + 0.05)
            ).all(-1)
        mask &= inside & target
        centers = net.splat.depth_centers
        points = unproject(
            uv.expand(1, n, 48, h, w, 2),
            centers[None, None, :, None, None].expand(1, n, 48, h, w),
            cache["K"],
            cache["T"],
        )
        rr, iv = net.splat.ranks(points)
        same = (rr == rank[:, :, None]) & iv
        probs = cache["probs"]
        binid = (centers[None, None, :, None, None] - d[:, :, None]).abs().argmin(2)
        onehot = torch.nn.functional.one_hot(binid, 48).permute(0, 1, 4, 2, 3)

        def mean(x):
            return float(x[mask].mean()) if mask.any() else None

        records.append(
            dict(
                group=row["group"],
                configuration=row["configuration"],
                n=int(mask.sum()),
                gt_bin_probability=mean(probs.gather(2, binid[:, :, None]).squeeze(2)),
                within3m_mass=mean(
                    (
                        probs
                        * (
                            (centers[None, None, :, None, None] - d[:, :, None]).abs()
                            < 3
                        )
                    ).sum(2)
                ),
                correct_voxel_mass=mean((probs * same).sum(2)),
                oracle_bin_correct_voxel=mean((onehot * same).sum(2).float()),
                expected_xy_error_m=mean(
                    (
                        probs
                        * torch.linalg.vector_norm(
                            points[..., :2] - p[:, :, None, :, :, :2], dim=-1
                        )
                    ).sum(2)
                ),
            )
        )
    atomic_json(
        out,
        dict(
            kind="actual_GPU_offline_validation_only_target_actor_geometry",
            records=records,
            source_sha256=file_hash(__file__),
            supersedes_scope="Initial geometry feature_actor_pixels used camera semantic actor tags including ego. This audit requires reliable rendered points inside non-ego placed Actor boxes. Initial interventions unchanged; corrected attribution only, no retraining.",
            reference="Rendered first-hit surfaces, AABB candidate gate only. Feature-center sampling misses small objects; zero n stays null.",
            test_used=False,
        ),
    )


if __name__ == "__main__":
    main()
