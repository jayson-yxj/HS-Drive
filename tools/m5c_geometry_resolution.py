"""Validation-only matched-ray geometry and native target sampling audit; no training."""

import json, time, argparse
from pathlib import Path
import numpy as np
import torch
from tools.m5a_learning import encoder_load, stack
from datasets.m5a_training import VisualEvidenceDataset
from datasets.m5b_pairs import camera_rays
from models.m5b_cached_encoder import cache_rgb
from models.view_transform import feature_uv, unproject
from geometry.reference import unproject_pixels_to_optical, transform_points
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", default="/root/gpufree-data/hs-drive-data/processed/m5b_index"
    )
    parser.add_argument("--prefix", default="m5c")
    args = parser.parse_args()
    for f in ["geometry_resolution", "feature_sampling"]:
        if Path(f"reports/{args.prefix}_{f}.json").exists():
            raise FileExistsError(f)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    ds = VisualEvidenceDataset(args.index, "val")
    records = []
    sampling = []
    outside = []
    start = time.perf_counter()
    for i, row in enumerate(ds.rows):
        if row["configuration"] in [0]:
            continue
        cache = cache_rgb(net, stack([ds[i]], "inputs"))
        _, n, _, h, w = cache["features"].shape
        # All variants use identical INTEGER pixels. Network center is 7.5 etc;
        # rounding shifts all reference variants equally, never just the oracle.
        uv = feature_uv(h, w, cache["image_hw"], "cuda").round()
        u = uv[..., 0].long().cpu().numpy()
        v = uv[..., 1].long().cpu().numpy()
        frame = json.loads(Path(row["raw_frame"]).read_text())
        root = Path(row["raw_frame"]).parent.parent
        man = json.loads((root / "manifest.json").read_text())
        cal = json.loads((root / "calibration.json").read_text())["cameras"]
        depth = []
        valid = []
        target = []
        for name in row["camera_names"]:
            rays, data = camera_rays(root, man, frame, name)
            d, s, inst = data
            yy, xx = np.indices(d.shape)
            c = cal[name]
            points = transform_points(
                unproject_pixels_to_optical(xx, yy, d, c["K"]), c["T_hs_ego_from_opt"]
            ).reshape(*d.shape, 3)
            reliable = np.zeros(d.shape, bool)
            reliable[rays["uv"][:, 1], rays["uv"][:, 0]] = rays["reliable"]
            actorall = np.zeros(d.shape, bool)
            for actor in frame["actors"]:
                if actor["is_ego"]:
                    continue
                box = np.array(actor["corners_hs_ego_m"])
                center = box.mean(0)
                if np.linalg.norm(center[:2]) > 80:
                    continue
                candidate = (
                    (points >= box.min(0) - 0.05) & (points <= box.max(0) + 0.05)
                ).all(-1)
                # Renderer first hit + AABB candidate; ID not assumed equal to CARLA actor ID.
                pix = candidate & np.isfinite(d) & (d >= 1.5) & (d < 80) & (inst > 0)
                obstacle = pix & np.isin(s, [12, 13, 14, 15, 16, 18, 19, 20])
                actorall |= obstacle
                obs = obstacle & reliable
                hit = {}
                for stride in [16, 8]:
                    us = np.rint(
                        (np.arange(640 // stride) + 0.5) * stride - 0.5
                    ).astype(int)
                    vs = np.rint(
                        (np.arange(384 // stride) + 0.5) * stride - 0.5
                    ).astype(int)
                    hit[str(stride)] = dict(
                        raw=int(obstacle[np.ix_(vs, us)].sum()),
                        reliable=int(obs[np.ix_(vs, us)].sum()),
                    )
                sampling.append(
                    dict(
                        group=row["group"],
                        configuration=row["configuration"],
                        camera=name,
                        type_id=actor["type_id"],
                        distance_m=float(np.linalg.norm(center[:2])),
                        aabb_gated_instance_pixels=int(pix.sum()),
                        obstacle_semantic_pixels=int(obstacle.sum()),
                        reliable_obstacle_pixels=int(obs.sum()),
                        boundary_or_visibility_rejected=int(
                            (obstacle & ~reliable).sum()
                        ),
                        stride=hit,
                    )
                )
            depth.append(d[v, u])
            valid.append(reliable[v, u])
            target.append(actorall[v, u])
        d = torch.tensor(np.array(depth)[None], device="cuda")
        mask = torch.tensor(np.array(valid)[None], device="cuda")
        actor = torch.tensor(np.array(target)[None], device="cuda")
        p = unproject(uv.expand(1, n, h, w, 2), d, cache["K"], cache["T"])
        rank, inside = net.splat.ranks(p)
        centers = net.splat.depth_centers
        probs = cache["probs"]
        full = unproject(
            uv.expand(1, n, 48, h, w, 2),
            centers[None, None, :, None, None].expand(1, n, 48, h, w),
            cache["K"],
            cache["T"],
        )
        ids = (centers[None, None, :, None, None] - d[:, :, None]).abs().argmin(2)
        weights = {
            "continuous_gt": None,
            "nearest_lid": torch.nn.functional.one_hot(ids, 48)
            .permute(0, 1, 4, 2, 3)
            .float(),
            "learned_distribution": probs,
            "expected_depth": None,
        }
        expected = (probs * centers[None, None, :, None, None]).sum(2)
        radial = torch.linalg.vector_norm(p[..., :2], dim=-1)
        rr, iv = net.splat.ranks(full)
        outs = mask & ~inside
        outside.append(
            dict(
                group=row["group"],
                configuration=row["configuration"],
                n=int(outs.sum()),
                entering_bev_probability=(
                    float((probs * iv).sum(2)[outs].mean()) if outs.any() else None
                ),
            )
        )
        for mode, weight in weights.items():
            if weight is None:
                q = (
                    p
                    if mode == "continuous_gt"
                    else unproject(
                        uv.expand(1, n, h, w, 2), expected, cache["K"], cache["T"]
                    )
                )
                q = q[:, :, None]
                weight = torch.ones_like(d[:, :, None])
            else:
                q = full
            qi, qinside = net.splat.ranks(q)
            xy = torch.linalg.vector_norm(q[..., :2] - p[:, :, None, ..., :2], dim=-1)
            xyz = torch.linalg.vector_norm(q - p[:, :, None], dim=-1)
            iz = (
                torch.bucketize(q[..., 2].contiguous(), net.splat.z_edges, right=True)
                - 1
            )
            gz = (
                torch.bucketize(p[..., 2].contiguous(), net.splat.z_edges, right=True)
                - 1
            )
            samez = iz == gz[:, :, None]
            samexy = (
                (
                    torch.floor(
                        (q[..., :2] + torch.tensor([20, 30], device="cuda")) / 0.5
                    )
                )
                == (
                    torch.floor(
                        (p[:, :, None, ..., :2] + torch.tensor([20, 30], device="cuda"))
                        / 0.5
                    )
                )
            ).all(-1)
            centroid = (q * weight[..., None]).sum(2)
            spread = (
                (weight * torch.sum((q - centroid[:, :, None]) ** 2, -1)).sum(2)
            ).sqrt()
            measures = dict(
                correct_xy=(weight * samexy).sum(2),
                correct_voxel=(weight * ((qi == rank[:, :, None]) & qinside)).sum(2),
                xy_error_m=(weight * xy).sum(2),
                xyz_error_m=(weight * xyz).sum(2),
                same_height_within1m_mass=(weight * (samez & (xy <= 1) & qinside)).sum(
                    2
                ),
                cross_height_mass=(weight * ~samez).sum(2),
                spatial_spread_rms_m=spread,
                in_grid_mass=(weight * qinside).sum(2),
            )
            for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]:
                m = mask & inside & actor & (radial >= lo) & (radial < hi)
                records.append(
                    dict(
                        group=row["group"],
                        configuration=row["configuration"],
                        mode=mode,
                        distance=f"{lo}_{hi}",
                        n=int(m.sum()),
                        **{
                            k: float(a[m].mean()) if m.any() else None
                            for k, a in measures.items()
                        },
                    )
                )
        print(row["sample_id"], int((mask & inside & actor).sum()), flush=True)
    summary = []
    for mode in weights:
        for distance in ["0_10", "10_15", "15_20", "20_35", "35_80"]:
            r = [
                x
                for x in records
                if x["mode"] == mode and x["distance"] == distance and x["n"]
            ]
            count = sum(x["n"] for x in r)
            summary.append(
                dict(
                    mode=mode,
                    distance=distance,
                    n=count,
                    groups=len(set(x["group"] for x in r)),
                    **{
                        k: sum(x[k] * x["n"] for x in r) / count if count else None
                        for k in measures
                    },
                )
            )
    common = dict(
        kind="actual_RTX4090_offline_existing_validation_only",
        source_sha256=file_hash(__file__),
        encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
        test_used=False,
        elapsed_s=time.perf_counter() - start,
    )
    atomic_json(
        f"reports/{args.prefix}_geometry_resolution.json",
        dict(
            **common,
            records=records,
            summary=summary,
            outside_gt_rays=outside,
            uv_contract="All four projections use rounded integer feature centers; probabilities remain associated with the original half-pixel feature centers. No sample mismatch between variants. This isolates bin geometry, not a replacement of native splat.",
            reference="Reliable rendered first-hit obstacle surfaces inside non-ego AABB candidates; not independent Mesh truth. Distribution metrics integrate all48 bins. Expected-depth is diagnostic only.",
        ),
    )
    atomic_json(
        f"reports/{args.prefix}_feature_sampling.json",
        dict(
            **common,
            records=sampling,
            stride8="Hypothetical grid sampling only. No encoder/DepthBin/Splat changes.",
            estimated_resources=dict(
                stride16_rays=6 * 24 * 40,
                stride8_rays=6 * 48 * 80,
                lift_point_factor=4,
                depth_probability_fp32_mib_stride16=6 * 24 * 40 * 48 * 4 / 2**20,
                depth_probability_fp32_mib_stride8=6 * 48 * 80 * 48 * 4 / 2**20,
                latency="4x projection/scatter point workload; full inference latency and peak memory require a separate implementation benchmark; backbone not estimated as4x.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
