"""Actual stride8 feature, UV and workload audit; no interpolation quality claim."""

import json, time
from pathlib import Path
import numpy as np, torch
from tools.m5a_learning import encoder_load, stack
from datasets.m5a_training import VisualEvidenceDataset
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb, cached_forward
from models.view_transform import feature_uv, unproject
from datasets.m5b_pairs import camera_rays
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5d_stride8_analysis.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5d_index", "val", supervision=False
    )
    records = []
    resources = []
    geometry = []
    for i, row in enumerate(ds.rows):
        if row["configuration"] not in [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14]:
            continue
        inp = stack([ds[i]], "inputs")
        frame = json.loads(Path(row["raw_frame"]).read_text())
        raw = Path(row["raw_frame"]).parent.parent
        man = json.loads((raw / "manifest.json").read_text())
        raydata = [camera_rays(raw, man, frame, n) for n in row["camera_names"]]
        bevs = {}
        for stride, fn in [(16, cache_rgb), (8, cache_p8)]:
            torch.cuda.reset_peak_memory_stats()
            times = []
            for repeat in range(4):
                torch.cuda.synchronize()
                start = time.perf_counter()
                cache = fn(net, inp)
                with torch.no_grad():
                    bev = cached_forward(net, cache)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            bevs[stride] = bev
            _, n, _, h, w = cache["features"].shape
            uv = feature_uv(h, w, cache["image_hw"], "cuda").round()
            u = uv[..., 0].long().cpu().numpy()
            v = uv[..., 1].long().cpu().numpy()
            dep = []
            masks = []
            for name, (r, data) in zip(row["camera_names"], raydata):
                candidate = np.zeros(len(r["endpoints"]), bool)
                for actor in frame["actors"]:
                    if actor["is_ego"]:
                        continue
                    box = np.array(actor["corners_hs_ego_m"])
                    candidate |= (
                        (r["endpoints"] >= box.min(0) - 0.05)
                        & (r["endpoints"] <= box.max(0) + 0.05)
                    ).all(-1)
                reliable = r["reliable"] & candidate & np.isin(r["semantics"], [2, 3])
                mask = np.zeros(data[0].shape, bool)
                xy = r["uv"][reliable]
                mask[xy[:, 1], xy[:, 0]] = True
                dep.append(data[0][v, u])
                masks.append(mask[v, u])
                records.append(
                    dict(
                        group=row["group"],
                        configuration=row["configuration"],
                        camera=name,
                        stride=stride,
                        reliable_native_pixels=int(reliable.sum()),
                        feature_rays=int(mask[v, u].sum()),
                    )
                )
            resources.append(
                dict(
                    group=row["group"],
                    configuration=row["configuration"],
                    stride=stride,
                    feature_shape=list(cache["features"].shape),
                    splat_points=cache["probs"].numel(),
                    latencies_ms=times[1:],
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    probability_sum_max_error=float(
                        (cache["probs"].sum(2) - 1).abs().max()
                    ),
                    all_finite=bool(torch.isfinite(bev).all()),
                )
            )
            d = torch.tensor(np.array(dep)[None], device="cuda")
            mask = torch.tensor(np.array(masks)[None], device="cuda")
            gt = unproject(uv.expand(1, n, h, w, 2), d, cache["K"], cache["T"])
            ids, inside = net.splat.ranks(gt)
            rad = gt[..., :2].norm(dim=-1)
            centers = net.splat.depth_centers
            full = unproject(
                uv.expand(1, n, 48, h, w, 2),
                centers[None, None, :, None, None].expand(1, n, 48, h, w),
                cache["K"],
                cache["T"],
            )
            binid = (d[:, :, None] - centers[None, None, :, None, None]).abs().argmin(2)
            onehot = (
                torch.nn.functional.one_hot(binid, 48).permute(0, 1, 4, 2, 3).float()
            )
            probs = cache["probs"]
            expected = (probs * centers[None, None, :, None, None]).sum(2)
            for mode in [
                "continuous_gt",
                "nearest_lid",
                "learned_probability",
                "expected_depth",
                "soft_xy_oracle_bin",
            ]:
                if mode in ["continuous_gt", "expected_depth"]:
                    points = unproject(
                        uv.expand(1, n, h, w, 2),
                        d if mode == "continuous_gt" else expected,
                        cache["K"],
                        cache["T"],
                    )[:, :, None]
                    weight = torch.ones_like(d[:, :, None])
                else:
                    points = full
                    weight = probs if mode == "learned_probability" else onehot
                rr, iv = net.splat.ranks(points)
                xy = (points[..., :2] - gt[:, :, None, ..., :2]).norm(dim=-1)
                xyz = (points - gt[:, :, None]).norm(dim=-1)
                z = torch.bucketize(
                    points[..., 2].contiguous(), net.splat.z_edges, right=True
                )
                gz = torch.bucketize(
                    gt[..., 2].contiguous(), net.splat.z_edges, right=True
                )
                samez = z == gz[:, :, None]
                exact = (rr == ids[:, :, None]) & iv
                if mode == "soft_xy_oracle_bin":
                    # Bilinear neighboring cell-center mass, no GT-dependent inference operation.
                    gtcenter = (
                        torch.floor(
                            (gt[..., :2] + torch.tensor([20, 30], device="cuda")) / 0.5
                        )
                        + 0.5
                    ) * 0.5 - torch.tensor([20, 30], device="cuda")
                    mass = (
                        (1 - (points[..., :2] - gtcenter[:, :, None]).abs() / 0.5)
                        .clamp(0, 1)
                        .prod(-1)
                        * samez
                        * iv
                    )
                    correct = (weight * mass).sum(2)
                else:
                    correct = (weight * exact).sum(2)
                measures = dict(
                    correct_voxel_mass=correct,
                    xy_error_m=(weight * xy).sum(2),
                    xyz_error_m=(weight * xyz).sum(2),
                    same_layer_1m_mass=(weight * samez * (xy <= 1) * iv).sum(2),
                    cross_layer_mass=(weight * ~samez).sum(2),
                )
                for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]:
                    m = mask & inside & (rad >= lo) & (rad < hi)
                    geometry.append(
                        dict(
                            group=row["group"],
                            configuration=row["configuration"],
                            stride=stride,
                            mode=mode,
                            distance=f"{lo}_{hi}",
                            n=int(m.sum()),
                            **{
                                k: float(x[m].mean()) if m.any() else None
                                for k, x in measures.items()
                            },
                        )
                    )
        resources[-1]["bev_vs_stride16_mean_abs"] = float(
            (bevs[8] - bevs[16]).abs().mean()
        )
        print(row["sample_id"], flush=True)
    atomic_json(
        dest,
        dict(
            kind="actual_RTX4090_new_validation_true_layer2_P8",
            records=records,
            resources=resources,
            source_sha256=file_hash(__file__),
            feature_source_sha256=file_hash("models/m5d_stride8.py"),
            limits="Real prepool ResNet layer2 information; existing lateral/output weights reused, frozen Depth distribution shifts. UV uses existing uniform imagecell-center convention for each resolution. This is a grid representation convention, not exact ResNet receptivefield center claim. Splat/48bins unchanged. BEV L1 is output change, not causal quality ranking.",
        ),
    )
    summary = []
    for stride in [16, 8]:
        for mode in [
            "continuous_gt",
            "nearest_lid",
            "learned_probability",
            "expected_depth",
            "soft_xy_oracle_bin",
        ]:
            for band in ["0_10", "10_15", "15_20", "20_35", "35_80"]:
                rows = [
                    r
                    for r in geometry
                    if r["stride"] == stride
                    and r["mode"] == mode
                    and r["distance"] == band
                    and r["n"]
                ]
                count = sum(r["n"] for r in rows)
                summary.append(
                    dict(
                        stride=stride,
                        mode=mode,
                        distance=band,
                        n=count,
                        groups=len(set(r["group"] for r in rows)),
                        **{
                            k: (
                                sum(r[k] * r["n"] for r in rows) / count
                                if count
                                else None
                            )
                            for k in measures
                        },
                    )
                )
    atomic_json(
        "reports/m5d_geometry_resolution.json",
        dict(
            kind="actual_GPU_matched_rays_within_each_stride_offline_validation",
            summary=summary,
            records=geometry,
            source_sha256=file_hash(__file__),
            limits="Compare four geometry modes on exactly same rays WITHIN stride only. Different stride sample populations cannot be attributed to bins. SoftXY is oraclebin bilinear diagnostic, not implemented default or trained candidate. Oracle subbin fraction known would recover continuousGT centroid, unavailable from categorical output without extra supervision. No Depth or Splat fine-tuning performed.",
        ),
    )


if __name__ == "__main__":
    main()
