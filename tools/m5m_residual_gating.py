"""No training: four fixed residuals, legal RGB gates and diagnostic oracle gate."""

import argparse, json, time
from pathlib import Path
import numpy as np, torch
from tools.m5l_common import rows, cache_index, write, asset, group_depth
from tools.m5f_common import inputs, pixel_label
from tools.m5a_learning import encoder_load
from tools.m5k_alignment import sample_gt, RANGES
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import conditioned
from models.m5k_depth_adapter import DepthResidual
from models.m5e_paths import geometry
from models.m5f_lift import lift
from tools.m5f_factorization import reference
from tools.m5j_metrics import evaluate, aggregate
from training.runtime import prepare_cuda
from simulator.recorder import file_hash
from diagnostics.m5m_contact import fit_plane, intersect
from diagnostics.m5l_objects import components
from tools.m5l_object_eval import errors
from geometry.reference import unproject_pixels_to_optical, transform_points


def mix(original, residual, gate):
    if (
        original.shape != residual.shape
        or gate.shape != original.shape[:2] + original.shape[3:]
    ):
        raise ValueError("shape")
    if not torch.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
        raise ValueError("gate")
    return original * (1 - gate[:, :, None]) + residual * gate[:, :, None]


def metrics(net, c, gt, g, pr, name, row, gate):
    dep = gt["depth"]
    exp = (pr * net.splat.depth_centers[None, None, :, None, None]).sum(2)
    mode = net.splat.depth_centers[pr.argmax(2)]
    delta = exp - dep
    fields = dict(
        expected_mae=delta.abs(),
        bias=delta,
        mode_mae=(mode - dep).abs(),
        space1=(pr * g["same_layer"] * (g["xy"] <= 1)).sum(2),
        gate=gate,
    )
    records = []
    for scope in ["all", "actor", "vehicle", "person", "cone", "background", "unknown"]:
        mask = gt["valid"] & g["inside"]
        if scope == "actor":
            mask = mask & gt["actor"]
        if scope in ["vehicle", "person", "cone"]:
            mask = (
                mask
                & gt["actor"]
                & (gt["classes"] == {"vehicle": 1, "person": 2, "cone": 3}[scope])
            )
        if scope == "background":
            mask = mask & ~gt["target"]
        if scope == "unknown":
            mask = (
                ~gt["valid"]
                & g["inside"]
                & torch.isfinite(dep)
                & (dep >= 1.5)
                & (dep < 80)
            )
        for band, lo, hi in RANGES:
            m = mask & (dep >= lo) & (dep < hi)
            n = int(m.sum())
            if n:
                records.append(
                    dict(
                        method=name,
                        group=row["group"],
                        sample_id=row["sample_id"],
                        scope=scope,
                        band=band,
                        n=n,
                        sums={k: float(v[m].double().sum()) for k, v in fields.items()},
                    )
                )
    return records


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    dest = Path(f"runs/m5m_gating_{a.split}")
    dest.mkdir(exist_ok=False)
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    models = []
    for r in json.loads(Path("reports/m5l_vehicle_training.json").read_text())[
        "records"
    ]:
        assert file_hash(r["selected"]["path"]) == r["selected"]["sha256"]
        ck = torch.load(r["selected"]["path"], weights_only=True)
        model = DepthResidual(mobile_condition=r["mode"] == "mobile").cuda().eval()
        model.load_state_dict(ck["adapter"])
        models.append((r, model))
    ix = cache_index(a.split)
    records = []
    bev = []
    planes = []
    objects = json.loads(Path(f"runs/m5m_contact_{a.split}.json").read_text())["actors"]
    lookup = {(r["sample_id"], r["seed"], r["camera"]): [] for r in objects}
    for r in objects:
        lookup[(r["sample_id"], r["seed"], r["camera"])].append(r)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for row in rows(a.split):
        x = inputs(row)
        d = pixel_label(row)
        c = cache_p8(net, x)
        f = conditioned(net, c)
        gt = sample_gt(d, 8)
        g = geometry(net, c, gt["depth"])
        z = np.load(ix[row["sample_id"]])
        truth, valid, actors = reference(d, x, row, row["camera_names"])
        am = [r["mask"] for r in actors]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            log = net.depth_head(f.flatten(0, 1)).float()
        original = log.softmax(1)[None]
        expected = (
            (original * net.splat.depth_centers[None, None, :, None, None])
            .sum(2)[0]
            .cpu()
            .numpy()
        )
        for seed in [101, 102]:
            q = torch.from_numpy(
                z[f"pixel_{seed}"][None, :, 4::8, 4::8].astype(np.float32)
            ).cuda()
            methods = [("original", original, torch.zeros_like(q))]
            for r, model in models:
                if r["seed"] != seed:
                    continue
                residual = model(f[0], log, q[0, :, None]).softmax(1)[None]
                methods.append(
                    (r["mode"] + "_unconditional", residual, torch.ones_like(q))
                )
                for threshold in [0.5, 0.9, 0.95]:
                    gate = (q >= threshold).float()
                    methods.append(
                        (
                            r["mode"] + f"_rgb{threshold}",
                            mix(original, residual, gate),
                            gate,
                        )
                    )
                gate = (gt["valid"] & gt["actor"] & (gt["classes"] == 1)).float()
                methods.append(
                    (r["mode"] + "_oracle_vehicle", mix(original, residual, gate), gate)
                )
            for name, pr, gate in methods:
                name = f"{name}_{seed}"
                records += metrics(net, c, gt, g, pr, name, row, gate)
                score = (
                    lift(net.splat, q, pr, c["K"], c["T"], c["image_hw"])[
                        "event_score"
                    ][0]
                    .cpu()
                    .numpy()
                )
                bev.append(
                    dict(
                        method=name,
                        seed=seed,
                        group=row["group"],
                        sample_id=row["sample_id"],
                        configuration=row["configuration"],
                        empty=(
                            not json.loads(
                                (
                                    Path(row["raw_frame"]).parent.parent
                                    / "manifest.json"
                                ).read_text()
                            )["configurations"][row["configuration"]]
                        ),
                        metrics=evaluate(score, truth, valid, am, 0.2),
                    )
                )
            # RGB-only local plane candidate from unchanged learned depth in lower neighborhood.
            for ci, cam in enumerate(row["camera_names"]):
                K = x["intrinsics"][0, ci].cpu().numpy()
                T = x["cam_to_ego"][0, ci].cpu().numpy()
                yy, xx = np.mgrid[:48, :80]
                u = 4 + xx * 8
                v = 4 + yy * 8
                for r in lookup.get((row["sample_id"], seed, cam), []):
                    if not r["detected"]:
                        continue
                    uv = np.array(r["output"]["image_region"])
                    bx, by, bw, bh = uv
                    center_uv = [bx + bw / 2, by + bh - 1]
                    mask = (
                        (u >= bx - 40)
                        & (u <= bx + bw + 40)
                        & (v >= by + bh)
                        & (v <= by + bh + 64)
                        & (q[0, ci].cpu().numpy() < 0.1)
                    )
                    pts = transform_points(
                        unproject_pixels_to_optical(
                            u[mask], v[mask], expected[ci][mask], K
                        ),
                        T,
                    )
                    plane = fit_plane(pts)
                    accepted = plane is not None and plane["slope_deg"] <= 15
                    # Use exact RGB component bottom from stored output's reconstructed component.
                    cc = next(
                        (
                            cc
                            for cc in components(
                                z[f"pixel_{seed}"][ci].astype(np.float32)
                            )
                            if cc["bbox"] == r["output"]["image_region"]
                        ),
                        None,
                    )
                    pred = (
                        intersect(cc["uv"], K, T, plane)
                        if accepted and cc is not None
                        else None
                    )
                    planes.append(
                        dict(
                            sample_id=row["sample_id"],
                            seed=seed,
                            camera=cam,
                            actor_id=r["actor_id"],
                            group=row["group"],
                            type=r["type"],
                            plane=plane,
                            accepted=pred is not None,
                            reference_status=r["status"],
                            error=(
                                None
                                if pred is None
                                else errors(pred, np.array(r["native_bottom_xyz"]))
                            ),
                            road_plane=r["rendered_local_road"],
                            limits="RGB depth foreground-excluded lower patch, no validated road semantic; conditional research only",
                        )
                    )
        print(row["sample_id"], flush=True)
    write(dest / "raw.json", dict(depth=records, bev=bev, planes=planes))
    write(
        f"reports/m5m_gating_{a.split}.json",
        dict(
            kind="actual_RTX4090_frozen_inference_no_training",
            depth=group_depth(records, ["method", "scope", "band"]),
            by_position=group_depth(records, ["method", "scope", "band", "group"]),
            bev=[
                dict(
                    method=m,
                    **aggregate([r["metrics"] for r in bev if r["method"] == m]),
                )
                for m in sorted({r["method"] for r in bev})
            ],
            empty_removed=[
                dict(
                    method=m,
                    **aggregate(
                        [r["metrics"] for r in bev if r["method"] == m and r["empty"]]
                    ),
                )
                for m in sorted({r["method"] for r in bev})
            ],
            raw=asset(dest / "raw.json"),
            elapsed_s=time.perf_counter() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            source_sha256=file_hash(__file__),
            limits="Unknown raw depth errors only diagnostic, NOT valid truth or negatives. BEV unknown high reported separately. Oracle gate uses GT only diagnostic. Empty configurations must cross-check manifest; removal is paired obstacle removal, not ego removal.",
        ),
    )


if __name__ == "__main__":
    main()
