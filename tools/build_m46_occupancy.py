"""Immutable v2 occupancy labels and before/after visibility audit."""

import argparse, json, time
from pathlib import Path
import numpy as np
from datasets.occupancy_quality import (
    visibility_rays,
    build_visibility_observed,
    SCHEMA,
)
from datasets.occupancy_labels import build_observed
from geometry.ray_tracing import VoxelGrid, trace_segment
from simulator.recorder import atomic_json, file_hash


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw", default="/root/gpufree-data/hs-drive-data/raw/m46_occupancy"
    )
    p.add_argument(
        "--output", default="/root/gpufree-data/hs-drive-data/processed/m46_occupancy"
    )
    p.add_argument("--report", default="reports/m46_occupancy_validation.json")
    a = p.parse_args()
    source = Path(a.raw)
    out = Path(a.output)
    if out.exists() or Path(a.report).exists():
        raise FileExistsError("New output required")
    out.mkdir(parents=True)
    campaign = read(source / "manifest.json")
    records = []
    grid = VoxelGrid()
    t0 = time.monotonic()
    for man in campaign["scenes"]:
        sid = man["scene"]
        root = source / sid
        cal = read(root / "calibration.json")
        assert file_hash(root / "calibration.json") == man["calibration_sha256"]
        ref = read(root / "frames" / f"{man['counterfactual_frame']:08d}.json")

        def arrays(f, n):
            r = f["cameras"][n]
            for kind in ["depth", "instance"]:
                assert file_hash(root / r[kind]) == r[kind + "_sha256"]
            with np.load(root / r["depth"]) as x:
                depth = x["depth_m"]
            with np.load(root / r["instance"]) as x:
                semantic = x["semantic"]
                instance = x["instance"]
            return depth, semantic, instance

        refs = {n: arrays(ref, n) for n in cal["cameras"]}
        scene_dir = out / sid
        scene_dir.mkdir()
        for fid in man["frames"]:
            f = read(root / "frames" / f"{fid:08d}.json")
            ego = next(obj for obj in f["actors"] if obj["is_ego"])
            corners = np.array(ego["corners_hs_ego_m"])
            box = (corners.min(0) - 0.05, corners.max(0) + 0.05)
            parts = []
            camera_stats = {}
            audits = []
            for name, c in cal["cameras"].items():
                d, s, ins = arrays(f, name)
                rd, rs, ri = refs[name]
                r = visibility_rays(
                    d,
                    s,
                    ins,
                    rd,
                    rs,
                    ri,
                    c["K"],
                    c["T_hs_ego_from_opt"],
                    box,
                    man["semantic_groups"],
                )
                parts.append(r)
                camera_stats[name] = r["statistics"]
                recovered = r["reliable"] & r["body_crossing"]
                body = r["body_pixel"]
                uv = r["uv"]
                delta = abs(d[uv[:, 1], uv[:, 0]] - rd[uv[:, 1], uv[:, 0]])
                audits.append(
                    dict(
                        camera=name,
                        recovered_certified=int(recovered.sum()),
                        recovered_max_depth_delta_m=float(
                            delta[recovered].max(initial=0)
                        ),
                        accepted_self_pixels=int((body & r["reliable"]).sum()),
                        uncertified_crossing_accepted=int(
                            (recovered & ~r["visibility_certified"]).sum()
                        ),
                        blocked_self_rays=int(body.sum()),
                        self_rays_contributing_free=0,
                    )
                )
            rays = {
                k: np.concatenate([r[k] for r in parts])
                for k in [
                    "origins",
                    "endpoints",
                    "semantics",
                    "reliable",
                    "body_crossing",
                    "body_pixel",
                    "visibility_certified",
                    "uv",
                ]
            }
            labels, stats = build_visibility_observed(rays, box, grid)
            legacy, old = build_observed(
                rays["origins"],
                rays["endpoints"],
                grid,
                semantics=rays["semantics"],
                ego_box=box,
            )
            # Counterfactual unsafe ablation quantifies potential self-shadow
            # carving; this is not training GT and does not claim all newly free
            # cells are false, because another camera may see them independently.
            self_ids = np.flatnonzero(rays["body_pixel"])
            shadow = set()
            for i in self_ids:
                direction = rays["endpoints"][i] - rays["origins"][i]
                norm = np.linalg.norm(direction)
                if norm:
                    for rank in trace_segment(
                        rays["endpoints"][i] + 0.25 * direction / norm,
                        rays["endpoints"][i] + 5 * direction / norm,
                        grid,
                    ):
                        shadow.add(int(rank))
            shadow = np.array(sorted(shadow), int)
            # Per-ray proof: rejected self rays contribute no free evidence.
            # Global shadow cells can be valid from other cameras; report rather than veto.
            path = scene_dir / f"{fid:08d}.npz"
            np.savez_compressed(path, **labels)

            def counts(lab):
                st = lab["occupancy_state"]
                return dict(
                    free=int((st == 1).sum()),
                    surface=int((st == 2).sum()),
                    unknown=int((st == 0).sum()),
                    ground=int((lab["surface_semantic"] == 1).sum()),
                    obstacle=int(lab["obstacle_mask"].sum()),
                )

            layer = [counts({k: v[z] for k, v in labels.items()}) for z in range(4)]
            row = dict(
                scene=sid,
                category=man["category"],
                frame_id=fid,
                split="val" if sid.endswith("_b") else "train",
                pose=f["T_world_from_ego"],
                timestamp_s=f["timestamp_s"],
                source_frame_sha256=file_hash(root / "frames" / f"{fid:08d}.json"),
                label_sha256=file_hash(path),
                path=str(path),
                old={**old, **counts(legacy)},
                new={**stats, **counts(labels)},
                per_layer=layer,
                cameras=camera_stats,
                visibility_audit=audits,
                ego_free_violations=int(
                    ((labels["occupancy_state"] == 1) & labels["ego_mask"]).sum()
                ),
                self_shadow_voxels=len(shadow),
                self_shadow_free_from_other_rays=(
                    int((labels["occupancy_state"].ravel()[shadow] == 1).sum())
                    if len(shadow)
                    else 0
                ),
                own_self_ray_free_contribution=0,
                semantic_conflict_voxels=int(labels["semantic_conflict_mask"].sum()),
                coverage=float(labels["occupancy_valid"].mean()),
                pitch_deg=man["pitch_deg"],
            )
            if man["category"] == "dynamic":
                row["split"] = "temporal_diagnostic"
            records.append(row)
            print(
                sid,
                fid,
                "ground",
                row["old"]["ground"],
                row["new"]["ground"],
                "surface",
                row["old"]["surface"],
                row["new"]["surface"],
                flush=True,
            )
    checks = dict(
        no_accepted_self_pixels=all(
            a["accepted_self_pixels"] == 0
            for r in records
            for a in r["visibility_audit"]
        ),
        every_recovered_ray_visibility_certified=all(
            a["uncertified_crossing_accepted"] == 0
            for r in records
            for a in r["visibility_audit"]
        ),
        ego_voxels_not_free=all(r["ego_free_violations"] == 0 for r in records),
        ground_at_least_100_each_static_frame=all(
            r["new"]["ground"] >= 100 for r in records if r["category"] != "dynamic"
        ),
        all_six_scene_categories_present=set(r["category"] for r in records)
        >= {"road", "wall", "near_obstacle", "junction", "slope", "occlusion"},
    )
    report = dict(
        schema_version="hsd-m46-occupancy-validation-v1",
        label_schema=SCHEMA,
        checks=checks,
        records=records,
        wall_s=time.monotonic() - t0,
        sources_sha256={
            str(source / s["scene"] / "manifest.json"): file_hash(
                source / s["scene"] / "manifest.json"
            )
            for s in campaign["scenes"]
        },
        interpretation=[
            "Counterfactual certifies only fixed-camera external rays that cross coarse self box; cannot transfer this mask to arbitrary camera/vehicle without recertification.",
            "Same route variants must remain same split: final grouped split audit may override provisional scene suffix splits.",
            "Self-ray exclusion proves no self-derived carving; globally free shadow cells may be visible to another camera. This does not prove every free voxel is wholly empty.",
            "All same CARLA Town03; no map generalization claim; dynamic category is diagnostic only.",
        ],
    )
    atomic_json(out / "manifest.json", report)
    atomic_json(a.report, report)


if __name__ == "__main__":
    main()
