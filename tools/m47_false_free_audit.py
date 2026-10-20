"""Reconstruct all M46 104 cell conflicts with exact sparse-ray provenance."""

import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
from datasets.occupancy_quality import visibility_rays
from datasets.m47_assurance import quarantine_conflicts, segment_distance
from geometry.ray_tracing import VoxelGrid, box_interval, trace_segment
from simulator.recorder import atomic_json, file_hash
from tools.m47_common import freeze_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m47_false_free_audit.json")
    p.add_argument("--artifacts", default="reports/m47_conflict_artifacts")
    p.add_argument(
        "--labels", default="/root/gpufree-data/hs-drive-data/processed/m47_quarantined"
    )
    a = p.parse_args()
    out = Path(a.artifacts)
    dest = Path(a.labels)
    if out.exists() or dest.exists() or Path(a.output).exists():
        raise FileExistsError("New paths required")
    out.mkdir()
    dest.mkdir(parents=True)
    sources = freeze_sources(
        out / "source",
        [
            __file__,
            "datasets/m47_assurance.py",
            "datasets/occupancy_quality.py",
            "datasets/occupancy_labels.py",
        ],
    )
    raw = Path("/root/gpufree-data/hs-drive-data/raw/m46_occupancy")
    sparse = json.loads(Path("reports/m46_occupancy_validation.json").read_text())[
        "records"
    ]
    dense = json.loads(Path("reports/m46_occupancy_dense.json").read_text())["records"]
    g = VoxelGrid()
    records = []
    label_records = []
    for sr, dr in zip(sparse, dense):
        assert (sr["scene"], sr["frame_id"]) == (dr["scene"], dr["frame_id"])
        root = raw / sr["scene"]
        man = json.loads((root / "manifest.json").read_text())
        frame = json.loads((root / "frames" / f"{sr['frame_id']:08d}.json").read_text())
        reference = json.loads(
            (root / "frames" / f"{man['counterfactual_frame']:08d}.json").read_text()
        )
        calibration = json.loads((root / "calibration.json").read_text())["cameras"]
        corners = np.array(
            next(x for x in frame["actors"] if x["is_ego"])["corners_hs_ego_m"]
        )
        box = (corners.min(0) - 0.05, corners.max(0) + 0.05)
        with np.load(dr["path"]) as z:
            labels = {k: z[k] for k in z.files}
        with np.load(sr["path"]) as z:
            sparse_labels = {k: z[k] for k in z.files}
        assert file_hash(dr["path"]) == dr["sha256"]
        sparse_rays = []
        held = {}
        for camera, c in calibration.items():

            def arrays(f):
                row = f["cameras"][camera]
                for k in ("depth", "instance"):
                    assert file_hash(root / row[k]) == row[k + "_sha256"]
                with np.load(root / row["depth"]) as x:
                    d = x["depth_m"]
                with np.load(root / row["instance"]) as x:
                    s = x["semantic"]
                    i = x["instance"]
                return d, s, i

            d, s, ins = arrays(frame)
            rd, rs, ri = arrays(reference)
            rays = visibility_rays(
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
                stride=1,
            )
            uv = rays["uv"]
            rank, inside = g.locate(rays["endpoints"])
            test = rays["reliable"] & inside & ~(uv % 2 == 0).all(-1)
            test &= ~labels["ego_mask"].ravel()[rank.clip(0)]
            conflict = test & (labels["occupancy_state"].ravel()[rank.clip(0)] == 1)

            def observation(index):
                u, v = uv[index]
                dep = float(d[v, u])
                patch = d[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2]
                return dict(
                    camera=camera,
                    uv=[int(u), int(v)],
                    depth_optical_z_m=dep,
                    semantic=int(s[v, u]),
                    surface_class=int(rays["semantics"][index]),
                    instance=int(ins[v, u]),
                    origin=rays["origins"][index].tolist(),
                    endpoint=rays["endpoints"][index].tolist(),
                    patch_depth_spread_m=float(patch.max() - patch.min()),
                    box_crossing=bool(rays["body_crossing"][index]),
                    self_pixel=bool(rays["body_pixel"][index]),
                    visibility_certified=bool(rays["visibility_certified"][index]),
                )

            for index in np.flatnonzero(conflict):
                held.setdefault(int(rank[index]), []).append(observation(index))
            for index in np.flatnonzero(rays["reliable"] & (uv % 16 == 0).all(-1)):
                sparse_rays.append(observation(index))
        assert len(held) == dr["audit"]["heldout_surface_marked_free"], (
            sr["scene"],
            len(held),
            dr["audit"],
        )
        free = {rank: [] for rank in held}
        surface = sparse_labels["occupancy_state"].ravel() == 2
        # Ego-masked surfaces excluded from saved state: original implementation
        # inserted them before carving. Reconstruct all reliable sparse endpoints.
        for r in sparse_rays:
            rank, inside = g.locate(np.array(r["endpoint"]))
            if inside:
                surface[rank] = True
        for ray in sparse_rays:
            o = np.array(ray["origin"])
            e = np.array(ray["endpoint"])
            ranks = trace_segment(o, e, g, end_margin=0.1)
            end_rank, end_inside = g.locate(e)
            for rank in ranks:
                if (end_inside and rank == end_rank) or surface[rank]:
                    break
                if int(rank) not in free:
                    continue
                z, x, y = np.unravel_index(rank, g.shape)
                lo = np.array([g.edges[0][x], g.edges[1][y], g.edges[2][z]])
                hi = np.array([g.edges[0][x + 1], g.edges[1][y + 1], g.edges[2][z + 1]])
                enter, leave = box_interval(o, e - o, lo, hi)
                enter = max(0, enter)
                leave = min(1 - 0.1 / np.linalg.norm(e - o), leave)
                free[int(rank)].append(
                    dict(
                        **ray,
                        entry=(o + enter * (e - o)).tolist(),
                        exit=(o + leave * (e - o)).tolist(),
                    )
                )
        for rank, observations in held.items():
            z, x, y = np.unravel_index(rank, g.shape)
            lo = np.array([g.edges[0][x], g.edges[1][y], g.edges[2][z]])
            hi = np.array([g.edges[0][x + 1], g.edges[1][y + 1], g.edges[2][z + 1]])
            pts = np.array([o["endpoint"] for o in observations])
            raylist = free[rank]
            assert raylist, ("Missing carving provenance", sr["scene"], rank)
            sep = min(
                float(
                    segment_distance(
                        pts, np.array(r["entry"]), np.array(r["exit"])
                    ).min()
                )
                for r in raylist
            )
            classes = sorted(set(o["surface_class"] for o in observations))
            cameras = sorted(set(o["camera"] for o in observations + raylist))
            plane = None
            if len(pts) >= 3:
                center = pts.mean(0)
                _, sv, vh = np.linalg.svd(pts - center, full_matrices=False)
                plane = dict(
                    normal=vh[-1].tolist(),
                    centroid=center.tolist(),
                    rms_residual_m=float(
                        np.sqrt(np.mean(((pts - center) @ vh[-1]) ** 2))
                    ),
                    point_count=len(pts),
                    noncollinear=bool(sv[1] > 0.01),
                )
            record = dict(
                id=f"{sr['scene']}:{sr['frame_id']}:{rank}",
                scene=sr["scene"],
                frame_id=sr["frame_id"],
                rank=int(rank),
                zxy=[int(z), int(x), int(y)],
                bounds_xyz_m=[lo.tolist(), hi.tolist()],
                free_rays=raylist,
                heldout_surface=observations,
                minimum_point_to_free_segment_m=sep,
                surface_classes=classes,
                local_plane=plane,
                category=(
                    "ground"
                    if classes == [1]
                    else "obstacle" if set(classes) <= {2, 3} else "mixed"
                ),
                same_physical_surface="unresolved: instance equality does not identify one mesh face",
                semantic_instance_agreement=any(
                    r["instance"] == o["instance"] and r["semantic"] == o["semantic"]
                    for r in raylist
                    for o in observations
                ),
                ego_mask=False,
                ego_box_min_distance_m=float(
                    np.linalg.norm(np.maximum(0, np.maximum(box[0] - hi, lo - box[1])))
                ),
                camera_overlap=len(cameras) > 1,
                cameras=cameras,
                min_height_boundary_distance_m=float(
                    np.minimum(pts[:, 2] - lo[2], hi[2] - pts[:, 2]).min()
                ),
                edge_candidate=any(
                    o["patch_depth_spread_m"] > 0.25 for o in observations
                ),
                classification="unresolved_surface_vs_ray_evidence",
                confidence="measured provenance; physical safety unproven",
                action="quarantine exactly this Free cell to Unknown; development repair, not independent success",
            )
            records.append(record)
        repaired = quarantine_conflicts(labels, list(held))
        folder = dest / sr["scene"]
        folder.mkdir(exist_ok=True)
        path = folder / f"{sr['frame_id']:08d}.npz"
        np.savez_compressed(path, **repaired)
        label_records.append(
            dict(
                scene=sr["scene"],
                frame_id=sr["frame_id"],
                path=str(path),
                sha256=file_hash(path),
                source_sha256=dr["sha256"],
                quarantined_cells=len(held),
                old_free=int((labels["occupancy_state"] == 1).sum()),
                coverage_loss_fraction=len(held) / labels["occupancy_state"].size,
            )
        )
        print(sr["scene"], sr["frame_id"], len(held), flush=True)
    for r in records:
        r["neighbor_frames_same_cell"] = [
            q["frame_id"]
            for q in records
            if q["scene"] == r["scene"]
            and q["rank"] == r["rank"]
            and q["frame_id"] != r["frame_id"]
        ]
    assert len(records) == 104
    atomic_json(
        a.output,
        dict(
            kind="offline_reconstruction_of_actual_carla_recordings",
            records=records,
            count=len(records),
            unique_scene_cells=len(set((r["scene"], r["rank"]) for r in records)),
            category_counts=dict(Counter(r["category"] for r in records)),
            all_carving_provenance_found=True,
            labels=label_records,
            sources=sources,
            definitions={
                "ray_free": "At least one observed segment traverses part of cell",
                "whole_cell_free": "Not established by a finite set of rays",
            },
            independent_ground_truth=False,
            quality_gate_closed=False,
        ),
    )


if __name__ == "__main__":
    main()
