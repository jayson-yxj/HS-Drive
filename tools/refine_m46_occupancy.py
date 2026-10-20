"""Dense-surface veto and held-out interleaved-pixel audit of M4.6 labels."""

import argparse, json
from pathlib import Path
import numpy as np
from datasets.occupancy_quality import visibility_rays, refine_surface_evidence
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="/root/gpufree-data/hs-drive-data/processed/m46_occupancy_refined",
    )
    p.add_argument("--report", default="reports/m46_occupancy_refined.json")
    p.add_argument("--surface-stride", type=int, default=8)
    p.add_argument("--source", default="reports/m46_occupancy_validation.json")
    p.add_argument(
        "--raw", default="/root/gpufree-data/hs-drive-data/raw/m46_occupancy"
    )
    a = p.parse_args()
    if a.surface_stride not in (2, 4, 8):
        raise ValueError("Surface stride must 2,4,8")
    out = Path(a.output)
    if out.exists() or Path(a.report).exists():
        raise FileExistsError("New paths required")
    out.mkdir(parents=True)
    source = json.loads(Path(a.source).read_text())
    root = Path(a.raw)
    g = VoxelGrid()
    records = []
    for row in source["records"]:
        folder = root / row["scene"]
        man = json.loads((folder / "manifest.json").read_text())
        cal = json.loads((folder / "calibration.json").read_text())["cameras"]
        frame_path = folder / "frames" / f"{row['frame_id']:08d}.json"
        f = json.loads(frame_path.read_text())
        ref = json.loads(
            (folder / "frames" / f"{man['counterfactual_frame']:08d}.json").read_text()
        )
        box = np.array(next(o for o in f["actors"] if o["is_ego"])["corners_hs_ego_m"])
        box = (box.min(0) - 0.05, box.max(0) + 0.05)
        endpoints = []
        semantics = []
        holdout = []
        hold_sem = []
        for name, c in cal.items():

            def arr(fr):
                r = fr["cameras"][name]
                for k in ["depth", "instance"]:
                    assert file_hash(folder / r[k]) == r[k + "_sha256"]
                with np.load(folder / r["instance"]) as x:
                    s = x["semantic"]
                    i = x["instance"]
                with np.load(folder / r["depth"]) as x:
                    d = x["depth_m"]
                return d, s, i

            d, s, i = arr(f)
            rd, rs, ri = arr(ref)
            ray = visibility_rays(
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
                stride=a.surface_stride // 2,
            )
            # Generated and held-out surface pixels interleave at half-stride.
            # The latter never construct the additional dense surface veto.
            train = (ray["uv"] % a.surface_stride == 0).all(-1) & ray["reliable"]
            test = ~(ray["uv"] % a.surface_stride == 0).all(-1) & ray["reliable"]
            endpoints.append(ray["endpoints"][train])
            semantics.append(ray["semantics"][train])
            holdout.append(ray["endpoints"][test])
            hold_sem.append(ray["semantics"][test])
        with np.load(row["path"]) as data:
            old = {k: data[k] for k in data.files}
        labels, refstats = refine_surface_evidence(
            old, np.concatenate(endpoints), np.concatenate(semantics), g
        )
        pts = np.concatenate(holdout)
        hs = np.concatenate(hold_sem)
        ranks, inside = g.locate(pts)
        ranks = ranks[inside]
        hs = hs[inside]
        nonself = ~labels["ego_mask"].ravel()[ranks]
        ranks = ranks[nonself]
        hs = hs[nonself]
        unique = np.unique(ranks)
        st = labels["occupancy_state"].ravel()
        sem = labels["surface_semantic"].ravel()
        observed = st[ranks] == 2
        semantic_valid = labels["surface_semantic_valid"].ravel()[ranks] & (hs > 0)
        dir = out / row["scene"]
        dir.mkdir(exist_ok=True)
        path = dir / f"{row['frame_id']:08d}.npz"
        np.savez_compressed(path, **labels)
        record = dict(
            scene=row["scene"],
            category=row["category"],
            frame_id=row["frame_id"],
            path=str(path),
            sha256=file_hash(path),
            source_label_sha256=row["label_sha256"],
            refinement=refstats,
            counts={
                k: int(mask.sum())
                for k, mask in dict(
                    free=labels["occupancy_state"] == 1,
                    surface=labels["occupancy_state"] == 2,
                    unknown=labels["occupancy_state"] == 0,
                    ground=labels["surface_semantic"] == 1,
                    obstacle=labels["obstacle_mask"],
                ).items()
            },
            audit=dict(
                heldout_surface_voxels=len(unique),
                heldout_surface_marked_free=int((st[unique] == 1).sum()),
                heldout_surface_marked_surface=int((st[unique] == 2).sum()),
                heldout_surface_unknown=int((st[unique] == 0).sum()),
                heldout_semantic_evaluated=int(semantic_valid.sum()),
                heldout_semantic_mismatch=int(
                    (sem[ranks][semantic_valid] != hs[semantic_valid]).sum()
                ),
            ),
            ground_by_layer=[
                int((labels["surface_semantic"][z] == 1).sum()) for z in range(4)
            ],
        )
        records.append(record)
        print(row["scene"], row["frame_id"], record["audit"], flush=True)
    atomic_json(
        a.report,
        dict(
            schema_version="hsd-m46-refined-occupancy-v1",
            records=records,
            protocol=f"Additional reliable surface endpoints stride{a.surface_stride}; 1 XY-cell same-height unknown margin around surface. Heldout stride{a.surface_stride//2} interleaved pixels unused in generation. Original stride16 ray carving unchanged.",
            source_sha256=file_hash(a.source),
            not_independent_sensor=True,
            limitations=[
                "Heldout pixels share camera renderer, so do not prove all 3D free space correct. Observed Free is ray evidence, not solid-volume emptiness.",
                "Dense refinement is a separate change from ego filtering comparison.",
                "Unknown boundary intentionally sacrifices coverage; no class manufactured at unseen interiors.",
            ],
        ),
    )
    atomic_json(
        out / "manifest.json", dict(report=a.report, sha256=file_hash(a.report))
    )


if __name__ == "__main__":
    main()
