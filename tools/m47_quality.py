"""Supplemental collision-geometry precision/recall with explicit unknown coverage."""

import argparse, json
from pathlib import Path
import numpy as np
from datasets.occupancy_quality import (
    visibility_rays,
    build_visibility_observed,
    refine_surface_evidence,
)
from geometry.ray_tracing import VoxelGrid, trace_segment
from simulator.recorder import atomic_json, file_hash
from datasets.m47_assurance import quarantine_conflicts


def evaluate_record(root, man, fid, out):
    cal = json.loads((root / "calibration.json").read_text())["cameras"]
    frame = json.loads((root / "frames" / f"{fid:08d}.json").read_text())
    ref = json.loads(
        (root / "frames" / f"{man['counterfactual_frame']:08d}.json").read_text()
    )
    geometry = json.loads((root / f"m47_geometry_{fid:08d}.json").read_text())
    box = np.array(next(x for x in frame["actors"] if x["is_ego"])["corners_hs_ego_m"])
    box = (box.min(0) - 0.05, box.max(0) + 0.05)
    g = VoxelGrid()
    parts = []
    dense = []
    sem = []
    native = []
    held = []
    for name, c in cal.items():

        def arr(f):
            r = f["cameras"][name]
            with np.load(root / r["depth"]) as z:
                d = z["depth_m"]
            with np.load(root / r["instance"]) as z:
                s = z["semantic"]
                i = z["instance"]
            return d, s, i

        d, s, i = arr(frame)
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
            stride=1,
        )
        selected = ray["reliable"] & (ray["uv"] % 16 == 0).all(-1)
        parts.append(
            {
                k: ray[k][selected]
                for k in ["origins", "endpoints", "semantics", "reliable"]
            }
        )
        selected = ray["reliable"] & (ray["uv"] % 2 == 0).all(-1)
        dense.append(ray["endpoints"][selected])
        sem.append(ray["semantics"][selected])
        selected = ray["reliable"] & ~(ray["uv"] % 2 == 0).all(-1)
        held.append(ray["endpoints"][selected])
        native.append((name, d, s))
    sparse = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    labels, _ = build_visibility_observed(sparse, box, g)
    labels, _ = refine_surface_evidence(
        labels, np.concatenate(dense), np.concatenate(sem), g
    )
    # Evaluate frozen generator BEFORE using any heldout query to quarantine.
    state = labels["occupancy_state"].ravel()
    ranks, inside = g.locate(np.concatenate(held))
    heldr = np.unique(ranks[inside])
    heldr = heldr[~labels["ego_mask"].ravel()[heldr]]
    endpoints = []
    classes = []
    collisionfree = set()
    mismatches = []
    for q in geometry["ray_queries"] + geometry["vertical_queries"]:
        for h in q["hits"]:
            point = np.asarray(h["point_hs"]).reshape(3)
            rank, inside = g.locate(point)
            if inside and not labels["ego_mask"].ravel()[rank]:
                endpoints.append(point)
                classes.append(man["semantic_groups"].get(str(h["semantic"]), 0))
        if "origin" in q and q["hits"]:
            o = np.array(q["origin"])
            hits = [np.asarray(h["point_hs"]).reshape(3) for h in q["hits"]]
            e = min(hits, key=lambda h: np.linalg.norm(h - o))
            for rank in trace_segment(o, e, g, end_margin=0.15):
                if not labels["ego_mask"].ravel()[rank]:
                    collisionfree.add(int(rank))
    endpoints = np.array(endpoints).reshape(-1, 3)
    classes = np.array(classes)
    rank, inside = g.locate(endpoints)
    rank = rank[inside]
    classes = classes[inside]
    unique = np.unique(rank)
    free = np.array(sorted(collisionfree), int)
    gt_surface = set(unique.tolist())
    free = np.array([r for r in free if r not in gt_surface], int)
    evaluated = np.union1d(unique, free)
    pred_free = evaluated[state[evaluated] == 1]
    conflicts = np.intersect1d(unique, pred_free)

    def metrics(mask):
        rr = rank[mask]
        cc = classes[mask]
        s = state[rr]
        return dict(
            surface_points=len(rr),
            surface_recall=float((s == 2).mean()) if len(rr) else None,
            conflict_rate=float((s == 1).mean()) if len(rr) else None,
            ground_recall=float((s[cc == 1] == 2).mean()) if (cc == 1).any() else None,
            obstacle_recall=(
                float((s[np.isin(cc, [2, 3])] == 2).mean())
                if np.isin(cc, [2, 3]).any()
                else None
            ),
        )

    distance = np.linalg.norm(endpoints[inside, :2], axis=1)
    zxy = np.array(np.unravel_index(rank, g.shape)).T
    byrange = {
        f"{lo}-{hi}": metrics((distance >= lo) & (distance < hi))
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]
    }
    bylayer = {str(z): metrics(zxy[:, 0] == z) for z in range(4)}
    ego_distance = np.linalg.norm(
        np.maximum(
            0, np.maximum(box[0] - endpoints[inside], endpoints[inside] - box[1])
        ),
        axis=1,
    )
    local = {
        "near_body_lt2m": metrics(ego_distance < 2),
        "ground": metrics(classes == 1),
        "obstacle_boundary": metrics(np.isin(classes, [2, 3])),
    }
    # Selective collision-evidence quarantine stored separately; never use its
    # in-sample disappearance of conflicts as independent performance.
    allconf = np.union1d(conflicts, heldr[state[heldr] == 1])
    repaired = quarantine_conflicts(labels, allconf)
    path = out / f"{fid:08d}.npz"
    np.savez_compressed(path, **repaired)
    return dict(
        scene=man["scene"],
        map=man["map"],
        vehicle=man.get("vehicle"),
        role=man["role"],
        frame_id=fid,
        metrics=metrics(np.ones(len(rank), bool)),
        by_distance=byrange,
        by_layer=bylayer,
        local=local,
        free_precision_collision_evidence=(
            1 - len(conflicts) / len(pred_free) if len(pred_free) else None
        ),
        free_evaluated_cells=len(pred_free),
        free_conflicts=len(conflicts),
        unknown_fraction=float((state == 0).mean()),
        valid_coverage=float((state != 0).mean()),
        heldout_render_surface_conflicts=int((state[heldr] == 1).sum()),
        heldout_render_surface_cells=len(heldr),
        collision_surface_cells=len(unique),
        quarantine_cells=len(allconf),
        quarantine_coverage_loss=len(allconf) / state.size,
        repaired_path=str(path),
        repaired_sha256=file_hash(path),
        geometry_sha256=file_hash(root / f"m47_geometry_{fid:08d}.json"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--output", default="reports/m47_occupancy_quality.json")
    p.add_argument(
        "--labels",
        default="/root/gpufree-data/hs-drive-data/processed/m47_reserved_labels",
    )
    a = p.parse_args()
    out = Path(a.labels)
    if out.exists() or Path(a.output).exists():
        raise FileExistsError("New output")
    out.mkdir(parents=True)
    records = []
    for path in a.raw:
        root = Path(path)
        campaign = json.loads((root / "manifest.json").read_text())
        for man in campaign["scenes"]:
            dest = out / man["scene"]
            dest.mkdir()
            for fid in man["frames"]:
                records.append(evaluate_record(root / man["scene"], man, fid, dest))
    atomic_json(
        a.output,
        dict(
            kind="offline_evaluation_actual_carla_collision_and_render_observations",
            records=records,
            generator_sha256=file_hash("datasets/occupancy_quality.py"),
            protocol="Frozen M46 sparse16+dense2 generator. Collision queries seeded independently of label pixel lattice. Metrics BEFORE per-record quarantine.",
            limitations=[
                "Collision geometry is supplemental representation, not exact render mesh or independent physical truth.",
                "Free precision conditional on probed cells, not full-volume emptiness; surface recall counts only queried surfaces.",
                "Ground/Obstacle recall evaluates geometric Surface detection; semantic validity retained separately.",
                "Unknown fraction includes unobserved cells; no claim finite probes certify unseen interior.",
                "Heldout scenes evaluated once after freezing generator; using their quarantined labels later would consume heldout status.",
            ],
            formal_training_ready=False,
        ),
    )


if __name__ == "__main__":
    main()
