"""Privileged point-witness supervision. Never infer whole-cell safe free."""

import json
from pathlib import Path
import numpy as np
import cv2
from geometry.ray_tracing import VoxelGrid
from datasets.occupancy_quality import visibility_rays
from simulator.recorder import file_hash


def camera_rays(root, man, frame, name):
    cal = json.loads((root / "calibration.json").read_text())["cameras"][name]
    ref = json.loads(
        (
            root / "frames" / f"{man['references'][str(frame['frame_id'])]:08d}.json"
        ).read_text()
    )
    box = np.array(next(x for x in frame["actors"] if x["is_ego"])["corners_hs_ego_m"])
    box = (box.min(0) - 0.05, box.max(0) + 0.05)

    def read(f):
        r = f["cameras"][name]
        for k in ["depth", "instance"]:
            if file_hash(root / r[k]) != r[k + "_sha256"]:
                raise ValueError("GT checksum")
        with np.load(root / r["depth"]) as z:
            d = z["depth_m"]
        with np.load(root / r["instance"]) as z:
            s, i = z["semantic"], z["instance"]
        return d, s, i

    data = read(frame)
    r = visibility_rays(
        *data,
        *read(ref),
        cal["K"],
        cal["T_hs_ego_from_opt"],
        box,
        man["semantic_groups"],
        stride=1,
    )
    return r, data


def pair_witness(root, man, a, b, la, lb, cache=None):
    """Per observed target point: same ray's counterpart first hit > .5m behind.

    Actor boxes only gate candidate changed pixels. Visible renderer endpoint and
    semantic obstacle are mandatory; boxes aren't treated as occupied volume.
    A count>=3 supports a voxel witness. Opposite obstacle-surface cells veto an
    absence target, since different objects can share a coarse voxel.
    """
    cache = {} if cache is None else cache

    def get(frame, name):
        key = (frame["frame_id"], name)
        if key not in cache:
            cache[key] = camera_rays(root, man, frame, name)
        return cache[key]

    g = VoxelGrid()
    n = int(np.prod(g.shape))
    counts = [np.zeros(n, np.int32), np.zeros(n, np.int32)]
    details = []
    for name in a["cameras"]:
        ra, da = get(a, name)
        rb, db = get(b, name)
        for side, (f, r, other, rd, od, own, opp) in enumerate(
            [(a, ra, rb, da, db, la, lb), (b, rb, ra, db, da, lb, la)]
        ):
            validmap = other["stable_mask"] & ~other["self_mask"] & np.isfinite(od[0])
            far_min = cv2.erode(od[0], np.ones((3, 3), np.uint8))
            u, v = r["uv"].T
            d = rd[0][v, u]
            behind = far_min[v, u] > d + 0.5
            pts = r["endpoints"]
            actor = np.zeros(len(pts), bool)
            for obj in f["actors"]:
                if obj["is_ego"]:
                    continue
                corners = np.asarray(obj["corners_hs_ego_m"])
                actor |= np.all(
                    (pts >= corners.min(0) - 0.05) & (pts <= corners.max(0) + 0.05),
                    axis=-1,
                )
            keep = (
                r["reliable"]
                & validmap[v, u]
                & behind
                & (rd[2][v, u] != od[2][v, u])
                & np.isin(r["semantics"], [2, 3])
                & actor
            )
            rank, inside = g.locate(pts)
            keep &= inside
            ids = rank[keep]
            opposite_obstacle = opp["surface_semantic"][1].ravel()[ids]
            safe = (
                ~own["ego_mask"].ravel()[ids]
                & ~opp["ego_mask"].ravel()[ids]
                & ~own["evidence_conflict_mask"].ravel()[ids]
                & ~own["semantic_conflict_mask"].ravel()[ids]
                & ~opposite_obstacle
            )
            ids = ids[safe]
            np.add.at(counts[side], ids, 1)
            details.append(
                dict(camera=name, side=side, reliable_changed_pixels=int(len(ids)))
            )
    vanished = (counts[0] >= 3).reshape(g.shape)
    appeared = (counts[1] >= 3).reshape(g.shape)
    conflict = vanished & appeared
    vanished &= ~conflict
    appeared &= ~conflict
    # Stable background is conservative common valid evidence with identical tasks.
    stable = (
        la["surface_valid"]
        & lb["surface_valid"]
        & (la["surface_evidence"] == lb["surface_evidence"])
        & la["obstacle_valid"]
        & lb["obstacle_valid"]
        & (la["surface_semantic"][1] == lb["surface_semantic"][1])
        & ~(appeared | vanished)
    )
    stable &= ~la["mixed_cell_mask"] & ~lb["mixed_cell_mask"]
    return (
        dict(
            appeared=appeared,
            vanished=vanished,
            stable=stable,
            witness_conflict=conflict,
            witness_count_a=counts[0].reshape(g.shape),
            witness_count_b=counts[1].reshape(g.shape),
        ),
        details,
    )
