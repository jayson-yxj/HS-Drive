"""Held-out rendered surface evaluation; privileged offline sidecar only.

This is disjoint pixel support, not independent physical 3D truth. Surface
negatives require reliable Ground; free-ray-only cells remain unlabelled.
"""

import numpy as np
from geometry.ray_tracing import VoxelGrid
from datasets.m5b_pairs import camera_rays


def heldout_surface(root, manifest, frame, camera_names, label):
    grid = VoxelGrid()
    size = int(np.prod(grid.shape))
    counts = np.zeros((3, size), np.int32)
    actors = []
    actor_counts = []
    for a in frame["actors"]:
        if a["is_ego"]:
            continue
        box = np.array(a["corners_hs_ego_m"])
        center = box.mean(0)
        if np.linalg.norm(center[:2]) >= 80:
            continue
        actors.append(
            dict(
                actor_id=a["actor_id"],
                type_id=a["type_id"],
                center_hs_ego=center.tolist(),
                distance_m=float(np.linalg.norm(center[:2])),
                aabb=[box.min(0).tolist(), box.max(0).tolist()],
            )
        )
        actor_counts.append(np.zeros(size, np.int32))
    for name in camera_names:
        r, _ = camera_rays(root, manifest, frame, name)
        u, v = r["uv"].T
        ids, inside = grid.locate(r["endpoints"])
        keep = r["reliable"] & inside & ((u + 2 * v) % 4 == 1)
        for k, classes in enumerate([[1], [2, 3], [0, 4]]):
            mask = keep & np.isin(r["semantics"], classes)
            np.add.at(counts[k], ids[mask], 1)
        for j, a in enumerate(actors):
            box = np.array(a["aabb"])
            candidate = (
                (r["endpoints"] >= box[0] - 0.05) & (r["endpoints"] <= box[1] + 0.05)
            ).all(-1)
            mask = keep & candidate & np.isin(r["semantics"], [2, 3])
            np.add.at(actor_counts[j], ids[mask], 1)
    counts = counts.reshape(3, *grid.shape)
    ground = counts[0] >= 3
    obstacle = counts[1] >= 3
    # Any contradictory semantic evidence leaves the classification unresolved.
    conflict = (counts > 0).sum(0) > 1
    valid = (ground | obstacle) & ~conflict & ~label["ego_mask"]
    am = np.array(
        [(c >= 3).reshape(grid.shape) & valid & obstacle for c in actor_counts],
        dtype=bool,
    ).reshape(-1, *grid.shape)
    return (
        dict(
            target=obstacle,
            valid=valid,
            ground=ground,
            mixed=label["mixed_cell_mask"],
            conflict=conflict,
            counts=counts,
            actor_masks=am,
        ),
        actors,
    )
