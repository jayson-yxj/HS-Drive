"""M4.6 visibility-certified self masking. Privileged offline supervision only."""

import numpy as np
import cv2
from geometry.ray_tracing import VoxelGrid, box_interval
from geometry.reference import unproject_pixels_to_optical, transform_points
from datasets.occupancy_labels import build_observed, validate_occupancy

SCHEMA = "hsd-m46-observed-occupancy-v2"


def visibility_rays(
    depth,
    semantic,
    instance,
    reference_depth,
    reference_semantic,
    reference_instance,
    K,
    cam_to_ego,
    ego_box,
    semantic_map,
    *,
    stride=16,
    edge_spread=0.5,
    stability_m=0.05,
):
    """Reference is same-camera ego-removed counterfactual, never a model input.

    Render instance identity is inferred geometrically, not equated to actor ID.
    A box-crossing ray needs invariant external first-hit evidence. Disappearing
    body pixels and a 2px silhouette margin always veto; uncertain rays Unknown.
    """
    shape = depth.shape
    if any(
        np.shape(x) != shape
        for x in [
            semantic,
            instance,
            reference_depth,
            reference_semantic,
            reference_instance,
        ]
    ):
        raise ValueError("Camera reference shapes differ")
    v, u = np.mgrid[: shape[0], : shape[1]]
    finite = np.isfinite(depth) & (depth > 0.05) & (depth < 80)
    points = transform_points(
        unproject_pixels_to_optical(u, v, np.where(finite, depth, 0), K, "optical_z"),
        cam_to_ego,
    ).reshape(*shape, 3)
    inside = np.all((points >= ego_box[0]) & (points <= ego_box[1]), axis=-1) & finite
    disappeared = (reference_depth > depth + 0.1) & (instance != reference_instance)
    # Any near-body endpoint is disallowed even if instance/semantic is missing.
    self_seed = inside & disappeared
    self_ids = np.unique(instance[self_seed])
    body = (
        np.isin(instance, self_ids) & (instance != 0)
        if len(self_ids)
        else np.zeros(shape, bool)
    )
    body |= inside
    body = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    stable = (
        (np.abs(depth - reference_depth) <= stability_m)
        & (semantic == reference_semantic)
        & (instance == reference_instance)
        & np.isfinite(reference_depth)
    )
    stable = cv2.erode(stable.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vv, uu = np.mgrid[0 : shape[0] : stride, 0 : shape[1] : stride]
    d = depth[vv, uu]
    patch = np.stack(
        [
            depth[np.clip(vv + dy, 0, shape[0] - 1), np.clip(uu + dx, 0, shape[1] - 1)]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    valid = (
        np.isfinite(patch).all(0)
        & (d >= 1.5)
        & (d < 80)
        & ((patch.max(0) - patch.min(0)) <= edge_spread)
    )
    end = points[vv, uu][valid]
    origin = np.broadcast_to(np.asarray(cam_to_ego)[:3, 3], end.shape).copy()
    delta = end - origin
    parallel = np.abs(delta) < 1e-12
    safe = np.where(parallel, 1.0, delta)
    a = (ego_box[0] - origin) / safe
    b = (ego_box[1] - origin) / safe
    low = np.where(parallel, -np.inf, np.minimum(a, b))
    high = np.where(parallel, np.inf, np.maximum(a, b))
    outside = np.any(parallel & ((origin < ego_box[0]) | (origin > ego_box[1])), axis=1)
    crosses = ~outside & (np.minimum(high.min(1), 1) > np.maximum(low.max(1), 0) + 1e-6)
    self_hit = body[vv, uu][valid]
    certified = stable[vv, uu][valid]
    reliable = ~self_hit & (~crosses | certified)
    sem = np.zeros(len(end), np.uint8)
    for label, code in semantic_map.items():
        sem[semantic[vv, uu][valid] == int(label)] = code
    return dict(
        origins=origin,
        endpoints=end,
        semantics=sem,
        reliable=reliable,
        body_crossing=crosses,
        body_pixel=self_hit,
        visibility_certified=certified,
        uv=np.stack([uu[valid], vv[valid]], -1),
        self_mask=body,
        stable_mask=stable,
        statistics=dict(
            valid_depth_rays=len(end),
            coarse_box_rejected=int(crosses.sum()),
            instance_body_rejected=int(self_hit.sum()),
            uncertified_box_rejected=int((crosses & ~certified & ~self_hit).sum()),
            accepted_rays=int(reliable.sum()),
            recovered_box_rays=int((crosses & reliable).sum()),
            self_instance_ids=self_ids.tolist(),
            visible_body_pixels=int(body.sum()),
        ),
    )


def build_visibility_observed(rays, ego_box, grid=None):
    """Replace the *ray* box test with certified visibility; keep ego voxel mask."""
    g = grid or VoxelGrid()
    labels, stats = build_observed(
        rays["origins"],
        rays["endpoints"],
        g,
        reliable=rays["reliable"],
        semantics=rays["semantics"],
    )
    z, x, y = np.indices(g.shape)
    edges = g.edges
    lower = np.stack((edges[0][x], edges[1][y], edges[2][z]), -1)
    upper = np.stack((edges[0][x + 1], edges[1][y + 1], edges[2][z + 1]), -1)
    ego = np.all(upper > ego_box[0], -1) & np.all(lower < ego_box[1], -1)
    labels["ego_mask"] = ego
    for key in ("occupancy_state", "surface_semantic"):
        labels[key][ego] = 0
    for key in (
        "occupancy_valid",
        "surface_semantic_valid",
        "obstacle_mask",
        "obstacle_valid",
    ):
        labels[key][ego] = False
    validate_occupancy(labels)
    s = labels["occupancy_state"]
    stats.update(
        free_voxels=int((s == 1).sum()),
        surface_voxels=int((s == 2).sum()),
        unknown_voxels=int((s == 0).sum()),
        ground_voxels=int((labels["surface_semantic"] == 1).sum()),
        obstacle_voxels=int(labels["obstacle_mask"].sum()),
        semantic_valid_voxels=int(labels["surface_semantic_valid"].sum()),
    )
    return labels, stats


def refine_surface_evidence(
    labels, endpoints, semantics, grid=None, *, xy_uncertainty_cells=1
):
    """Dense reliable first-hit support; never carve free using extra rays.

    Dense endpoint evidence overrides sparse Free. One XY cell around surfaces
    (same height) is Unknown unless itself observed surface. Fixed margin protects
    coarse discretization; it is not occupied dilation.
    """
    g = grid or VoxelGrid()
    result = {k: v.copy() for k, v in labels.items()}
    ranks, inside = g.locate(endpoints)
    ranks = ranks[inside]
    sem = np.asarray(semantics)[inside]
    cells = int(np.prod(g.shape))
    hits = np.bincount(ranks, minlength=cells).reshape(g.shape)
    support = (hits > 0) & ~result["ego_mask"]
    state = result["occupancy_state"]
    original_free = state == 1
    state[support] = 2
    result["surface_count"] += hits.astype(np.uint32)
    bit = np.zeros(cells, np.uint8)
    # Include sparse semantics, including existing ambiguity.
    for code in (1, 2, 3, 4):
        np.bitwise_or.at(bit, ranks[sem == code], np.uint8(1 << code))
        bit[result["surface_semantic"].ravel() == code] |= np.uint8(1 << code)
    semantic = np.zeros(cells, np.uint8)
    for code in (1, 2, 3, 4):
        semantic[bit == (1 << code)] = code
    conflict = ((bit != 0) & (semantic == 0)).reshape(g.shape) | result[
        "semantic_conflict_mask"
    ]
    semantic = semantic.reshape(g.shape)
    semantic[conflict | result["ego_mask"]] = 0
    boundary = np.zeros(g.shape, bool)
    kernel = np.ones((2 * xy_uncertainty_cells + 1,) * 2, np.uint8)
    for z in range(g.shape[0]):
        boundary[z] = cv2.dilate((state[z] == 2).astype(np.uint8), kernel).astype(
            bool
        ) & (state[z] != 2)
    uncertain = boundary & (state == 1)
    state[uncertain] = 0
    result.update(
        occupancy_state=state,
        occupancy_valid=state != 0,
        surface_semantic=semantic,
        surface_semantic_valid=(semantic > 0) & (state == 2),
        obstacle_mask=((semantic == 2) | (semantic == 3)) & (state == 2),
        obstacle_valid=((semantic >= 1) & (semantic <= 3)) & (state == 2),
        semantic_conflict_mask=conflict,
        surface_boundary_unknown=uncertain,
        dense_surface_support=support,
    )
    validate_occupancy(result)
    return result, dict(
        sparse_free_overridden_by_dense_surface=int((original_free & support).sum()),
        boundary_free_made_unknown=int(uncertain.sum()),
        dense_surface_voxels=int(support.sum()),
    )
