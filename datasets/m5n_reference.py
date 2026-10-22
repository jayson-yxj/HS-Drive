"""Evaluation/supervision-only simulator OBB center contract."""

import numpy as np
from geometry.reference import transform_points, CARLA_EGO_TO_HS_EGO as S


def actor_reference(actor, T_world_hs_from_ego, T_ego_from_opt, K):
    corners = np.asarray(actor["corners_hs_ego_m"], float)
    pose = np.asarray(T_world_hs_from_ego, float)
    A = np.asarray(actor["world_matrix"], float)
    T = np.asarray(T_ego_from_opt, float)
    K = np.asarray(K, float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("eight finite bbox vertices required")
    center = corners.mean(0)
    world_hs = transform_points(center, pose)[0]
    world_carla = world_hs * [1, -1, 1]
    local = transform_points(
        transform_points(corners, pose) * [1, -1, 1], np.linalg.inv(A)
    )
    local_center = local.mean(0)
    edges = np.stack(
        [local[4] - local[0], local[2] - local[0], local[1] - local[0]], axis=1
    )
    dims = np.linalg.norm(edges, axis=0)
    if (dims <= 0).any():
        raise ValueError("zero box dimension")
    R = edges / dims
    orth = float(np.max(abs(R.T @ R - np.eye(3))))
    sym = float(np.max(abs(local.mean(0) - (0.5 * (local.min(0) + local.max(0))))))
    if orth > 0.01:
        raise ValueError("non-orthogonal saved bbox edges")
    expected = np.array(
        [local[0] + edges @ np.array([i // 4, (i // 2) % 2, i % 2]) for i in range(8)]
    )
    residual = float(np.max(abs(expected - local)))
    if residual > 0.005:
        raise ValueError("invalid vertex ordering/box geometry")
    origin_ego = transform_points(A[:3, 3] * [1, -1, 1], np.linalg.inv(pose))[0]
    opt = transform_points(center, np.linalg.inv(T))[0]
    uv = (K @ opt)[:2] / opt[2] if opt[2] > 0 else None
    reconstructed = transform_points(local_center, A)[0]
    closure = float(np.max(abs(reconstructed - world_carla)))
    return dict(
        box_center_world_carla=world_carla.tolist(),
        box_center_world_hs=world_hs.tolist(),
        box_center_ego=center.tolist(),
        box_center_camera=opt.tolist(),
        optical_z=float(opt[2]),
        euclidean_camera_m=float(np.linalg.norm(opt)),
        euclidean_ego_m=float(np.linalg.norm(center)),
        ego_xy_range_m=float(np.linalg.norm(center[:2])),
        projected_center_uv=None if uv is None else uv.tolist(),
        actor_origin_ego=origin_ego.tolist(),
        bbox_local_center=local_center.tolist(),
        bbox_dimensions_xyz=dims.tolist(),
        bbox_local_rotation=R.tolist(),
        orthogonality_residual=orth,
        vertex_reconstruction_residual_m=residual,
        world_closure_residual_m=closure,
        actor_origin_center_offset_m=float(np.linalg.norm(center - origin_ego)),
        reference="simulator_obb_center_not_mesh_or_contact",
    )


def synchronized(frame, inp):
    if abs(frame["timestamp_s"] - inp["timestamp_s"]) > 1e-5:
        return False
    if not np.allclose(frame["T_world_from_ego"], inp["ego_pose"], atol=1e-5):
        return False
    for camera in frame["cameras"].values():
        for kind in ["rgb", "depth", "instance"]:
            if (
                kind + "_frame_id" in camera
                and camera[kind + "_frame_id"] != frame["frame_id"]
            ):
                return False
            if (
                kind + "_timestamp_s" in camera
                and abs(camera[kind + "_timestamp_s"] - frame["timestamp_s"]) > 1e-5
            ):
                return False
    return True
