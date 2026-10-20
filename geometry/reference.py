"""Small, tested CPU reference geometry helpers for HS-Drive M0--M2.

Coordinate conventions
----------------------
Optical camera: +X right, +Y down, +Z forward.
CARLA ego/camera: +X forward, +Y right, +Z up.
HS ego: +X forward, +Y left, +Z up.

This is a *reference* tool. Depth interpretation (optical Z vs ray range)
must be experimentally verified for the chosen CARLA setup.
"""
from __future__ import annotations

import numpy as np

CARLA_EGO_TO_HS_EGO = np.diag([1.0, -1.0, 1.0, 1.0])
OPTICAL_TO_CARLA_CAM = np.array([
    [0.0,  0.0, 1.0, 0.0],
    [1.0,  0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0,  0.0, 0.0, 1.0],
], dtype=np.float64)


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Return K with integer pixel centers u=0..W-1 (not half-pixel indices)."""
    if not (width > 0 and height > 0 and 1.0 < hfov_deg < 179.0):
        raise ValueError('Invalid image dimensions / horizontal FOV')
    f = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return np.array([
        [f, 0.0, (width - 1.0) / 2.0],
        [0.0, f, (height - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def camera_mount_to_hs_from_optical(carla_ego_from_carla_camera: np.ndarray) -> np.ndarray:
    """Turn a CARLA camera mount transform into optical-to-HS-ego 4x4.

    For a CARLA Python Transform use np.asarray(mount.get_matrix()).
    IMPORTANT: `mount` is the *relative-to-ego* transform, not world pose.
    """
    m = np.asarray(carla_ego_from_carla_camera, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError('Expected a 4x4 relative camera mount transform')
    return CARLA_EGO_TO_HS_EGO @ m @ OPTICAL_TO_CARLA_CAM


def decode_carla_depth_bgra(raw_data: bytes, height: int, width: int) -> np.ndarray:
    """Decode CARLA 0.9.15 depth raw_data (BGRA uint8) to metric float32."""
    src = np.frombuffer(raw_data, dtype=np.uint8)
    if src.size != height * width * 4:
        raise ValueError('Raw BGRA depth size mismatch')
    bgra = src.reshape(height, width, 4)
    b = bgra[..., 0].astype(np.uint32)
    g = bgra[..., 1].astype(np.uint32)
    r = bgra[..., 2].astype(np.uint32)
    z = (r + (g << 8) + (b << 16)).astype(np.float32)
    return z * np.float32(1000.0 / 16777215.0)


def unproject_pixels_to_optical(
    u: np.ndarray, v: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
    depth_convention: str = 'optical_z'
) -> np.ndarray:
    """Unproject pixel coordinates into Nx3 optical-frame points.

    `depth_convention='optical_z'`: Z equals optical axis depth.
    `depth_convention='ray_range'`: distance along viewing ray.
    These are alternatives requiring an empirical CARLA depth semantics test.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = np.asarray(depth_m, dtype=np.float64)
    u, v, d = np.broadcast_arrays(u, v, d)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError('K must be 3x3')
    ray = np.stack([(u - K[0, 2]) / K[0, 0],
                    (v - K[1, 2]) / K[1, 1],
                    np.ones_like(d)], axis=-1)
    if depth_convention == 'optical_z':
        points = ray * d[..., None]
    elif depth_convention == 'ray_range':
        points = ray / np.linalg.norm(ray, axis=-1, keepdims=True) * d[..., None]
    else:
        raise ValueError('depth_convention must be optical_z or ray_range')
    return points.reshape(-1, 3)


def project_optical_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return Nx2 image coords for Nx3 optical points with positive Z."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if np.any(xyz[:, 2] <= 0):
        raise ValueError('Projection requires all points in front of camera')
    uvh = (np.asarray(K, dtype=np.float64) @ xyz.T).T
    return uvh[:, :2] / uvh[:, 2:3]


def transform_points(points: np.ndarray, T_out_from_in: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T_out_from_in, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError('Transform must be 4x4')
    xyz1 = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (T @ xyz1.T).T[:, :3]


def temporal_prev_to_current(T_world_from_prev: np.ndarray,
                             T_world_from_current: np.ndarray) -> np.ndarray:
    """Coordinate transform mapping previous ego points into current ego frame."""
    return np.linalg.inv(T_world_from_current) @ T_world_from_prev


def points_to_height_bev(
    points_hs_ego: np.ndarray,
    x_range_m=(-20.0, 60.0), y_range_m=(-30.0, 30.0),
    z_edges_m=(-2.0, -0.5, 1.0, 2.5, 4.0), resolution_m=0.5,
) -> np.ndarray:
    """Return [Zbin, Xgrid, Ygrid] visible-point *counts*, NOT occupancy GT."""
    if resolution_m <= 0:
        raise ValueError('resolution_m must be positive')
    pts = np.asarray(points_hs_ego, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    for bounds in (x_range_m, y_range_m):
        cells = (bounds[1] - bounds[0]) / resolution_m
        if not np.isfinite(cells) or cells <= 0 or not np.isclose(cells, round(cells)):
            raise ValueError('BEV ranges must contain a positive integer number of cells')
    if not np.isfinite(z_edges_m).all() or np.any(np.diff(z_edges_m) <= 0) or len(z_edges_m) < 2:
        raise ValueError('Height edges must be finite and strictly increasing')
    nx = round((x_range_m[1]-x_range_m[0])/resolution_m)
    ny = round((y_range_m[1]-y_range_m[0])/resolution_m)
    z_edges = np.asarray(z_edges_m, dtype=np.float64)
    out = np.zeros((len(z_edges)-1, nx, ny), dtype=np.uint32)
    i = np.floor((pts[:, 0] - x_range_m[0]) / resolution_m).astype(np.int64)
    j = np.floor((pts[:, 1] - y_range_m[0]) / resolution_m).astype(np.int64)
    k = np.searchsorted(z_edges, pts[:, 2], side='right') - 1
    valid = (i>=0) & (i<nx) & (j>=0) & (j<ny) & (k>=0) & (k<len(z_edges)-1)
    np.add.at(out, (k[valid], i[valid], j[valid]), 1)
    return out
