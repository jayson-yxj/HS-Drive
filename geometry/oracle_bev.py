"""Privileged visible-surface point counts. Empty cells are NOT free space."""
import numpy as np
from .reference import unproject_pixels_to_optical, transform_points, points_to_height_bev
from .carla_depth import valid_depth


def depth_to_ego(depth, K, cam_to_ego, *, depth_convention, stride=1,
                 min_m=1.5, max_m=80.0):
    if stride < 1:
        raise ValueError('stride must be positive')
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    d = depth[::stride, ::stride]
    mask = valid_depth(d, min_m, max_m)
    optical = unproject_pixels_to_optical(u[mask], v[mask], d[mask], K, depth_convention)
    return transform_points(optical, cam_to_ego)


def height_bev(points, config):
    return points_to_height_bev(points, **{k: config[k] for k in
        ('x_range_m', 'y_range_m', 'z_edges_m', 'resolution_m')})


def plane_depth_comparison(depth, K, normal_opt, offset_m, mask):
    """Known plane n.p=offset: compare both depth hypotheses without fitting scale.

    Mask must identify only the physical plane, including central AND edge pixels.
    A plane selected/fitted from the same depth is not independent ground truth.
    """
    v, u = np.indices(depth.shape)
    rays = unproject_pixels_to_optical(u, v, np.ones_like(depth), K).reshape(*depth.shape, 3)
    normal = np.asarray(normal_opt, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError('Invalid plane normal')
    denom = rays @ normal
    with np.errstate(divide='ignore', invalid='ignore'):
        z = offset_m / denom
    valid = mask & np.isfinite(depth) & (depth > 0) & (depth < 1000) & (z > 0) & np.isfinite(z)
    if np.count_nonzero(valid) < 20:
        raise ValueError('Need at least 20 valid known-plane pixels')
    expected = {'optical_z': z, 'ray_range': z * np.linalg.norm(rays, axis=-1)}
    return {name: float(np.mean(np.abs(depth[valid] - pred[valid])))
            for name, pred in expected.items()}
