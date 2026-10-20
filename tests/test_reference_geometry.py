"""Run in repo root: python -m pytest tests -q"""
import numpy as np
from geometry.reference import (
    camera_mount_to_hs_from_optical,
    decode_carla_depth_bgra,
    intrinsics_from_hfov,
    points_to_height_bev,
    project_optical_points,
    temporal_prev_to_current,
    transform_points,
    unproject_pixels_to_optical,
)


def test_projection_roundtrip_submillimeter():
    K = intrinsics_from_hfov(640, 384, 100.)
    pts = np.array([[0.2, -0.12, 4.0], [1.0, 0.5, 9.0], [-0.6, 0.2, 5.0]])
    uv = project_optical_points(pts, K)
    got = unproject_pixels_to_optical(uv[:, 0], uv[:, 1], pts[:, 2], K)
    np.testing.assert_allclose(got, pts, atol=1e-10)


def test_rays_carla_to_hs_left_right():
    T = camera_mount_to_hs_from_optical(np.eye(4))
    np.testing.assert_allclose(transform_points([[0,0,5]], T), [[5,0,0]])
    np.testing.assert_allclose(transform_points([[1,0,5]], T), [[5,-1,0]])
    np.testing.assert_allclose(transform_points([[-1,0,5]], T), [[5,1,0]])


def test_raw_depth_encoding_decode():
    depth = 10.1234
    enc = round(depth / 1000.0 * 16777215)
    r = enc & 255
    g = (enc >> 8) & 255
    b = (enc >> 16) & 255
    result = decode_carla_depth_bgra(bytes([b,g,r,255]), 1, 1)[0,0]
    assert abs(result - depth) < 1e-4


def test_ray_distance_is_not_optical_depth():
    K = intrinsics_from_hfov(640, 384, 100.)
    uv = np.array([[500., 100.]])
    pt = unproject_pixels_to_optical(uv[:,0], uv[:,1], np.array([12.]), K, 'ray_range')
    assert abs(np.linalg.norm(pt[0]) - 12.) < 1e-10
    assert pt[0,2] < 12.


def test_bev_index_and_boundary():
    p = np.array([[0,0,0], [-20,-30,-1], [60,0,0], [0,30,0]])
    bev = points_to_height_bev(p)
    assert bev.shape == (4,160,120)
    assert bev.sum() == 2
    assert bev[1,40,60] == 1  # z=0 inside [-0.5,1.0)
    assert bev[0,0,0] == 1


def test_forward_ego_movement():
    p = np.array([[10., 0., 0.]])
    T_prev = np.eye(4)
    T_curr = np.eye(4)
    T_curr[0,3] = 1.0  # ego moved forward by 1 m
    out = transform_points(p, temporal_prev_to_current(T_prev, T_curr))
    np.testing.assert_allclose(out, [[9., 0., 0.]])


def test_nan_points_are_ignored():
    bev = points_to_height_bev([[float("nan"), 1, 0], [3., 0., 0.]])
    assert bev.sum() == 1
