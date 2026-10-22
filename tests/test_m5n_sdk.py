"""CARLA SDK-only independent geometry construction, no server required."""

import pytest, numpy as np

carla = pytest.importorskip("carla")
from datasets.m5n_reference import actor_reference
from geometry.reference import transform_points
from geometry.coordinates import carla_pose_to_hs


def test_sdk_local_bbox_rotation_center_and_pose():
    bbox = carla.BoundingBox(carla.Location(0.7, 0.2, 0.4), carla.Vector3D(2, 1, 0.5))
    bbox.rotation = carla.Rotation(pitch=12, yaw=28, roll=3)
    actor = carla.Transform(carla.Location(12, 8, 2), carla.Rotation(yaw=-30, pitch=5))
    ego = carla.Transform(carla.Location(2, 3, 0.4), carla.Rotation(yaw=17))
    world = np.array([[v.x, v.y, v.z] for v in bbox.get_world_vertices(actor)])
    P = carla_pose_to_hs(ego.get_matrix())
    corners = transform_points(world * [1, -1, 1], np.linalg.inv(P))
    ref = actor_reference(
        dict(world_matrix=actor.get_matrix(), corners_hs_ego_m=corners),
        P,
        np.eye(4),
        np.eye(3),
    )
    assert np.allclose(ref["bbox_local_center"], [0.7, 0.2, 0.4], atol=1e-5)
    assert np.allclose(ref["bbox_dimensions_xyz"], [4, 2, 1], atol=1e-5)
    expected = actor.transform(bbox.location)
    assert np.allclose(
        ref["box_center_world_carla"], [expected.x, expected.y, expected.z], atol=1e-5
    )
