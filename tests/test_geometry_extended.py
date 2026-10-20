import numpy as np
import pytest
from geometry.reference import (intrinsics_from_hfov, camera_mount_to_hs_from_optical,
                                transform_points, project_optical_points)
from geometry.coordinates import carla_pose_to_hs, carla_angular_velocity_to_hs_rad
from geometry.carla_depth import decode_rgb_bgra, valid_depth
from geometry.oracle_bev import depth_to_ego, plane_depth_comparison
from datasets.schema import validate_learned_inputs


def test_rgb_order_and_depth_mask():
    np.testing.assert_array_equal(decode_rgb_bgra(bytes([1,2,3,255]),1,1), [[[3,2,1]]])
    np.testing.assert_array_equal(valid_depth(np.array([np.nan,0,1.5,79.9,80,1000])),
                                  [False,False,True,True,False,False])


def test_reflected_world_pose_consistent():
    carla_T = np.eye(4)
    carla_T[:3,3] = [10,2,3]
    hs = carla_pose_to_hs(carla_T)
    np.testing.assert_allclose(hs[:3,3],[10,-2,3])
    assert np.linalg.det(hs[:3,:3]) == pytest.approx(1)
    np.testing.assert_allclose(carla_angular_velocity_to_hs_rad([0,0,90]),[0,0,-np.pi/2])


def test_six_mounts_against_carla_math():
    carla = pytest.importorskip('carla')
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path('configs/cameras.yaml').read_text())
    for name, camera in cfg['cameras'].items():
        mount = carla.Transform(carla.Location(*camera['xyz_m']),
            carla.Rotation(yaw=camera['yaw_deg'],pitch=camera['pitch_deg']))
        T = camera_mount_to_hs_from_optical(mount.get_matrix())
        p = transform_points([[0,0,10]],T)[0]
        actual = mount.transform(carla.Location(x=10))
        np.testing.assert_allclose(p,[actual.x,-actual.y,actual.z],atol=2e-6)
        assert np.linalg.det(T[:3,:3]) == pytest.approx(1,abs=1e-6)


def test_synthetic_plane_distinguishes_z_and_range():
    K = intrinsics_from_hfov(64,38,100)
    depth = np.full((38,64),10.)
    result = plane_depth_comparison(depth,K,[0,0,1],10,np.ones_like(depth,dtype=bool))
    assert result['optical_z'] < 1e-10
    assert result['ray_range'] > 1
    points = depth_to_ego(depth,K,camera_mount_to_hs_from_optical(np.eye(4)),
                          depth_convention='optical_z',stride=1)
    np.testing.assert_allclose(points[:,0],10)


def test_gt_allowlist_rejects_depth():
    validate_learned_inputs({'images': np.zeros((6,3,4,4))})
    with pytest.raises(ValueError,match='Forbidden'):
        validate_learned_inputs({'images': None, 'depth': None})


@pytest.mark.parametrize('kwargs', [dict(resolution_m=.7),dict(z_edges_m=[0,2,1]),dict(x_range_m=[0,0])])
def test_bad_grid_config_rejected(kwargs):
    from geometry.reference import points_to_height_bev
    with pytest.raises(ValueError):
        points_to_height_bev([[1,0,0]],**kwargs)


def test_allowlist_rejects_nested_supervision_under_ego_state():
    import pytest
    with pytest.raises(TypeError):
        validate_learned_inputs({'images':np.zeros((6,3,4,4)), 'ego_state':{'depth':np.zeros(4)}})
    with pytest.raises(TypeError):
        validate_learned_inputs({'ego_state':np.array([{'actor_gt':1}],dtype=object)})
