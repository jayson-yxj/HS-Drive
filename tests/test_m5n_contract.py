import numpy as np, torch, pytest
from datasets.m5n_reference import actor_reference, synchronized
from models.m5n_direct import DirectCenterHead, reconstruct, output_contract
from geometry.reference import transform_points


def test_rotated_local_box_center_not_actor_origin():
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    local = np.array(
        [[x, y, z] for x in [-2, 2] for y in [-1, 1] for z in [-0.5, 0.5]]
    ) @ R.T + [0.7, 0.2, 0.3]
    A = np.eye(4)
    A[:3, 3] = [12, 2, 0]
    world = transform_points(local, A)
    ego = world * [1, -1, 1]
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    r = actor_reference(
        dict(corners_hs_ego_m=ego, world_matrix=A), np.eye(4), T, np.eye(3)
    )
    assert np.allclose(r["bbox_local_center"], [0.7, 0.2, 0.3])
    assert np.allclose(r["bbox_dimensions_xyz"], [4, 2, 1])
    assert np.allclose(r["bbox_local_rotation"], R)
    assert (
        r["actor_origin_center_offset_m"] > 0.7
        and r["world_closure_residual_m"] < 1e-10
    )


def test_reconstruction_center_offset_is_not_roi_center():
    K = torch.eye(3)[None]
    T = torch.eye(4)[None]
    r = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    p = torch.tensor([[np.log(10), 0.25, -0.25]])
    xyz = reconstruct(p, r, K, T)
    assert torch.allclose(xyz, torch.tensor([[10.0, 0.0, 10.0]]), atol=1e-5)


def test_no_contact_upgrade_no_gt_inputs():
    o = output_contract(
        "x",
        [0, 0, 10, 10],
        [10.0, 0, 0],
        timestamp=1.0,
        proxy_error_m=10,
        threshold=0.9,
    )
    assert o["status"] == "DetectedNoLocation" and o["position_ego_xy"] is not None
    with pytest.raises(TypeError):
        output_contract("x", [], [], timestamp=1.0, proxy_error_m=1, actor_gt={})
    m = DirectCenterHead()
    assert m(torch.zeros(2, 600)).shape == (2, 3)
    with pytest.raises(ValueError):
        m(torch.full((1, 600), float("nan")))


def test_sync_rejects_stale_actor_frame():
    f = dict(
        timestamp_s=1.0,
        T_world_from_ego=np.eye(4).tolist(),
        frame_id=2,
        cameras={"front": {"rgb_frame_id": 1}},
    )
    inp = dict(timestamp_s=1.0, ego_pose=np.eye(4).tolist())
    assert not synchronized(f, inp)
