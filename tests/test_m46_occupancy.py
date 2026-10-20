import numpy as np
from datasets.occupancy_quality import visibility_rays, build_visibility_observed
from datasets.occupancy_quality import refine_surface_evidence
from datasets.occupancy_labels import build_observed
from geometry.ray_tracing import VoxelGrid


def inputs():
    shape = (7, 7)
    k = np.array([[100.0, 0, 3], [0, 100.0, 3], [0, 0, 1]])
    # Optical Z points through a coarse box; the rendered surface is farther.
    dep = np.full(shape, 4.0)
    sem = np.ones(shape, np.uint8)
    ids = np.full(shape, 12, np.uint16)
    return dep, sem, ids, k, (np.array([-0.5, -0.5, 1]), np.array([0.5, 0.5, 2]))


def test_box_crossing_requires_visibility_certificate():
    d, s, i, k, box = inputs()
    args = (d, s, i, d.copy(), s.copy(), i.copy(), k, np.eye(4), box, {1: 1})
    r = visibility_rays(*args, stride=1)
    assert r["body_crossing"].all() and r["reliable"].all()
    bad = d.copy()
    bad[:] = 5
    r = visibility_rays(d, s, i, bad, s, i, k, np.eye(4), box, {1: 1}, stride=1)
    assert not r["reliable"].any()


def test_self_instance_and_corrupted_depth_never_carves_behind():
    d, s, i, k, box = inputs()
    d[:] = 1.7
    s[:] = 14
    i[:] = 99
    r = visibility_rays(
        d,
        s,
        i,
        np.full_like(d, 4),
        np.ones_like(s),
        np.full_like(i, 12),
        k,
        np.eye(4),
        box,
        {14: 3},
        stride=1,
    )
    assert r["body_pixel"].all() and not r["reliable"].any()
    labels, _ = build_visibility_observed(
        r, box, VoxelGrid((-2, 2), (-2, 2), (0, 1, 2, 3, 5), 0.5)
    )
    assert not labels["occupancy_valid"].any()  # entire behind-self volume unknown


def test_ego_cells_stay_unknown_after_recovered_rays():
    d, s, i, k, box = inputs()
    r = visibility_rays(d, s, i, d, s, i, k, np.eye(4), box, {1: 1}, stride=1)
    lab, _ = build_visibility_observed(
        r, box, VoxelGrid((-2, 2), (-2, 2), (0, 1, 2, 3, 5), 0.5)
    )
    assert lab["occupancy_valid"].any()
    assert not lab["occupancy_valid"][lab["ego_mask"]].any()
    assert not lab["obstacle_mask"].any()


def test_dense_surface_overrides_free_without_dilating_occupied():
    grid = VoxelGrid((0, 6), (-2, 2), (0, 1, 3), 1.0)
    lab, _ = build_observed(
        np.array([[0.5, 0.5, 1.5]]), np.array([[5.5, 0.5, 1.5]]), grid
    )
    refined, stats = refine_surface_evidence(
        lab, np.array([[2.5, 0.5, 1.5]]), np.array([1]), grid
    )
    assert refined["occupancy_state"][1, 2, 2] == 2
    assert refined["surface_semantic"][1, 2, 2] == 1
    assert refined["occupancy_state"][1, 1, 2] == 0  # uncertainty, not occupied
    assert not refined["obstacle_mask"][1, 2, 2]
    assert stats["sparse_free_overridden_by_dense_surface"] == 1
