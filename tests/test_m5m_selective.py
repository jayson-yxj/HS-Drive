import numpy as np
import pytest
from diagnostics.m5m_contact import fit_plane, reference_status, intersect
from models.m5m_selective import predict
from geometry.reference import intrinsics_from_hfov, camera_mount_to_hs_from_optical


def test_gold_requires_independent_anatomy_and_consistency():
    p = {"coefficients": [0, 0, 0]}
    assert reference_status(plane=p, visible_pixels=20) == "Candidate"
    assert (
        reference_status(
            plane=p,
            visible_pixels=20,
            anatomy_verified=True,
            collision_agrees=True,
            stable_pose=True,
            ambiguity=False,
        )
        == "Gold"
    )
    assert reference_status(plane=None, visible_pixels=20) == "Unknown"


def test_plane_and_degenerate_support():
    x, y = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
    p = fit_plane(np.c_[x.ravel(), y.ravel(), (0.1 * x + 0.2 * y - 0.25).ravel()])
    assert np.allclose(p["coefficients"], [0.1, 0.2, -0.25])
    assert fit_plane(np.zeros((20, 3))) is None


def test_unknown_candidate_preserved_and_gt_rejected():
    K = intrinsics_from_hfov(640, 384, 100)
    T = camera_mount_to_hs_from_optical(
        np.array([[1, 0, 0, 1.65], [0, 1, 0, 0], [0, 0, 1, 1.65], [0, 0, 0, 1.0]])
    )
    q = np.zeros((384, 640))
    q[200:230, 300:320] = 0.99
    out = predict(q, K, T, timestamp=3.0)
    assert (
        len(out) == 1
        and out[0]["status"] == "Unknown"
        and out[0]["class_hint"] == "foreground_unknown_class"
    )
    assert out[0]["xy_hypothesis"] is not None
    with pytest.raises(TypeError):
        predict(q, K, T, timestamp=3.0, actor_gt={})
    with pytest.raises(ValueError):
        predict(q * np.nan, K, T, timestamp=3.0)


def test_probability_mixture_preserves_distribution_and_endpoints():
    import torch
    from tools.m5m_residual_gating import mix

    p = torch.rand(1, 6, 48, 2, 3).softmax(2)
    q = torch.rand_like(p).softmax(2)
    g = torch.ones(1, 6, 2, 3)
    assert torch.equal(mix(p, q, g), q)
    assert torch.equal(mix(p, q, g * 0), p)
    assert torch.allclose(mix(p, q, g * 0.3).sum(2), g)
    with pytest.raises(ValueError):
        mix(p, q, g * 2)
