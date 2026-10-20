import pytest

torch = pytest.importorskip("torch")
from geometry.temporal_visibility import align_visible_history


def test_dynamic_old_position_removed_occluded_current_stays_invalid():
    bev = torch.ones(1, 2, 160, 120)
    pose = torch.eye(4)[None]
    prior = torch.ones(1, 1, 160, 120, dtype=torch.bool)
    observed = prior.clone()
    observed[:, :, 60:70, 40:60] = False
    dynamic = torch.zeros_like(prior)
    dynamic[:, :, 80:84, 50:55] = True
    current = torch.zeros_like(prior)
    current[:, :, 85:89, 50:55] = True
    aligned, mask, _ = align_visible_history(
        bev,
        pose,
        pose,
        prior,
        observed,
        previous_dynamic=dynamic,
        current_dynamic=current,
    )
    assert not mask[dynamic | current | ~observed].any()
    assert not aligned[~mask.expand_as(aligned)].any()
    assert mask[:, :, 20:30, 20:30].all()


def test_warp_outside_stays_invalid_even_with_current_support():
    bev = torch.ones(1, 1, 160, 120)
    p = torch.eye(4)[None]
    q = p.clone()
    q[:, 0, 3] = 1
    mask = torch.ones_like(bev, dtype=torch.bool)
    aligned, valid, _ = align_visible_history(bev, p, q, mask, mask)
    assert not valid[:, :, -2:].any() and not aligned[:, :, -2:].any()
