import numpy as np
from datasets import m5c_surface as module


def test_heldout_rays_only_and_ambiguous_or_ego_veto(monkeypatch):
    # Each cell gets three heldout pixels. Unknown semantics veto ground negatives.
    uv = np.array(
        [
            [1, 0],
            [5, 0],
            [9, 0],
            [13, 0],
            [17, 0],
            [21, 0],
            [25, 0],
            [29, 0],
            [33, 0],
            [37, 0],
            [41, 0],
            [45, 0],
            [0, 0],
        ]
    )
    pts = np.array(
        [[5.1, 0.1, 1.2]] * 3
        + [[6.1, 0.1, 1.2]] * 4
        + [[7.1, 0.1, 1.2]] * 3
        + [[8.1, 0.1, 1.2]] * 3
    )
    sem = np.array([2] * 3 + [1] * 3 + [0] + [1] * 3 + [2] * 3)
    r = dict(uv=uv, endpoints=pts, semantics=sem, reliable=np.ones(len(pts), bool))
    monkeypatch.setattr(module, "camera_rays", lambda *a: (r, None))
    blank = np.zeros((4, 160, 120), bool)
    ego = blank.copy()
    ego[2, 54, 60] = True
    data, actors = module.heldout_surface(
        None,
        None,
        dict(actors=[]),
        ["front"],
        dict(ego_mask=ego, mixed_cell_mask=blank),
    )
    assert data["valid"][2, 50, 60] and data["target"][2, 50, 60]
    assert not data["valid"][2, 52, 60]  # observed unknown + ground in same cell
    assert not data["valid"][2, 54, 60]  # ego
    assert not data["valid"][2, 56, 60]  # only two held-out support pixels
    assert data["valid"].sum() == 1 and not actors
