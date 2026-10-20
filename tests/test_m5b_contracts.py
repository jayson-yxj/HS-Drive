import numpy as np
import pytest
from diagnostics.m5b_metrics import change_metrics


def test_direction_and_stable_false_changes():
    shape = (4, 160, 120)
    a = np.zeros(shape)
    b = a.copy()
    pair = {k: np.zeros(shape, bool) for k in ["appeared", "vanished", "stable"]}
    pair["appeared"][1, 50, 60] = True
    pair["vanished"][1, 60, 60] = True
    pair["stable"][1, 70, 60] = True
    a[1, 60, 60] = 0.8
    b[1, 50, 60] = 0.8
    b[1, 70, 60] = 0.3
    b[3, 100, 100] = 1  # unknown ignored
    r = change_metrics(a, b, pair, 0.1)
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 0
    assert r["iou"] == 2 / 3 and r["appearance_localization"]["nearest_xy_mae_m"] == 0


def test_pair_loss_unknown_no_gradient():
    torch = pytest.importorskip("torch")
    from training.m5b_change_loss import change_loss

    shape = (2, 4, 2, 2)
    out = dict(
        surface=torch.zeros(shape, requires_grad=True),
        semantic=torch.zeros((2, 3, 4, 2, 2), requires_grad=True),
    )
    pair = {
        k: torch.zeros(shape[1:], dtype=torch.bool)
        for k in ["appeared", "vanished", "stable"]
    }
    pair["appeared"][1, 0, 0] = True
    loss, _ = change_loss(out, pair)
    loss.backward()
    assert out["surface"].grad[1, 1, 0, 0] < 0 and out["surface"].grad[0, 1, 0, 0] > 0
    assert torch.count_nonzero(out["surface"].grad) == 2


def test_cached_path_matches_formal_rgb_and_context_gradients():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from pathlib import Path

    if not Path("/root/gpufree-data/hs-drive-data/processed/m5b_pilot_index").exists():
        pytest.skip("server pilot unavailable")
    from tools.m5a_learning import encoder_load, stack
    from datasets.m5a_training import VisualEvidenceDataset
    from models.m5b_cached_encoder import cache_rgb, cached_forward
    from training.runtime import prepare_cuda

    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    ds = VisualEvidenceDataset(
        "/root/gpufree-data/hs-drive-data/processed/m5b_pilot_index",
        "train",
        supervision=False,
    )
    inputs = stack([ds[0]], "inputs")
    with torch.no_grad():
        expected = net(**inputs)["bev_current"]
    cache = cache_rgb(net, inputs, amp=False)
    with torch.no_grad():
        actual = cached_forward(net, cache, amp=False)
    # FP32 verifies algebraic equivalence, avoiding BF16 rounding threshold amplification.
    assert torch.allclose(actual, expected, atol=0.002, rtol=0.002)
    for p in net.context_head.parameters():
        p.requires_grad_(True)
    cached_forward(net, cache).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in net.context_head.parameters()
    )
    assert all(
        p.grad is None
        for n, p in net.named_parameters()
        if not n.startswith("context_head.")
    )


@pytest.mark.parametrize(
    "occluded,opposite,expected",
    [(False, False, 1), (True, False, 0), (False, True, 0)],
)
def test_witness_rejects_occlusion_and_other_coarse_obstacle(
    monkeypatch, occluded, opposite, expected
):
    import datasets.m5b_pairs as module

    shape = (4, 160, 120)
    label = {
        k: np.zeros(shape, bool)
        for k in [
            "ego_mask",
            "evidence_conflict_mask",
            "semantic_conflict_mask",
            "surface_valid",
            "surface_evidence",
            "obstacle_valid",
            "mixed_cell_mask",
        ]
    }
    label["surface_semantic"] = np.zeros((3, *shape), bool)
    la = {k: v.copy() for k, v in label.items()}
    lb = {k: v.copy() for k, v in label.items()}
    la["surface_semantic"][1, 2, 50, 60] = opposite
    uv = np.array([[1, 1], [1, 2], [2, 1]])
    pts = np.tile([5.1, 0.1, 1.2], (3, 1))
    r = dict(
        uv=uv,
        endpoints=pts,
        reliable=np.ones(3, bool),
        semantics=np.full(3, 2),
        stable_mask=np.ones((4, 4), bool),
        self_mask=np.zeros((4, 4), bool),
    )

    def fake(root, man, f, name):
        d = np.full(
            (4, 4),
            5.0 if f["frame_id"] == 2 else (4.0 if occluded else 10.0),
            np.float32,
        )
        i = np.full((4, 4), f["frame_id"])
        return r, (d, np.full((4, 4), 14), i)

    monkeypatch.setattr(module, "camera_rays", fake)
    a = dict(frame_id=1, cameras={"front": {}}, actors=[])
    b = dict(
        frame_id=2,
        cameras={"front": {}},
        actors=[dict(is_ego=False, corners_hs_ego_m=[[4, -1, 0], [6, 1, 2]])],
    )
    pair, _ = module.pair_witness(None, None, a, b, la, lb)
    assert pair["appeared"].sum() == expected
