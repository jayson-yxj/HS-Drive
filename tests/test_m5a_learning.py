import numpy as np
import pytest

torch = pytest.importorskip("torch")
from diagnostics.m5a_metrics import binary_metrics
from models.m48_evidence_head import PositionPrior
from training.m48_evidence_loss import evidence_loss


def test_ap_ties_and_empty_classes():
    r = binary_metrics(
        np.array([0.5, 0.5]), np.array([1, 0]), np.ones(2, bool), threshold=0.4
    )
    assert r["pr_auc_ap"] == 0.5 and r["iou"] == 0.5
    assert (
        binary_metrics(np.ones(2), np.zeros(2), np.zeros(2, bool))["pr_auc_ap"] is None
    )


def test_prior_rgb_invariance():
    p = PositionPrior()
    a = p(torch.zeros(2, 1))
    b = p(torch.randn(2, 1))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_unknown_no_negative_and_nonexclusive_mixed():
    shape = (1, 4, 2, 2)
    labels = {
        k: torch.zeros(shape, dtype=torch.bool)
        for k in [
            "surface_evidence",
            "free_ray_evidence",
            "surface_valid",
            "free_evidence_valid",
            "surface_semantic_valid",
            "obstacle_valid",
        ]
    }
    labels["surface_semantic"] = torch.zeros((1, 3, 4, 2, 2), dtype=torch.bool)
    for k in [
        "surface_evidence",
        "free_ray_evidence",
        "surface_valid",
        "free_evidence_valid",
    ]:
        labels[k][0, 0, 0, 0] = True
    out = {
        k: torch.zeros(
            (1, 3, 4, 2, 2) if k == "semantic" else shape, requires_grad=True
        )
        for k in ["surface", "free", "semantic"]
    }
    loss, _ = evidence_loss(out, labels)
    loss.backward()
    assert out["surface"].grad[0, 0, 0, 0] < 0 and out["free"].grad[0, 0, 0, 0] < 0
    assert (out["surface"].grad != 0).sum() == 1 and not out["semantic"].grad.any()


def test_inference_does_not_load_sidecar():
    from pathlib import Path
    from datasets.m5a_training import VisualEvidenceDataset

    root = "/root/gpufree-data/hs-drive-data/processed/m5a_pilot_index"
    if not Path(root).exists():
        pytest.skip("server pilot data not installed")
    ds = VisualEvidenceDataset(root, "train", supervision=False)
    ds.rows[0]["label"] = "/nonexistent_gt"
    ds.rows[0]["label_sha256"] = "invalid"
    item = ds[0]
    assert item.targets is None
    assert set(item.inputs) == {
        "images",
        "intrinsics",
        "cam_to_ego",
        "ego_pose",
        "ego_state",
        "timestamps_s",
        "route_command",
        "valid_cameras",
    }


def test_threshold_tie_matches_selection():
    r = binary_metrics(np.array([0.5, 0.51]), np.array([1, 1]), np.ones(2, bool), 0.5)
    assert r["tp"] == 1 and r["fn"] == 1
