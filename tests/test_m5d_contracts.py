import numpy as np
import torch
from models.m5c_obstacle_head import VisibleObstacleHead
from training.m5b_change_loss import change_loss
from training.m5d_existence_loss import existence_loss
from diagnostics.m5d_metrics import witness_metrics, aggregate_witness


def test_pair_direct_gradient_and_unknown_zero():
    logits = torch.zeros((2, 4, 160, 120), requires_grad=True)
    mask = torch.zeros((4, 160, 120), dtype=torch.bool)
    p = {k: mask.clone() for k in ["appeared", "vanished", "stable"]}
    p["appeared"][2, 60, 60] = True
    p["vanished"][1, 70, 60] = True
    loss, _ = existence_loss(dict(visible_obstacle=logits), p)
    loss.backward()
    assert logits.grad[1, 2, 60, 60] < 0 and logits.grad[0, 2, 60, 60] > 0
    assert logits.grad[0, 1, 70, 60] < 0 and logits.grad[1, 1, 70, 60] > 0
    assert logits.grad.count_nonzero() == 4


def test_old_pair_loss_has_no_direct_output_parameter_gradient():
    head = VisibleObstacleHead()
    p = {
        k: torch.zeros((4, 3, 3), dtype=torch.bool)
        for k in ["appeared", "vanished", "stable"]
    }
    p["appeared"][1, 1, 1] = True
    loss, _ = change_loss(head(torch.randn(2, 128, 3, 3)), p)
    loss.backward()
    assert head.visible_obstacle.weight.grad is None
    assert head.shared[0].weight.grad is not None


def test_absence_cannot_pass_by_predicting_all_negative():
    m = np.zeros((4, 160, 120), bool)
    m[2, 60, 60] = True
    p = dict(appeared=m, vanished=np.zeros_like(m))
    a = np.zeros_like(m, dtype=float)
    b = a.copy()
    rows = witness_metrics(a, b, p, 0.5)
    s = aggregate_witness([dict(group="g", witness=rows)])["by_scene"]["g"]
    assert s["absence_fpr"] == 0 and s["presence_recall"] == 0 and s["harmonic"] == 0


def test_certification_veto_replacement_and_unreliable_counterfactual(monkeypatch):
    import datasets.m5d_witness as mod

    shape = (4, 160, 120)
    blank = np.zeros(shape, bool)
    lab = {
        k: blank.copy()
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
    lab["surface_semantic"] = np.zeros((3, *shape), bool)
    uv = np.array([[1, 1], [1, 2], [2, 1]])
    ra = dict(
        uv=uv,
        endpoints=np.tile([10.0, 0.0, 1.2], (3, 1)),
        reliable=np.ones(3, bool),
        semantics=np.ones(3),
        stable_mask=np.ones((4, 4), bool),
        self_mask=np.zeros((4, 4), bool),
    )
    rb = {
        **ra,
        "endpoints": np.tile([5.1, 0.1, 1.2], (3, 1)),
        "semantics": np.full(3, 2),
    }
    a = dict(frame_id=1, cameras={"f": {}}, actors=[])
    b = dict(
        frame_id=2,
        cameras={"f": {}},
        actors=[dict(is_ego=False, corners_hs_ego_m=[[4, -1, 0], [6, 1, 2]])],
    )
    legacy = {k: blank.copy() for k in ["appeared", "vanished", "stable"]}
    legacy["appeared"][2, 50, 60] = True
    monkeypatch.setattr(mod, "pair_witness", lambda *a: (legacy, []))

    def camera(root, man, f, name):
        return (ra if f["frame_id"] == 1 else rb), (
            np.full((4, 4), 10.0 if f["frame_id"] == 1 else 5.0, dtype=np.float32),
            np.ones((4, 4)),
            np.full((4, 4), f["frame_id"]),
        )

    monkeypatch.setattr(mod, "camera_rays", camera)
    p, _ = mod.certified_witness(None, None, a, b, lab, lab)
    assert p["appeared"].sum() == 1
    ra["reliable"][:] = False
    p, _ = mod.certified_witness(None, None, a, b, lab, lab)
    assert p["appeared"].sum() == 0
    ra["reliable"][:] = True
    ra["endpoints"][:] = [5.1, 0.1, 1.2]
    ra["semantics"][:] = 2
    p, _ = mod.certified_witness(None, None, a, b, lab, lab)
    assert p["appeared"].sum() == 0
