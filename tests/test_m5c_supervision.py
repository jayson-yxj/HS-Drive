import torch
from models.m48_evidence_head import EvidenceHead
from models.m5c_obstacle_head import VisibleObstacleHead
from training.m5c_obstacle_loss import visible_obstacle_loss


def test_unknown_free_only_and_conflicting_semantics_have_zero_gradient():
    x = torch.zeros(1, 4, 2, 2, requires_grad=True)
    sem = torch.zeros(1, 3, 4, 2, 2, dtype=torch.bool)
    sem[0, 1, 0, 0, 0] = True
    sem[0, 0, 0, 0, 1] = True
    sem[0, :2, 0, 1, 0] = True
    valid = torch.ones_like(x, dtype=torch.bool)
    blank = torch.zeros_like(valid)
    labels = dict(
        surface_semantic=sem,
        obstacle_valid=valid,
        surface_semantic_valid=valid,
        ego_mask=blank,
        evidence_conflict_mask=blank,
        semantic_conflict_mask=blank,
        free_ray_evidence=valid,
    )
    loss, count = visible_obstacle_loss(dict(visible_obstacle=x), labels)
    loss.backward()
    assert count == dict(positive=1, negative=1)
    assert x.grad[0, 0, 0, 0] < 0 and x.grad[0, 0, 0, 1] > 0
    assert x.grad.count_nonzero() == 2


def test_added_head_preserves_identical_seeded_shared_initialization():
    torch.manual_seed(71)
    old = EvidenceHead()
    torch.manual_seed(71)
    new = VisibleObstacleHead()
    for name, param in old.state_dict().items():
        assert torch.equal(param, new.state_dict()[name])
    x = torch.randn(1, 128, 3, 4)
    a = old(x)
    b = new(x)
    for k in a:
        assert torch.equal(a[k], b[k])
    assert b["visible_obstacle"].shape == (1, 4, 3, 4)
