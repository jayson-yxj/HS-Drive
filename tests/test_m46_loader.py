import json
from pathlib import Path
import numpy as np, pytest

torch = pytest.importorskip("torch")
from datasets.occupancy_training import OccupancyDataset
from simulator.recorder import atomic_json, file_hash
from training.occupancy_loss import observed_occupancy_loss


def test_unknown_is_ignored_not_negative_and_valid_classes_have_gradients():
    state = torch.tensor([[[[0, 1, 2]]]], dtype=torch.uint8)
    logits = torch.zeros(1, 2, 1, 1, 3, requires_grad=True)
    observed_occupancy_loss(logits, state, state != 0).backward()
    assert not logits.grad[..., 0].any()
    assert (logits.grad[..., 1:] != 0).all()
    assert logits.grad[0, 0, 0, 0, 1] < 0 and logits.grad[0, 1, 0, 0, 2] < 0
    with pytest.raises(ValueError, match="ignored exactly"):
        observed_occupancy_loss(logits, state, torch.ones_like(state, dtype=torch.bool))


def test_empty_observation_mask_has_zero_gradient():
    logits = torch.zeros(1, 2, 1, 1, 3, requires_grad=True)
    state = torch.zeros(1, 1, 1, 3, dtype=torch.uint8)
    observed_occupancy_loss(logits, state, state != 0).backward()
    assert not logits.grad.any()


def test_inference_never_opens_privileged_label(tmp_path, monkeypatch):
    index = Path("/root/gpufree-data/hs-drive-data/processed/m46_training_index")
    if not index.exists():
        pytest.skip("Actual M46 index not installed")
    ds = OccupancyDataset(index, "val", supervision=False)

    def forbidden(*a, **kw):
        raise AssertionError("Inference read NPZ GT")

    monkeypatch.setattr(np, "load", forbidden)
    item = ds[0]
    assert item.targets is None
    assert item.inputs["images"].shape == (6, 3, 384, 640)
    for key in ["occupancy_state", "instance", "reference_depth", "oracle_actors"]:
        with pytest.raises(KeyError):
            item.inputs[key]


def test_group_overlap_fails_closed(tmp_path):
    rows = [dict(group="same_route", split=s) for s in ["train", "val"]]
    atomic_json(tmp_path / "index.json", dict(samples=rows))
    atomic_json(
        tmp_path / "manifest.json",
        dict(
            schema_version="hsd-m46-training-index-v1",
            index_sha256=file_hash(tmp_path / "index.json"),
        ),
    )
    with pytest.raises(ValueError, match="crosses split"):
        OccupancyDataset(tmp_path, "train")
