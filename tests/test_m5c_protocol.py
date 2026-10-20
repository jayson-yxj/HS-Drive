import json
from pathlib import Path
import numpy as np


def test_groups_protocol_and_single_variable_training():
    p = json.loads(Path("configs/m5c_protocol.json").read_text())
    groups = p["groups"]
    assert len(groups) == 16 and len(set(g["group"] for g in groups)) == 16
    assert {
        s: sum(g["split"] == s for g in groups) for s in ["train", "val", "test"]
    } == {"train": 8, "val": 4, "test": 4}
    for a in groups:
        for b in groups:
            if a["group"] != b["group"] and a["map"] == b["map"]:
                assert np.linalg.norm(np.array(a["pose"][:2]) - b["pose"][:2]) >= 45
    t = p["training"]
    assert t["seeds"] == [71, 72] and t["steps"] == 400
    assert set(t["models"]) == {
        "P_prior",
        "A_frozen",
        "B_context",
        "C_pair",
        "D_direct",
    }
    assert p["configurations"][0] == p["configurations"][-1] == []


def test_no_original_architecture_or_controller_change():
    import hashlib

    f = json.loads(Path("reports/m5c_baseline_freeze.json").read_text())[
        "tracked_sha256"
    ]
    for path in [
        "models/depth_bev.py",
        "models/view_transform.py",
        "datasets/m48_evidence.py",
        "simulator/motion_oracle.py",
        "simulator/m48_motion.py",
        "simulator/m5a_async.py",
    ]:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == f[path]
