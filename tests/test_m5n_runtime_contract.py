import ast
from pathlib import Path
import numpy as np
from diagnostics.m5n_metrics import fit_risk, predict_risk, qualification


def test_risk_calibration_numeric_and_no_permission_from_field_names():
    X = np.c_[np.ones(30), np.linspace(0, 2, 30), np.ones((30, 5))]
    y = np.exp(X[:, 1])
    cal = fit_risk(X, y)
    r = predict_risk(cal, X)
    assert np.isfinite(r).all() and np.corrcoef(r, y)[0, 1] > 0.95
    assert not qualification(
        dict(coverage=0.2, localized=100, positions=6, xy={"p95": 1}, severe5m_rate=0),
        {"n": 100},
    )


def test_offline_candidate_cannot_control_or_access_actor():
    tree = ast.parse(Path("models/m5n_candidate.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(v.name != "carla" for v in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in [
                "apply_control",
                "tick",
                "get_world",
                "get_actors",
                "get_map",
            ]
