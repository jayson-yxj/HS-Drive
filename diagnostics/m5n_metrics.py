"""Visible-denominator object localization and empirical calibration diagnostics."""

import numpy as np
from tools.m5n_common import stats


def error(position, reference):
    p = np.asarray(position)
    g = np.asarray(reference)
    a = np.arctan2(p[1], p[0]) - np.arctan2(g[1], g[0])
    delta = np.linalg.norm(p[:2]) - np.linalg.norm(g[:2])
    return dict(
        xy=float(np.linalg.norm(p[:2] - g[:2])),
        range_abs=float(abs(delta)),
        range_bias=float(delta),
        bearing_deg=float(abs(np.degrees(np.arctan2(np.sin(a), np.cos(a))))),
    )


def risk_features(candidate, predicted):
    x, y, w, h = candidate["region"]
    return np.array(
        [
            1,
            np.log(max(np.linalg.norm(predicted[:2]), 1)),
            np.log(max(w, 1)),
            np.log(max(h, 1)),
            candidate["pixel_area"] / (w * h),
            candidate["q_mean"],
            float(x <= 1 or y <= 1 or x + w >= 639 or y + h >= 383),
        ],
        float,
    )


def fit_risk(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    mean = X[:, 1:].mean(0)
    scale = X[:, 1:].std(0).clip(0.1)
    Z = np.c_[np.ones(len(X)), (X[:, 1:] - mean) / scale]
    pen = np.eye(Z.shape[1]) * 10
    pen[0, 0] = 0
    beta = np.linalg.solve(Z.T @ Z + pen, Z.T @ np.log(np.maximum(y, 0.1)))
    return dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist())


def predict_risk(cal, X):
    X = np.asarray(X)
    Z = np.c_[np.ones(len(X)), (X[:, 1:] - cal["mean"]) / cal["scale"]]
    return np.exp(np.clip(Z @ cal["beta"], np.log(0.1), np.log(80)))


def aggregate(actors, predictions, method, threshold=0.0):
    selected = []
    det = 0
    for r in actors:
        if not r["detected"]:
            continue
        det += 1
        p = predictions[r["candidate_id"]]
        pos = p["positions"].get(method)
        if pos is None or not np.isfinite(pos).all():
            continue
        if method == "direct" and p["confidence"] < threshold:
            continue
        selected.append((r, p, error(pos, r["box_center_ego"])))
    return dict(
        visible=len(actors),
        detected=det,
        localized=len(selected),
        positions=len({r["group"] for r, _, _ in selected}),
        coverage=len(selected) / max(len(actors), 1),
        detection_rate=det / max(len(actors), 1),
        undetected=len(actors) - det,
        rejected_detected=det - len(selected),
        xy=stats([e["xy"] for _, _, e in selected]),
        range_abs=stats([e["range_abs"] for _, _, e in selected]),
        signed_range_bias=stats([e["range_bias"] for _, _, e in selected]),
        bearing_deg=stats([e["bearing_deg"] for _, _, e in selected]),
        severe5m_rate=(
            float(np.mean([e["xy"] > 5 for _, _, e in selected])) if selected else None
        ),
        unlocalized_not_zero_error=len(actors) - len(selected),
    )


def same_target(actors, predictions, threshold=0.0):
    rr = [
        a
        for a in actors
        if a["detected"]
        and predictions[a["candidate_id"]]["confidence"] >= threshold
        and all(
            predictions[a["candidate_id"]]["positions"].get(m) is not None
            for m in ["plane", "dense_expected", "dense_mode", "direct"]
        )
    ]
    return dict(
        n=len(rr),
        methods={
            m: aggregate(rr, predictions, m)
            for m in [
                "plane",
                "dense_expected",
                "dense_mode",
                "direct",
                "geometry_prior",
            ]
        },
    )


def qualification(metrics, matched):
    if not (
        metrics["coverage"] >= 0.3
        and metrics["localized"] >= 20
        and metrics["positions"] >= 3
        and metrics["xy"].get("p95", 1e9) <= 5
        and metrics["severe5m_rate"] is not None
        and metrics["severe5m_rate"] <= 0.05
        and matched["n"] >= 20
    ):
        return False
    d = matched["methods"]["direct"]["xy"]
    return all(
        d["mean"] < matched["methods"][m]["xy"]["mean"]
        and d["p95"] <= matched["methods"][m]["xy"]["p95"]
        for m in ["plane", "dense_expected", "dense_mode"]
    )
