"""Location-heldout empirical risk diagnostics. No sample-iid safety claim."""

import numpy as np
from scipy.stats import spearmanr
from diagnostics.m5n_metrics import fit_risk, predict_risk, risk_features
from datasets.m5o_benchmark import evaluate, METHODS, summary


def fit_calibration(candidates, actors, predictions):
    byid = {c["candidate_id"]: c for c in candidates}
    fit = [a for a in actors if a["detected"] and a.get("unique_association", True)]
    X = np.array(
        [
            risk_features(
                byid[a["candidate_id"]],
                predictions[a["candidate_id"]]["positions"]["direct"],
            )
            for a in fit
        ]
    )
    y = np.array(
        [
            np.linalg.norm(
                np.array(predictions[a["candidate_id"]]["positions"]["direct"])[:2]
                - a["box_center_ego"][:2]
            )
            for a in fit
        ]
    )
    groups = np.array([a["group"] for a in fit])
    if len(set(groups)) < 3:
        raise ValueError("independent calibration locations insufficient")
    allX = np.array(
        [
            risk_features(c, predictions[c["candidate_id"]]["positions"]["direct"])
            for c in candidates
        ]
    )
    allg = np.array([c["group"] for c in candidates])
    oof = {}
    folds = []
    for g in sorted(set(groups)):
        ix = groups != g
        cal = fit_risk(X[ix], y[ix])
        mask = allg == g
        vals = predict_risk(cal, allX[mask])
        oof.update(
            {
                candidates[i]["candidate_id"]: float(v)
                for i, v in zip(np.flatnonzero(mask), vals)
            }
        )
        folds.append(
            dict(heldout=g, train_groups=sorted(set(groups[ix])), fit_n=int(ix.sum()))
        )
    risk = np.array([oof[a["candidate_id"]] for a in fit])
    ratios = y / risk
    multiplier = float(np.quantile(ratios, 0.9))
    return (
        dict(
            model=fit_risk(X, y),
            empirical90_multiplier=multiplier,
            folds=folds,
            calibration_groups=sorted(set(groups)),
            samples=len(y),
            oof_spearman=float(spearmanr(risk, y).statistic),
            calibration_oof_coverage=float((y <= risk * multiplier).mean()),
            limitations="Empirical location-heldout residual scaling; clustered/heterogeneous data and threshold selection invalidate iid conformal/safety guarantees.",
        ),
        oof,
    )


def apply_calibration(cal, candidates, predictions, override=None):
    X = [
        risk_features(c, predictions[c["candidate_id"]]["positions"]["direct"])
        for c in candidates
    ]
    risks = predict_risk(cal["model"], X)
    for c, value in zip(candidates, risks):
        key = c["candidate_id"]
        risk = float((override or {}).get(key, value))
        predictions[key].update(
            proxy_error_m=risk,
            confidence=float(np.exp(-risk / 5)),
            empirical90_radius_m=risk * cal["empirical90_multiplier"],
        )


def common_targets(actors, predictions, threshold):
    return [
        a
        for a in actors
        if a["detected"]
        and a.get("unique_association", True)
        and predictions[a["candidate_id"]].get("confidence", 1) >= threshold
        and all(
            predictions[a["candidate_id"]]["positions"].get(m) is not None
            for m in METHODS
        )
    ]


def quality_gate(actors, predictions, threshold):
    metrics = evaluate(actors, predictions, "direct", threshold)
    aa = common_targets(actors, predictions, threshold)
    same = {m: evaluate(aa, predictions, m, threshold) for m in METHODS}
    d = same["direct"]["xy"]
    passed = (
        metrics["coverage"] >= 0.3
        and metrics["localized"] >= 20
        and metrics["groups"] >= 3
        and metrics["xy"]["p95"] <= 5
        and metrics["xy"]["severe5m_rate"] <= 0.05
        and len(aa) >= 20
    )
    passed = passed and all(
        d["mean"] < same[m]["xy"]["mean"] and d["p95"] <= same[m]["xy"]["p95"]
        for m in ["plane", "geometry_prior"]
    )
    return dict(
        threshold=threshold,
        passed=bool(passed),
        metrics=metrics,
        same_target_n=len(aa),
        same_target=same,
    )


def risk_report(actors, predictions):
    aa = [a for a in actors if a["detected"] and a.get("unique_association", True)]
    errors = np.array(
        [
            np.linalg.norm(
                np.array(predictions[a["candidate_id"]]["positions"]["direct"])[:2]
                - a["box_center_ego"][:2]
            )
            for a in aa
        ]
    )
    risks = np.array([predictions[a["candidate_id"]]["proxy_error_m"] for a in aa])
    groups = np.array([a["group"] for a in aa])
    order = np.argsort(risks, kind="stable")
    points = []
    for f in [0.1, 0.3, 0.5, 0.7, 1]:
        count = int(np.ceil(len(actors) * f))
        ids = order[:count]
        points.append(
            dict(
                requested_visible_coverage=f,
                achieved_coverage=len(ids) / max(len(actors), 1),
                feasible=count <= len(aa),
                xy=summary(errors[ids]),
            )
        )
    prefix = [
        i for i in range(1, len(order) + 1) if (errors[order[:i]] > 5).mean() <= 0.05
    ]
    pergroup = [
        dict(
            group=g,
            n=int((groups == g).sum()),
            xy=summary(errors[groups == g]),
            interval_coverage=float(
                np.mean(
                    [
                        errors[i]
                        <= predictions[a["candidate_id"]]["empirical90_radius_m"]
                        for i, a in enumerate(aa)
                        if a["group"] == g
                    ]
                )
            ),
        )
        for g in sorted(set(groups))
    ]
    # Whole-location bootstrap, no artificial independence for views/frames. Descriptive only.
    rng = np.random.default_rng(5150)
    unique = sorted(set(groups))
    draws = []
    for _ in range(500 if unique else 0):
        sampled = rng.choice(unique, len(unique), replace=True)
        e = np.concatenate([errors[groups == g] for g in sampled])
        draws.append(np.quantile(e, 0.95))
    return dict(
        n=len(aa),
        spearman=float(spearmanr(risks, errors).statistic) if len(aa) > 2 else None,
        fixed_coverage=points,
        max_coverage_at_empirical_severe5m_le05=(
            max(prefix) / max(len(actors), 1) if prefix else 0
        ),
        interval_coverage=(
            float(
                np.mean(
                    [
                        e <= predictions[a["candidate_id"]]["empirical90_radius_m"]
                        for a, e in zip(aa, errors)
                    ]
                )
            )
            if aa
            else None
        ),
        group_p95_bootstrap95=(
            np.quantile(draws, [0.025, 0.975]).tolist() if draws else None
        ),
        locations=pergroup,
        limits="Risk-ordered curves and bootstrap are descriptive only; no test threshold chosen. Repeated layout family and map correlation remain; few locations do not establish population bounds.",
    )
