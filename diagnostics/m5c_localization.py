"""Absolute visible surface metrics. Known surface support != observed volume.

Symmetric nearest-neighbor matching within identical height layer, XY <= tau.
Many-to-one surface coverage, NOT one-to-one object association. Unknown output
is reported separately and never counted as a safe negative.
"""

import numpy as np
from scipy.spatial import cKDTree
from diagnostics.m5a_metrics import binary_metrics


def directed_distance(source, target):
    result = np.full(len(source), np.inf)
    for layer in range(4):
        a = source[:, 0] == layer
        b = target[:, 0] == layer
        if a.any() and b.any():
            result[a] = cKDTree(target[b, 1:] * 0.5).query(source[a, 1:] * 0.5)[0]
    return result


def matching(predicted, truth, tolerances=(0.5, 1.0)):
    p = np.argwhere(predicted)
    g = np.argwhere(truth)
    pg = directed_distance(p, g)
    gp = directed_distance(g, p)

    def dist(x):
        finite = x[np.isfinite(x)]
        return dict(
            n=len(x),
            no_same_layer_candidate=int((~np.isfinite(x)).sum()),
            mean_m=float(finite.mean()) if len(finite) else None,
            p95_m=float(np.quantile(finite, 0.95)) if len(finite) else None,
            max_m=float(finite.max()) if len(finite) else None,
        )

    out = dict(pred_to_gt=dist(pg), gt_to_pred=dist(gp), tolerances={})
    for t in tolerances:
        mp = int((pg <= t + 1e-8).sum())
        mg = int((gp <= t + 1e-8).sum())
        precision = mp / len(p) if len(p) else 0.0
        recall = mg / len(g) if len(g) else None
        out["tolerances"][str(t)] = dict(
            pred_count=len(p),
            gt_count=len(g),
            matched_pred=mp,
            matched_gt=mg,
            extra_pred=len(p) - mp,
            unmatched_gt=len(g) - mg,
            precision=precision,
            recall=recall,
            fscore=(
                2 * precision * recall / (precision + recall)
                if recall is not None and precision + recall
                else 0.0
            ),
        )
    return out


def absolute_metrics(score, target, valid, threshold, *, actor_masks=None):
    predicted = score > threshold
    observed_pred = predicted & valid
    truth = target & valid
    result = dict(
        exact=binary_metrics(score, target, valid, threshold),
        spatial=matching(observed_pred, truth),
        known_support_fraction=float(valid.mean()),
        unknown_predicted=int((predicted & ~valid).sum()),
        unknown_cells=int((~valid).sum()),
        unknown_predicted_fraction=(
            float(predicted[~valid].mean()) if (~valid).any() else None
        ),
        by_layer={},
        by_layer_spatial={},
        by_distance={},
        actors=[],
    )
    for z in range(4):
        result["by_layer"][str(z)] = binary_metrics(
            score[z], target[z], valid[z], threshold
        )
    for z in range(4):
        area = np.zeros_like(valid)
        area[z] = True
        result["by_layer_spatial"][str(z)] = matching(
            observed_pred & area, truth & area
        )
    x = -20 + (np.arange(160) + 0.5) * 0.5
    y = -30 + (np.arange(120) + 0.5) * 0.5
    d = np.hypot(x[:, None], y[None, :])
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]:
        area = np.broadcast_to(((d >= lo) & (d < hi))[None], score.shape)
        result["by_distance"][f"{lo}_{hi}"] = dict(
            exact=binary_metrics(score, target, valid & area, threshold),
            spatial=matching(observed_pred & area, truth & area),
        )
    corridor = (x[:, None] > 0) & (x[:, None] < 35) & (abs(y[None, :]) <= 1.5)
    result["corridor"] = binary_metrics(
        score, target, valid & corridor[None], threshold
    )
    if actor_masks:
        for actor in actor_masks:
            m = actor["mask"] & valid
            n = int(m.sum())
            dist = directed_distance(np.argwhere(m), np.argwhere(observed_pred))
            hit = int((dist <= 1.0 + 1e-8).sum())
            fraction = hit / max(1, n)
            result["actors"].append(
                dict(
                    actor_id=actor["actor_id"],
                    type_id=actor["type_id"],
                    distance_m=actor.get("distance_m"),
                    visible_cells=n,
                    matched_cells_1m=hit,
                    visible_surface_recall_1m=fraction if n else None,
                    detected=n >= 3 and hit >= 3 and fraction >= 0.2,
                    eligible=n >= 3,
                    exact_hits=int((predicted & m).sum()),
                )
            )
    return result


def scene_score(prob, labels, rows, thresholds):
    best = (-1, None)
    for threshold in thresholds:
        scores = []
        for group in sorted(set(r["group"] for r in rows)):
            ids = [i for i, r in enumerate(rows) if r["group"] == group]
            m = labels["obstacle_valid"][ids]
            t = labels["surface_semantic"][ids, 1]
            p = prob[ids] > threshold
            tp = int((p & t & m).sum())
            fp = int((p & ~t & m).sum())
            fn = int((~p & t & m).sum())
            scores.append(tp / max(1, tp + fp + fn))
        value = float(np.mean(scores))
        if value > best[0]:
            best = (value, threshold)
    return best
