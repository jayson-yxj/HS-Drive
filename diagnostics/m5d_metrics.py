"""Presence/absence are conditional point-witness scores, not whole-volume safety."""

import numpy as np
from diagnostics.m5c_localization import matching
from diagnostics.m5a_metrics import binary_metrics


def witness_metrics(pa, pb, pair, threshold, legacy=False):
    prefix = "legacy_" if legacy else ""
    values = []
    rows = []
    xx = -20 + (np.arange(160) + 0.5) * 0.5
    yy = -30 + (np.arange(120) + 0.5) * 0.5
    dist = np.hypot(xx[:, None], yy[None, :])
    for direction, present, absent in [("appeared", pb, pa), ("vanished", pa, pb)]:
        mask = pair[prefix + direction]
        pos = present[mask]
        neg = absent[mask]
        n = len(pos)

        def metric(m):
            p = present[m]
            a = absent[m]
            count = len(p)
            return dict(
                n=count,
                presence_hits=int((p > threshold).sum()),
                absence_false_positives=int((a > threshold).sum()),
                both_correct=int(((p > threshold) & (a <= threshold)).sum()),
                presence_recall=float((p > threshold).mean()) if count else None,
                absence_fpr=float((a > threshold).mean()) if count else None,
                mean_presence=float(p.mean()) if count else None,
                mean_absence=float(a.mean()) if count else None,
                direction_correct=float((p > a).mean()) if count else None,
            )

        rows.append(
            dict(
                direction=direction,
                **metric(mask),
                by_layer={
                    str(z): metric(
                        mask
                        & np.broadcast_to(
                            (np.arange(4) == z)[:, None, None], mask.shape
                        )
                    )
                    for z in range(4)
                },
                by_distance={
                    f"{lo}_{hi}": metric(mask & ((dist >= lo) & (dist < hi))[None])
                    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]
                },
            )
        )
    return rows


def aggregate_witness(records):
    groups = {}
    for g in sorted(set(r["group"] for r in records)):
        rows = [x for r in records if r["group"] == g for x in r["witness"]]
        n = sum(r["n"] for r in rows)
        hit = sum(r["presence_hits"] for r in rows)
        fp = sum(r["absence_false_positives"] for r in rows)
        rec = hit / n if n else 0
        spec = 1 - fp / n if n else 0
        groups[g] = dict(
            n=n,
            presence_hits=hit,
            absence_false_positives=fp,
            presence_recall=rec if n else None,
            absence_fpr=1 - spec if n else None,
            precision=hit / (hit + fp) if hit + fp else None,
            both_correct_fraction=(
                sum(r["both_correct"] for r in rows) / n if n else None
            ),
            harmonic=2 * rec * spec / (rec + spec) if rec + spec else 0,
        )
    return dict(
        by_scene=groups,
        mean_scene_harmonic=(
            float(np.mean([r["harmonic"] for r in groups.values()])) if groups else 0
        ),
    )


def choose(prob, pairs, lookup, thresholds):
    scores = []
    for t in thresholds:
        records = [
            dict(
                group=r["group"],
                witness=witness_metrics(
                    prob[lookup[r["a"]]], prob[lookup[r["b"]]], d, t
                ),
            )
            for r, d in pairs
        ]
        scores.append((aggregate_witness(records)["mean_scene_harmonic"], t))
    return max(scores, key=lambda x: (x[0], -x[1]))


def voxel_3d_matching(pred, truth):
    from scipy.spatial import cKDTree

    def points(mask):
        z, x, y = np.where(mask)
        return np.stack(
            (-20 + (x + 0.5) * 0.5, -30 + (y + 0.5) * 0.5, -2 + (z + 0.5) * 1.5), -1
        )

    p = points(pred)
    g = points(truth)

    def directed(a, b):
        if not len(a):
            return dict(n=0, no_candidate=0, mean_m=None, p95_m=None, unmatched_1m=0)
        if not len(b):
            return dict(
                n=len(a),
                no_candidate=len(a),
                mean_m=None,
                p95_m=None,
                unmatched_1m=len(a),
            )
        d = cKDTree(b).query(a)[0]
        return dict(
            n=len(a),
            no_candidate=0,
            mean_m=float(d.mean()),
            p95_m=float(np.quantile(d, 0.95)),
            unmatched_1m=int((d > 1).sum()),
        )

    return dict(
        pred_to_gt=directed(p, g),
        gt_to_pred=directed(g, p),
        reference="coarse voxel centers; not precise surface point 3D errors",
    )
