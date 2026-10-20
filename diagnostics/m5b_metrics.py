"""Paired signed changes on supported witness/background cells; never volume-free."""

import numpy as np
from diagnostics.m5a_metrics import binary_metrics


def change_metrics(pa, pb, pair, threshold):
    delta = np.asarray(pb) - np.asarray(pa)
    appear = pair["appeared"]
    vanish = pair["vanished"]
    stable = pair["stable"]
    valid = appear | vanish | stable
    predicted_up = (delta > threshold) & valid
    predicted_down = (delta < -threshold) & valid
    tp = int((predicted_up & appear).sum() + (predicted_down & vanish).sum())
    fp = int((predicted_up & ~appear).sum() + (predicted_down & ~vanish).sum())
    fn = int((appear & ~predicted_up).sum() + (vanish & ~predicted_down).sum())
    change = appear | vanish
    direction = appear.astype(float) - vanish.astype(float)
    out = dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
        valid=int(valid.sum()),
        changed=int(change.sum()),
        appeared=int(appear.sum()),
        vanished=int(vanish.sum()),
        stable=int(stable.sum()),
        mean_signed_change=(
            float((delta[change] * direction[change]).mean()) if change.any() else None
        ),
        direction_correct=(
            float((delta[change] * direction[change] > 0).mean())
            if change.any()
            else None
        ),
        disappearance_failed_drop=(
            float((delta[vanish] >= -threshold).mean()) if vanish.any() else None
        ),
        stable_false_change=(
            float((abs(delta[stable]) > threshold).mean()) if stable.any() else None
        ),
        threshold=threshold,
    )
    # Localization nearest predicted changed occupied witness in XY; unmatched explicit.
    gt = np.argwhere(appear)
    pr = np.argwhere(predicted_up)
    errors = []
    if len(gt) and len(pr):
        for pts in np.array_split(gt, max(1, int(np.ceil(len(gt) / 128)))):
            errors.extend(
                np.linalg.norm((pts[:, None, 1:] - pr[None, :, 1:]) * 0.5, axis=-1)
                .min(1)
                .tolist()
            )
    out["disappearance_residual_score_mean"] = (
        float(np.asarray(pb)[vanish].mean()) if vanish.any() else None
    )
    out["disappearance_residual_above_half"] = (
        float((np.asarray(pb)[vanish] > 0.5).mean()) if vanish.any() else None
    )
    out["residual_half_threshold_scope"] = (
        "diagnostic fixed score .5, not safety threshold"
    )
    out["appearance_localization"] = dict(
        reference_surface_cells=len(gt),
        predicted_change_cells=len(pr),
        unmatched=len(gt) if not len(pr) else 0,
        nearest_xy_mae_m=float(np.mean(errors)) if errors else None,
    )
    x = -20 + (np.arange(160) + 0.5) * 0.5
    y = -30 + (np.arange(120) + 0.5) * 0.5
    d = np.hypot(x[:, None], y[None, :])
    out["visible_distance_recall"] = {}
    for lo, hi in [(0, 7.5), (7.5, 12.5), (12.5, 20), (20, 35), (35, 80)]:
        m = change & ((d >= lo) & (d < hi))[None]
        n = int(m.sum())
        hit = ((predicted_up & appear) | (predicted_down & vanish)) & m
        out["visible_distance_recall"][f"{lo}_{hi}"] = dict(
            n=n, recall=float(hit.sum() / n) if n else None
        )
    out["by_layer"] = {
        str(z): dict(
            n=int(change[z].sum()),
            hit=int(
                ((predicted_up[z] & appear[z]) | (predicted_down[z] & vanish[z])).sum()
            ),
        )
        for z in range(4)
    }
    return out


def aggregate(records):
    groups = {}
    for group in sorted(set(r["group"] for r in records)):
        rows = [r["metrics"] for r in records if r["group"] == group]
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        groups[group] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            iou=tp / max(1, tp + fp + fn),
            recall=tp / max(1, tp + fn),
            precision=tp / max(1, tp + fp),
            pairs=len(rows),
            changed=sum(r["changed"] for r in rows),
        )
    return dict(
        by_scene=groups,
        mean_scene_iou=(
            float(np.mean([v["iou"] for v in groups.values()])) if groups else None
        ),
    )
