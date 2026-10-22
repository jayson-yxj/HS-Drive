"""Optional RGB-only target hypotheses. No actor/map/contact GT input accepted."""

import numpy as np
from diagnostics.m5l_objects import components, locate, plane_intersection


def candidate_features(component, q, K, T):
    x, y, w, h = component["bbox"]
    m = component["mask"]
    uv = component["uv"]
    p, reason = locate(component, K, T, q.shape)
    solidity = float(m.sum() / max(w * h, 1))
    bottom = m & (np.indices(m.shape)[0] >= uv[1] - 2)
    bottom_q = float(q[bottom].mean()) if bottom.any() else 0.0
    sensitivity = None
    if p is not None:
        points = [
            plane_intersection(uv + np.array([0, d]), K, T, z)[0]
            for d, z in [(-1, -0.25), (1, -0.25), (0, -0.35), (0, -0.15)]
        ]
        sensitivity = max(
            float(np.linalg.norm(v[:2] - p[:2])) for v in points if v is not None
        )
    confidence = (
        0.0
        if p is None
        else float(
            np.clip(
                min(1.0, h / 12)
                * min(1.0, w / 8)
                * solidity
                * bottom_q
                * np.exp(-sensitivity / 3),
                0,
                1,
            )
        )
    )
    reasons = [] if p is not None else [reason]
    if w < 4 or h < 6:
        reasons.append("insufficient_rgb_extent")
    if solidity < 0.35:
        reasons.append("fragmented_component")
    return p, dict(
        confidence=confidence,
        sensitivity_m=sensitivity,
        solidity=solidity,
        bottom_foreground_probability=bottom_q,
        reasons=reasons,
    )


def predict(q, K, T, *, timestamp, threshold=0.0, camera="front"):
    if not np.isfinite(timestamp) or not 0 <= threshold <= 1:
        raise ValueError("invalid time or threshold")
    q = np.asarray(q)
    if q.ndim != 2 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("foreground probability")
    out = []
    for i, c in enumerate(components(q)):
        p, f = candidate_features(c, q, K, T)
        reasons = list(f["reasons"])
        if f["confidence"] < threshold:
            reasons.append("below_validation_confidence")
        eligible = p is not None and not reasons
        out.append(
            dict(
                candidate_id=f"{camera}:{i}",
                class_hint="foreground_unknown_class",
                image_region=c["bbox"],
                xy_hypothesis=p[:2].tolist() if p is not None else None,
                uncertainty=dict(
                    kind="sensitivity_not_calibrated_error_bound",
                    xy_sensitivity_m=f["sensitivity_m"],
                ),
                confidence=f["confidence"],
                status="Unknown",
                timestamp=float(timestamp),
                conditional_status="ConditionalPosition" if eligible else "Detected",
                assumptions=[
                    "fixed_ground_z=-0.25m",
                    "RGB_bottom_is_contact_unverified",
                ],
                rejection_reasons=reasons + ["contact_and_ground_not_verified"],
                rgb_features=f,
            )
        )
    return out
