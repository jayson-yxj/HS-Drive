"""Contact evidence: geometry references are evaluation-only, never RGB inputs."""

import numpy as np


def fit_plane(points, minimum=12, max_rmse=0.10):
    """Fit z=a*x+b*y+c with deterministic trimming, reject degenerate support."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]
    if len(p) < minimum:
        return None
    if len(p) > 2000:
        p = p[np.linspace(0, len(p) - 1, 2000).astype(int)]
    A = np.c_[p[:, :2], np.ones(len(p))]
    keep = np.ones(len(p), bool)
    for _ in range(3):
        if keep.sum() < minimum or np.linalg.matrix_rank(A[keep]) < 3:
            return None
        beta = np.linalg.lstsq(A[keep], p[keep, 2], rcond=None)[0]
        residual = np.abs(A @ beta - p[:, 2])
        keep = residual <= max(0.03, float(np.quantile(residual, 0.8)))
    rmse = float(np.sqrt(np.mean((A[keep] @ beta - p[keep, 2]) ** 2)))
    if rmse > max_rmse:
        return None
    return dict(
        coefficients=beta.tolist(),
        rmse_m=rmse,
        n=int(keep.sum()),
        slope_deg=float(np.degrees(np.arctan(np.linalg.norm(beta[:2])))),
        support_xy_min=p[keep, :2].min(0).tolist(),
        support_xy_max=p[keep, :2].max(0).tolist(),
    )


def intersect(uv, K, T, plane):
    a, b, c = plane["coefficients"]
    normal = np.array([-a, -b, 1.0])
    ray = T[:3, :3] @ np.linalg.solve(K, np.r_[uv, 1.0])
    den = normal @ ray
    if den >= -0.02:
        return None
    t = (c - normal @ T[:3, 3]) / den
    p = T[:3, 3] + t * ray
    return p if t > 0 and np.linalg.norm(p[:2]) <= 50 else None


def reference_status(
    *,
    plane,
    visible_pixels,
    anatomy_verified=False,
    collision_agrees=False,
    stable_pose=False,
    ambiguity=True,
):
    if plane is None or visible_pixels < 3:
        return "Unknown"
    if anatomy_verified and collision_agrees and stable_pose and not ambiguity:
        return "Gold"
    return "Candidate"
