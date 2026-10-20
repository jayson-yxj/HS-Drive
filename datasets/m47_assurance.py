"""Evidence-based quarantine, never a blanket Unknown dilation."""

import numpy as np
from datasets.occupancy_labels import validate_occupancy


def quarantine_conflicts(labels, ranks):
    out = {k: v.copy() for k, v in labels.items()}
    selected = np.zeros(out["occupancy_state"].size, bool)
    selected[np.asarray(ranks, dtype=int)] = True
    selected = selected.reshape(out["occupancy_state"].shape) & (
        out["occupancy_state"] == 1
    )
    out["occupancy_state"][selected] = 0
    out["occupancy_valid"][selected] = False
    out["surface_semantic"][selected] = 0
    for key in ["surface_semantic_valid", "obstacle_mask", "obstacle_valid"]:
        out[key][selected] = False
    out["m47_conflict_quarantine"] = selected
    validate_occupancy(out)
    return out


def segment_distance(points, a, b):
    d = b - a
    t = np.clip(((points - a) * d).sum(-1) / max(float(d @ d), 1e-12), 0, 1)
    return np.linalg.norm(points - a - t[:, None] * d, axis=-1)
