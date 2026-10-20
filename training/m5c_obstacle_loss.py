"""Independent surface obstacle BCE: no absence inferred from free-ray evidence."""

import torch
from torch.nn import functional as F


def visible_obstacle_loss(outputs, labels):
    logits = outputs["visible_obstacle"].float()
    sem = labels["surface_semantic"].bool()
    safe = labels["obstacle_valid"].bool() & labels["surface_semantic_valid"].bool()
    for key in ["ego_mask", "evidence_conflict_mask", "semantic_conflict_mask"]:
        safe = safe & ~labels[key].bool()
    pos = safe & sem[:, 1] & ~sem[:, 0] & ~sem[:, 2]
    neg = safe & sem[:, 0] & ~sem[:, 1] & ~sem[:, 2]
    terms = []
    if pos.any():
        terms.append(F.softplus(-logits[pos]).mean())
    if neg.any():
        terms.append(F.softplus(logits[neg]).mean())
    return (sum(terms) / len(terms) if terms else logits.sum() * 0), dict(
        positive=int(pos.sum()), negative=int(neg.sum())
    )
