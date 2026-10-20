"""Direct logit supervision, equally averaged across direction and radial bands."""

import torch
from torch.nn import functional as F


def existence_loss(output, pair, *, stable=False):
    logits = output["visible_obstacle"].float()
    p = logits.sigmoid()
    x = -20 + (torch.arange(logits.shape[-2], device=logits.device) + 0.5) * 0.5
    y = -30 + (torch.arange(logits.shape[-1], device=logits.device) + 0.5) * 0.5
    distance = torch.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    terms = []
    ranks = []
    counts = {}
    for key, positive, negative in [("appeared", 1, 0), ("vanished", 0, 1)]:
        parts = []
        rr = []
        for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]:
            mask = pair[key] & ((distance >= lo) & (distance < hi))[None]
            counts[f"{key}_{lo}_{hi}"] = int(mask.sum())
            if mask.any():
                parts.append(
                    (
                        F.softplus(-logits[positive][mask]).mean()
                        + F.softplus(logits[negative][mask]).mean()
                    )
                    / 2
                )
                rr.append(F.relu(0.1 - (p[positive] - p[negative])[mask]).mean())
        if parts:
            terms.append(torch.stack(parts).mean())
            ranks.append(torch.stack(rr).mean())
    bce = torch.stack(terms).mean() if terms else logits.sum() * 0
    ranking = torch.stack(ranks).mean() if ranks else logits.sum() * 0
    consistency = (
        (p[1] - p[0])[pair["stable"]].square().mean()
        if stable and pair["stable"].any()
        else logits.sum() * 0
    )
    return bce + 0.1 * ranking + 0.2 * consistency, dict(
        bce=bce, ranking=ranking, stable=consistency, counts=counts
    )
