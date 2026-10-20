"""Pair labels enter loss only; single-frame RGB inference unchanged."""

import torch
from training.m48_evidence_loss import evidence_loss


def obstacle_score(output):
    return (
        output["surface"].float().sigmoid() * output["semantic"][:, 1].float().sigmoid()
    )


def change_loss(output, pair, margin=0.1):
    p = obstacle_score(output)
    delta = p[1] - p[0]
    terms = []
    for mask, sign in [(pair["appeared"], 1), (pair["vanished"], -1)]:
        if mask.any():
            terms.append(torch.relu(margin - sign * delta[mask]).mean())
    ranking = sum(terms) / len(terms) if terms else delta.sum() * 0
    stable = (
        delta[pair["stable"]].square().mean()
        if pair["stable"].any()
        else delta.sum() * 0
    )
    return ranking + 0.1 * stable, dict(ranking=ranking, stable=stable)
