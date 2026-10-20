"""Observed Free/Surface loss; Unknown is an ignore mask, never a negative."""

import torch
from torch.nn import functional as F


def observed_occupancy_loss(logits, state, valid):
    """Logits [B,2,Z,X,Y]; states 0 Unknown, 1 Free, 2 Surface.

    This plumbing interface does not certify the source label quality and is
    separate from existing M4 losses and model forward inputs.
    """
    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError("Expected [B,2,Z,X,Y] observed occupancy logits")
    if state.shape != logits.shape[:1] + logits.shape[2:] or valid.shape != state.shape:
        raise ValueError("Occupancy target shape mismatch")
    if valid.dtype != torch.bool or not torch.equal(valid, state != 0):
        raise ValueError("Unknown must be ignored exactly")
    if not torch.all((state >= 0) & (state <= 2)):
        raise ValueError("Unknown occupancy state")
    if not valid.any():
        return logits.sum() * 0
    labels = torch.where(valid, state.long() - 1, -100)
    return F.cross_entropy(logits.float(), labels, ignore_index=-100)
