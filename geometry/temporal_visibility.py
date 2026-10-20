"""Conservative temporal support gating; visibility/motion sources explicit."""

import torch
from geometry.temporal_warp import warp_bev


def align_visible_history(
    previous,
    previous_pose,
    current_pose,
    previous_valid,
    current_observed,
    *,
    previous_dynamic=None,
    current_dynamic=None,
    **grid,
):
    """Unknown current cells never acquire a Free label from historical zeros.

    Dynamic masks and current observed support may be Oracle for diagnostics.
    Deploying this with GT masks is forbidden; learned/predicted masks must be
    validated separately. No categorical labels are interpolated by this API.
    """
    for value in (previous_valid, current_observed):
        if value.dtype != torch.bool:
            raise TypeError("Support masks must be boolean")
    valid = previous_valid
    if previous_dynamic is not None:
        if (
            previous_dynamic.shape != valid.shape
            or previous_dynamic.dtype != torch.bool
        ):
            raise ValueError("Dynamic mask schema")
        valid = valid & ~previous_dynamic
    aligned, overlap, transform = warp_bev(
        previous, previous_pose, current_pose, previous_valid=valid, **grid
    )
    if current_observed.shape != overlap.shape:
        raise ValueError("Current support shape mismatch")
    support = overlap & current_observed
    if current_dynamic is not None:
        if (
            current_dynamic.shape != support.shape
            or current_dynamic.dtype != torch.bool
        ):
            raise ValueError("Dynamic mask schema")
        support &= ~current_dynamic
    return aligned * support, support, transform
