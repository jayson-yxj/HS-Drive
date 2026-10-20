"""RGB-free per-cell prior comparator with the same four evidence tasks."""

import torch
from torch import nn


class ExistencePrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 24, 160, 120))

    def forward(self, bev):
        p = self.logits.expand(len(bev), -1, -1, -1)
        return dict(
            surface=p[:, :4],
            free=p[:, 4:8],
            semantic=p[:, 8:20].reshape(-1, 3, 4, 160, 120),
            visible_obstacle=p[:, 20:],
        )
