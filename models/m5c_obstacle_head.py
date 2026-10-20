"""Auxiliary independent visible-obstacle output; original V2 outputs retained."""

from torch import nn
from models.m48_evidence_head import EvidenceHead


class VisibleObstacleHead(EvidenceHead):
    def __init__(self, channels=128, width=64):
        super().__init__(channels, width)
        self.visible_obstacle = nn.Conv2d(width, 4, 1)

    def forward(self, bev):
        x = self.shared(bev)
        return dict(
            surface=self.surface(x),
            free=self.free(x),
            semantic=self.semantic(x).reshape(-1, 3, 4, *x.shape[-2:]),
            visible_obstacle=self.visible_obstacle(x),
        )
