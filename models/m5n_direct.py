"""Opt-in RGB candidate center regression; GT has no entry into inference."""

import math
import numpy as np
import torch
from torch import nn
from models.m5l_p4 import sample_map


def region_geometry(component, q, K, T):
    x, y, w, h = component["bbox"]
    u = x + (w - 1) / 2
    v = y + (h - 1) / 2
    mask = component["mask"]
    area = float(mask.sum())
    return np.array(
        [
            (u - K[0, 2]) / K[0, 0],
            (v - K[1, 2]) / K[1, 1],
            math.log(w / 640),
            math.log(h / 384),
            area / (640 * 384),
            area / (w * h),
            float(q[mask].mean()),
            float(x <= 1 or y <= 1 or x + w >= 639 or y + h >= 383),
            K[0, 0] / 640,
            K[1, 1] / 384,
            K[0, 2] / 640,
            K[1, 2] / 384,
            *T[:3, :3].reshape(-1),
            *(T[:3, 3] / 3),
        ],
        np.float32,
    )


def feature_vector(feature, component, q, K, T):
    x, y, w, h = component["bbox"]
    u, v = np.meshgrid(np.linspace(x, x + w - 1, 3), np.linspace(y, y + h - 1, 3))
    uv = torch.tensor(
        np.stack([u.ravel(), v.ravel()], -1)[None],
        device=feature.device,
        dtype=torch.float32,
    )
    visual = sample_map(feature, uv, stride=8, center=0).flatten()
    geo = torch.tensor(region_geometry(component, q, K, T), device=feature.device)
    return torch.cat([visual, geo])


class DirectCenterHead(nn.Module):
    def __init__(self, features=600):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))

    def forward(self, features):
        if (
            features.ndim != 2
            or features.shape[1] != len(self.mean)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("legal feature tensor required")
        x = self.net((features.float() - self.mean) / self.scale)
        return torch.cat(
            [
                (x[:, :1] + math.log(10)).clamp(math.log(1), math.log(80)),
                2 * x[:, 1:].tanh(),
            ],
            -1,
        )


def reconstruct(prediction, regions, K, T):
    """FP32 optical-Z and projected-center offset to ego XYZ."""
    p = prediction.float()
    r = regions.float()
    z = p[:, 0].exp()
    uv = r[:, :2] + (r[:, 2:] - 1) / 2 + p[:, 1:] * r[:, 2:]
    ones = torch.ones_like(z)
    opt = (
        torch.linalg.solve(
            K.float(), torch.stack([uv[:, 0], uv[:, 1], ones], -1)[..., None]
        )[..., 0]
        * z[:, None]
    )
    return (T[:, :3, :3].float() @ opt[..., None])[..., 0] + T[:, :3, 3].float()


def output_contract(
    candidate_id, region, position, *, timestamp, proxy_error_m, threshold=0.0
):
    if not np.isfinite(timestamp):
        raise ValueError("timestamp")
    valid = position is not None and np.isfinite(position).all()
    confidence = (
        float(np.exp(-proxy_error_m / 5)) if np.isfinite(proxy_error_m) else 0.0
    )
    return dict(
        candidate_id=candidate_id,
        image_region=region,
        position_ego_xy=list(position[:2]) if valid else None,
        range_estimate=float(np.linalg.norm(position[:2])) if valid else None,
        localization_uncertainty=dict(
            proxy_xy_error_m=(
                float(proxy_error_m) if np.isfinite(proxy_error_m) else None
            ),
            confidence=confidence,
            guarantee=False,
        ),
        status=(
            "DetectedWithEstimate"
            if valid and confidence >= threshold
            else "DetectedNoLocation"
        ),
        timestamp=float(timestamp),
        class_hint="foreground_unknown_class",
        reference="ActorBoxCenter_research_not_contact",
    )
