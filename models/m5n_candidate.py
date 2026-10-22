"""Offline-only opt-in six-RGB actor-center estimator, no CARLA/control access."""

import json
from pathlib import Path
import torch, numpy as np
from torch import nn
from datasets.inputs import validate_tensor_inputs
from tools.m5i_data import load
from models.m5j_candidate import validate_packet, CAMERAS
from models.m5n_direct import (
    DirectCenterHead,
    feature_vector,
    reconstruct,
    output_contract,
)
from diagnostics.m5l_objects import components
from diagnostics.m5n_metrics import risk_features, predict_risk
from simulator.recorder import file_hash


class DirectActorCandidate(nn.Module):
    def __init__(self, *, enable=False, seed=101):
        super().__init__()
        if not enable:
            raise ValueError("Offline research opt-in required")
        cfg = json.loads(Path("configs/m5j_models.json").read_text())
        rec = next(
            r for r in cfg["pixel"] if r["name"] == "mobile_gt" and r["seed"] == seed
        )
        self.mobile, _ = load(rec)
        train = json.loads(Path("reports/m5n_direct_range_training.json").read_text())
        rec = next(
            r for r in train["records"] if r["seed"] == seed and r["arm"] == "direct"
        )
        assert file_hash(rec["selected"]["path"]) == rec["selected"]["sha256"]
        ck = torch.load(rec["selected"]["path"], weights_only=True)
        self.head = DirectCenterHead().cuda()
        self.head.load_state_dict(ck["model"])
        lock = json.loads(Path("reports/m5n_validation_lock.json").read_text())
        self.cal = lock["calibration"][str(seed)]
        self.threshold = lock["threshold"]
        self.qualified = lock["promoted"]
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, *, now_s=None, camera_names=CAMERAS, **inputs):
        validate_packet(inputs, camera_names, now_s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = self.mobile.fuse(self.mobile.levels(inputs["images"]))
            q = self.mobile.head(f, (384, 640)).float().sigmoid()
        q = (
            (torch.clamp((q * 256).floor(), max=255) / 256)
            .reshape(6, 384, 640)
            .cpu()
            .numpy()
        )
        features = []
        records = []
        for ci, cam in enumerate(camera_names):
            K = inputs["intrinsics"][0, ci].cpu().numpy()
            T = inputs["cam_to_ego"][0, ci].cpu().numpy()
            for i, c in enumerate(components(q[ci])):
                features.append(feature_vector(f[ci : ci + 1], c, q[ci], K, T))
                records.append(
                    dict(
                        candidate_id=f"{cam}:{i}",
                        region=c["bbox"],
                        pixel_area=int(c["mask"].sum()),
                        q_mean=float(q[ci][c["mask"]].mean()),
                        K=K,
                        T=T,
                    )
                )
        if not records:
            return []
        pred = self.head(torch.stack(features))
        regions = torch.tensor([r["region"] for r in records], device=f.device)
        K = torch.tensor(np.stack([r["K"] for r in records]), device=f.device)
        T = torch.tensor(np.stack([r["T"] for r in records]), device=f.device)
        xyz = reconstruct(pred, regions, K, T).cpu().numpy()
        rx = [risk_features(r, p) for r, p in zip(records, xyz)]
        risk = predict_risk(self.cal, rx)
        out = []
        for r, p, e in zip(records, xyz, risk):
            o = output_contract(
                r["candidate_id"],
                r["region"],
                p,
                timestamp=float(inputs["timestamps_s"][0]),
                proxy_error_m=e,
                threshold=self.threshold,
            )
            o["localization_uncertainty"]["empirical90_radius_m"] = float(
                e * self.cal["empirical90_multiplier"]
            )
            o["research_qualified"] = self.qualified
            out.append(o)
        return out
