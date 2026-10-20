"""New grouped input index, unchanged inference allowlist and independent V2 GT."""

import numpy as np
import torch
from datasets.occupancy_training import OccupancyDataset, checked
from datasets.training import DataItem
from datasets.m48_evidence import validate_evidence


class VisualEvidenceDataset(OccupancyDataset):
    def __init__(self, index, split, *, supervision=True):
        super().__init__(index, split, supervision=False)
        self.evidence_supervision = supervision

    def __getitem__(self, i):
        item = super().__getitem__(i)
        if not self.evidence_supervision:
            return item
        r = self.rows[i]
        with np.load(checked(r["label"], r["label_sha256"]), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        validate_evidence(d)
        reliable = (
            ~d["evidence_conflict_mask"] & ~d["semantic_conflict_mask"] & ~d["ego_mask"]
        )
        for k in [
            "surface_valid",
            "free_evidence_valid",
            "surface_semantic_valid",
            "obstacle_valid",
        ]:
            d[k] &= reliable
        return DataItem(
            item.inputs,
            {k: torch.from_numpy(v) for k, v in d.items()},
            dict(**item.metadata, configuration=r["configuration"]),
        )
