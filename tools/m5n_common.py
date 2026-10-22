"""M5-N immutable study I/O and position-preserving data lineage."""

import json
from pathlib import Path
from tools.m5l_common import rows as lrows, write, asset, summary

ROOT = Path("/root/gpufree-data/hs-drive-data/processed")


def rows(split):
    if split in ["train", "val"]:
        return [dict(r, study_split=split) for r in lrows(split)] + [
            dict(r, study_split=split)
            for r in json.loads((ROOT / "m5m_dev_pixels/index.json").read_text())[
                "records"
            ]
            if r["split"] == split
        ]
    if split == "regression":
        rr = (
            lrows("regression")
            + lrows("test")
            + json.loads((ROOT / "m5m_sealed_pixels/index.json").read_text())["records"]
        )
        return [dict(r, study_split=split) for r in rr]
    if split == "sealed":
        return [
            dict(r, study_split=split)
            for r in json.loads((ROOT / "m5n_sealed_pixels/index.json").read_text())[
                "records"
            ]
        ]
    raise ValueError(split)


def stats(x):
    import numpy as np

    d = summary(x)
    if len(x):
        d["p90"] = float(np.quantile(x, 0.9))
    return d
