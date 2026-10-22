"""M5-O immutable I/O and provenance. Historical readers remain unchanged."""

import json
import subprocess
from pathlib import Path
from tools.m5l_common import write, asset
from simulator.recorder import file_hash

BASE = "2afe740be0cf8e0df8dff52a7ed2b03a6f001dc5"
DATA = Path("/root/gpufree-data/hs-drive-data")
SPLITS = ("train", "val", "calibration", "sealed")


def rows(split):
    if split == "regression":
        from tools.m5n_common import rows as oldrows

        return oldrows("sealed")
    if split in ["smoke", "smoke_v3"]:
        return json.loads(
            (DATA / f"processed/m5o_{split}_pixels/index.json").read_text()
        )["records"]
    if split not in SPLITS:
        raise ValueError(split)
    return json.loads(
        (DATA / "processed" / f"m5o_{split}_pixels/index.json").read_text()
    )["records"]


def provenance():
    paths = sorted(
        p
        for folder in ["tools", "datasets", "diagnostics", "tests", "configs"]
        for p in Path(folder).glob("*m5o*")
        if p.is_file()
    )
    return dict(
        base=BASE,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        sources={str(p): file_hash(p) for p in paths},
    )
