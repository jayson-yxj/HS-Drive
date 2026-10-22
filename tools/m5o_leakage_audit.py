"""Audit frozen-model lineage, real split identities, and inference/GT path isolation."""

import ast, json
from pathlib import Path
import numpy as np
from tools.m5o_common import DATA, write, asset, provenance


def main():
    cfg = json.loads(Path("configs/m5o_capture_dev_v3.json").read_text())
    records = []
    old = []
    # Historical model dataset may cover moving roads beyond static capture protocols.
    man = json.loads(Path("runs/m4_expanded_joint/manifest.json").read_text())
    index = json.loads(Path(man["config"]["index"], "index.json").read_text())
    for g in cfg["groups"]:
        positions = []
        for scene in index["scenes"]:
            if scene["map"].split("/")[-1] == g["map"]:
                positions.extend((np.array(scene["positions_xy"]) * [1, -1]).tolist())
        dist = min(
            [float(np.linalg.norm(np.array(g["pose"][:2]) - v)) for v in positions]
            or [1e9]
        )
        records.append(
            dict(
                group=g["group"],
                split=g["split"],
                map=g["map"],
                min_M4_training_path_m=None if not positions else dist,
                historical_encoder_geography_overlap=bool(dist < 45),
            )
        )
    actual = []
    groupsets = {}
    sample_sets = {}
    failures = []
    for split in ["train", "val", "calibration", "sealed"]:
        path = DATA / "processed" / f"m5o_{split}_pixels/index.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["records"]
        groupsets[split] = {r["group"] for r in rows}
        sample_sets[split] = {r["sample_id"] for r in rows}
        for r in rows:
            inp = json.loads(Path(r["input"]).read_text())
            allowed = {
                "sample_id",
                "images",
                "intrinsics",
                "cam_to_ego",
                "ego_pose",
                "timestamp_s",
            }
            if set(inp) != allowed:
                failures.append([r["sample_id"], "input_keys"])
            if r["split"] != split:
                failures.append([r["sample_id"], "split"])
            if not all(x["path"].endswith("_rgb.png") for x in inp["images"]):
                failures.append([r["sample_id"], "rgb_path"])
        actual.append(
            dict(
                split=split,
                frames=len(rows),
                groups=sorted(groupsets[split]),
                index=asset(path),
            )
        )
    for a in groupsets:
        for b in groupsets:
            if a >= b:
                continue
            if groupsets[a] & groupsets[b] or sample_sets[a] & sample_sets[b]:
                failures.append([a, b, "identity_overlap"])
    tree = ast.parse(Path("models/m5n_direct.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"carla", "actor_reference", "pixel_label"}
    write(
        "reports/m5o_leakage_audit.json",
        dict(
            kind="offline_actual_manifest_and_source_audit",
            failures=failures,
            actual_splits=actual,
            legacy_encoder_geography=records,
            inference_gt_entrypoints=False,
            provenance=provenance(),
            limits="New regressors use only new Train positions. Frozen Mobile/Depth have historical training geography; min historical M4 path distance disclosed rather than claiming globally unseen encoder geography. History K/L/M/N tests regression-only. Split isolation refers to new task model selection/calibration, not all pretraining data.",
        ),
    )
    assert not failures, failures


if __name__ == "__main__":
    main()
