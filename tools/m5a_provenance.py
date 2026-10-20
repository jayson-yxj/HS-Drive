"""Preserve prior assets and quantify historical route overlap, no split relabeling."""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    freeze = json.loads(Path("reports/m5a_asset_freeze.json").read_text())
    bad = []
    counts = {}
    for key in ["tracked_sha256", "checkpoint_sha256", "external_m48"]:
        for p, sha in freeze[key].items():
            if not Path(p).exists() or file_hash(p) != sha:
                bad.append([key, p])
        counts[key] = len(freeze[key])
    for p, ref in freeze["data_size_mtime"].items():
        if not Path(p).exists():
            bad.append(["missing", p])
            continue
        s = Path(p).stat()
        if [s.st_size, s.st_mtime_ns] != ref:
            bad.append(["old_data_stat", p])
    counts["old_data_stat"] = len(freeze["data_size_mtime"])
    protocol = json.loads(Path("configs/m5a_protocol.json").read_text())
    old = json.loads(
        Path(
            "/root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted/index.json"
        ).read_text()
    )
    overlap = []
    for g in protocol["groups"]:
        xy = np.array([g["pose"][0], -g["pose"][1]])
        dist = []
        for scene in old["scenes"]:
            if scene["map"] == g["map"]:
                distance = float(
                    np.linalg.norm(np.asarray(scene["positions_xy"]) - xy, axis=1).min()
                )
                dist.append(
                    dict(
                        scene=scene["scene_id"],
                        split=scene["split"],
                        distance_m=distance,
                    )
                )
        overlap.append(
            dict(
                group=g["group"],
                split=g["split"],
                historical_m4_routes=sorted(dist, key=lambda x: x["distance_m"]),
            )
        )
    current = json.loads(
        Path(
            "/root/gpufree-data/hs-drive-data/processed/m5a_index/index.json"
        ).read_text()
    )
    group_dist = []
    for a in protocol["groups"]:
        for b in protocol["groups"]:
            if (
                a["map"] == b["map"]
                and a["split"] != b["split"]
                and a["group"] < b["group"]
            ):
                group_dist.append(
                    dict(
                        a=a["group"],
                        b=b["group"],
                        distance_m=float(
                            np.linalg.norm(np.array(a["pose"][:2]) - b["pose"][:2])
                        ),
                    )
                )
    new = {}
    for folder in [
        "/root/gpufree-data/hs-drive-data/raw/m5a_pilot_capture",
        "/root/gpufree-data/hs-drive-data/raw/m5a_capture",
        "/root/gpufree-data/hs-drive-data/processed/m5a_index",
        "runs/m5a_learning",
    ]:
        for p in Path(folder).rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                new[str(p)] = file_hash(p)
    atomic_json(
        "reports/m5a_dataset_manifest.json",
        dict(
            samples=current["samples"],
            group_separation=group_dist,
            historical_route_overlap=overlap,
            scope="Held out from M5-A head fitting, not guaranteed novel to pretrained M4 Encoder. Historical route proximity explicitly audited. No new-map claim.",
            protocol_sha256=file_hash("configs/m5a_protocol.json"),
        ),
    )
    atomic_json(
        "reports/m5a_asset_verification.json",
        dict(
            kind="actual_server_filesystem_hash_and_stat_audit",
            base=freeze["commit"],
            counts=counts,
            mismatches=bad,
            old_assets_preserved=not bad,
            new_external_sha256=new,
        ),
    )
    print(counts, "mismatches", bad, flush=True)
    for r in overlap:
        if r["split"] == "test":
            print(r["group"], r["historical_m4_routes"][:2])
    assert not bad


if __name__ == "__main__":
    main()
