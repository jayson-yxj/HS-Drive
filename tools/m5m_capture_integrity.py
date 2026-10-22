"""Sensor joins, camera calibration and paired non-ego transforms audit."""

import argparse, json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset, summary
from simulator.recorder import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    a = p.parse_args()
    root = Path("/root/gpufree-data/hs-drive-data/raw") / (
        "m5m_dev_v3" if a.phase == "dev" else "m5m_sealed"
    )
    deltas = []
    frames = 0
    images = 0
    bad = []
    for path in root.glob("*/manifest.json"):
        man = json.loads(path.read_text())
        folder = path.parent
        assert file_hash(folder / "calibration.json") == man["calibration_sha256"]
        for fid, ref in man["references"].items():
            pair = [
                json.loads((folder / "frames" / f"{int(i):08d}.json").read_text())
                for i in [fid, ref]
            ]
            b = {r["actor_id"]: r for r in pair[1]["actors"]}
            for actor in pair[0]["actors"]:
                if not actor["is_ego"]:
                    deltas.append(
                        float(
                            np.max(
                                np.abs(
                                    np.array(actor["world_matrix"])
                                    - b[actor["actor_id"]]["world_matrix"]
                                )
                            )
                        )
                    )
            for frame in pair:
                frames += 1
                for camera, entry in frame["cameras"].items():
                    for kind in ["rgb", "depth", "instance"]:
                        images += 1
                        if (
                            entry[kind + "_frame_id"] != frame["frame_id"]
                            or abs(entry[kind + "_timestamp_s"] - frame["timestamp_s"])
                            > 1e-5
                            or file_hash(folder / entry[kind])
                            != entry[kind + "_sha256"]
                        ):
                            bad.append([frame["frame_id"], camera, kind])
    assert not bad
    assert (
        max(deltas, default=0) < 0.001
    ), "Paired actors moved beyond tolerance; invalidate self-mask reference"
    write(
        f"reports/m5m_capture_integrity_{a.phase}.json",
        dict(
            kind="offline_actual_CARLA_capture_integrity",
            frames_with_references=frames,
            images=images,
            non_ego_transform_max_abs_element=summary(deltas),
            pair_tolerance=0.001,
            mismatches=bad,
            manifest=asset(root / "manifest.json"),
            limits="Small deterministic re-settling differences bounded; matrix max combines rotation entries and translation meters, not standalone contact stability time series. No Gold claim.",
        ),
    )


if __name__ == "__main__":
    main()
