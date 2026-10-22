"""Stamped Actor snapshot and exact restored non-ego reference transforms."""

import json
from pathlib import Path
import numpy as np
from tools.m5n_common import write, asset, stats
from simulator.recorder import file_hash


def main():
    root = Path("/root/gpufree-data/hs-drive-data/raw/m5n_sealed")
    pairs = []
    bad = []
    n = 0
    for path in root.glob("*/manifest.json"):
        m = json.loads(path.read_text())
        assert file_hash(path.parent / "calibration.json") == m["calibration_sha256"]
        for fid, rid in m["references"].items():
            f = json.loads(
                (path.parent / "frames" / f"{int(fid):08d}.json").read_text()
            )
            r = json.loads((path.parent / "frames" / f"{rid:08d}.json").read_text())
            bi = {a["actor_id"]: a for a in r["actors"]}
            delta = max(
                [
                    float(
                        np.max(
                            abs(
                                np.array(a["world_matrix"])
                                - bi[a["actor_id"]]["world_matrix"]
                            )
                        )
                    )
                    for a in f["actors"]
                    if not a["is_ego"]
                ]
                or [0]
            )
            pairs.append(
                dict(group=m["group"], frame_id=int(fid), transform_delta=delta)
            )
            n += 1
            if delta >= 0.001:
                bad.append([m["group"], fid, "paired_pose"])
            for a in f["actors"]:
                if (
                    not a["actor_snapshot_present"]
                    or a["actor_snapshot_frame_id"] != f["frame_id"]
                    or abs(a["actor_snapshot_timestamp_s"] - f["timestamp_s"]) > 1e-5
                ):
                    bad.append([m["group"], fid, "actor_snapshot"])
            for name, cam in f["cameras"].items():
                for kind in ["rgb", "depth", "instance"]:
                    if (
                        cam[kind + "_frame_id"] != f["frame_id"]
                        or abs(cam[kind + "_timestamp_s"] - f["timestamp_s"]) > 1e-5
                        or file_hash(path.parent / cam[kind]) != cam[kind + "_sha256"]
                    ):
                        bad.append([m["group"], fid, name, kind])
    write(
        "reports/m5n_sealed_capture_quality.json",
        dict(
            kind="offline_actual_CARLA_sensor_snapshot_pair_integrity",
            frames=n,
            pairs=pairs,
            failures=bad,
            passed=not bad,
            limits="Actor snapshot exactly stamped; bbox localrotation/location/extents now recorded explicitly; reference remains simulation definition, not mesh/contact Gold.",
        ),
    )
    assert not bad, bad


if __name__ == "__main__":
    main()
