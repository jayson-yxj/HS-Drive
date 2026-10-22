"""Predeclared capture consistency exclusions; independent of RGB predictions."""

import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset
from simulator.recorder import file_hash


def main():
    root = Path("/root/gpufree-data/hs-drive-data/raw/m5m_sealed")
    bad = []
    pairs = []
    for p in root.glob("*/manifest.json"):
        m = json.loads(p.read_text())
        for fid, ref in m["references"].items():
            a = json.loads((p.parent / "frames" / f"{int(fid):08d}.json").read_text())
            b = json.loads((p.parent / "frames" / f"{int(ref):08d}.json").read_text())
            bi = {v["actor_id"]: v for v in b["actors"]}
            delta = max(
                [
                    float(
                        np.max(
                            abs(
                                np.array(v["world_matrix"])
                                - bi[v["actor_id"]]["world_matrix"]
                            )
                        )
                    )
                    for v in a["actors"]
                    if not v["is_ego"]
                ]
                or [0]
            )
            sid = m["group"] + "/" + f"{int(fid):08d}"
            pairs.append(
                dict(sample_id=sid, max_transform_delta=delta, valid=delta < 0.001)
            )
            if delta >= 0.001:
                bad.append(sid)
    index = Path(
        "/root/gpufree-data/hs-drive-data/processed/m5m_sealed_pixels/index.json"
    )
    original = index.with_name("index_unfiltered_preserved.json")
    if original.exists():
        raise FileExistsError(original)
    index.rename(original)
    d = json.loads(original.read_text())
    d["records"] = [r for r in d["records"] if r["sample_id"] not in bad]
    d["m5m_exclusions"] = bad
    d["unfiltered_index_sha256"] = file_hash(original)
    index.write_text(json.dumps(d, indent=2) + "\n")
    write(
        "reports/m5m_capture_integrity_sealed.json",
        dict(
            kind="actual_CARLA_pair_integrity_with_predeclared_exclusions",
            status="Partial",
            total=36,
            eligible=len(d["records"]),
            excluded=bad,
            pairs=pairs,
            tolerance=0.001,
            unfiltered_index=asset(original),
            evaluation_index=asset(index),
            limits="Two actor re-settling mismatches; failed frames preserved. Exclusion from main metrics independent of RGB output. No threshold tuning or recapture. Dataset gate remains Partial; claimed sealed results conditional on eligible34 frames.",
        ),
    )


if __name__ == "__main__":
    main()
