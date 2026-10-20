"""Finite actual CPU visibility-label processing load; never calls CARLA."""

import argparse, json, time
from pathlib import Path
from tools.m48_spatial import frame_rays
from datasets.m48_evidence import build_evidence
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=100)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    rows = json.loads(
        Path(
            "/root/gpufree-data/hs-drive-data/processed/m5c_index/index.json"
        ).read_text()
    )["samples"]
    start = time.perf_counter()
    times = []
    i = 0
    while time.perf_counter() - start < a.duration:
        row = rows[i % len(rows)]
        raw = Path(row["raw_frame"]).parent.parent
        f = json.loads(Path(row["raw_frame"]).read_text())
        man = json.loads((raw / "manifest.json").read_text())
        man["counterfactual_frame"] = man["references"][str(f["frame_id"])]
        t = time.perf_counter()
        r, box = frame_rays(raw, man, f["frame_id"])
        uv = r["uv"]
        sparse = (uv % 16 == 0).all(-1)
        held = (uv[:, 0] + 2 * uv[:, 1]) % 4 == 1
        build_evidence(
            r["origins"][sparse],
            r["endpoints"][sparse],
            r["endpoints"][~held],
            r["semantics"][~held],
            ego_box=box,
        )
        times.append(time.perf_counter() - t)
        i += 1
    atomic_json(
        a.report,
        dict(
            kind="actual_CPU_visibility_and_V2_label_generation_no_mutation",
            seconds=time.perf_counter() - start,
            frames=i,
            frame_seconds=times,
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
