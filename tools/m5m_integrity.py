"""Preserve historical assets; index final source, data, logs and all failed attempts."""

import json, subprocess
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5l_common import write, asset


def main():
    baseline = json.loads(Path("runs/m5m_logs/baseline.json").read_text())
    bad = []
    for group in ["tracked", "assets"]:
        for p, h in baseline[group].items():
            if not Path(p).exists() or file_hash(p) != h:
                bad.append(p)
    assert not bad, bad
    paths = []
    for root in [
        "models",
        "tools",
        "diagnostics",
        "simulator",
        "configs",
        "tests",
        "reports",
        "docs",
        "runs",
    ]:
        paths.extend(
            p
            for p in Path(root).rglob("*")
            if p.is_file() and "m5m" in str(p).lower() and "__pycache__" not in str(p)
        )
    for root in ["raw", "processed"]:
        for p in Path("/root/gpufree-data/hs-drive-data", root).glob("m5m*"):
            paths.extend(f for f in p.rglob("*") if f.is_file())
    index = {
        str(p): dict(sha256=file_hash(p), bytes=p.stat().st_size)
        for p in sorted(set(paths))
    }
    dst = "/root/gpufree-data/hs-drive-data/m5m_asset_sha_index.json"
    write(dst, dict(assets=index))
    write(
        "reports/m5m_artifact_manifest.json",
        dict(
            base=baseline["base"],
            assets=len(index),
            index=asset(dst),
            source_hashes={
                p: v["sha256"] for p, v in index.items() if p.endswith(".py")
            },
            model_hashes=json.loads(
                Path("reports/m5m_pretest_freeze.json").read_text()
            )["models"],
            tests=[
                asset("runs/m5m_logs/full_tests_delivery.log"),
                asset("runs/m5m_logs/sdk_tests.log"),
            ],
        ),
    )
    write(
        "reports/m5m_integrity.json",
        dict(
            historical_tracked=len(baseline["tracked"]),
            historical_assets=len(baseline["assets"]),
            changed=bad,
            base=baseline["base"],
            new_asset_index=asset(dst),
            source_commit="Final Git commit contains these source hashes; report itself cannot self-hash",
        ),
    )


if __name__ == "__main__":
    main()
