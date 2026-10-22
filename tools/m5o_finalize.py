"""Freeze measured M5-O artifacts and verify historical assets are byte-identical."""

import json, subprocess
from pathlib import Path
from tools.m5o_common import DATA, write, asset, provenance
from simulator.recorder import file_hash


def main():
    baseline = json.loads(Path("runs/m5o_logs/historical_freeze.json").read_text())
    bad = []
    for kind in ["tracked", "assets"]:
        for p, h in baseline[kind].items():
            if not Path(p).is_file() or file_hash(p) != h:
                bad.append(dict(path=p, kind=kind))
    tests = Path("runs/m5o_logs/full_tests_final.log").read_text()
    if "failed" in tests or "passed" not in tests:
        raise RuntimeError("tests not passed")
    if bad:
        raise RuntimeError(f"historical mutation {bad}")
    paths = set()
    for root in [
        "models",
        "datasets",
        "diagnostics",
        "tools",
        "tests",
        "configs",
        "docs",
        "reports",
        "runs",
    ]:
        paths.update(
            p
            for p in Path(root).rglob("*")
            if p.is_file() and "m5o" in str(p).lower() and "__pycache__" not in str(p)
        )
    for root in ["raw", "processed"]:
        for directory in (DATA / root).glob("m5o*"):
            if not directory.is_symlink():
                paths.update(
                    p
                    for p in directory.rglob("*")
                    if p.is_file() and not p.is_symlink()
                )
    ix = {
        str(p): dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in sorted(paths)
    }
    dest = DATA / "m5o_asset_sha_index.json"
    write(dest, dict(assets=ix))
    write(
        "reports/m5o_artifact_manifest.json",
        dict(
            provenance=provenance(),
            index=asset(dest),
            assets=len(ix),
            historical_tracked=len(baseline["tracked"]),
            historical_assets=len(baseline["assets"]),
            historical_changed=bad,
            tests=asset("runs/m5o_logs/full_tests_final.log"),
            models={p: v for p, v in ix.items() if p.endswith(".pt")},
            limits="Large raw data/checkpoints/logs retained locally; Git contains source/config/reports/figures. Index SHA authenticates local asset paths; pushing Git does not upload raw datasets.",
        ),
    )
    print(
        "Preserved",
        len(baseline["tracked"]),
        len(baseline["assets"]),
        "new assets",
        len(ix),
    )


if __name__ == "__main__":
    main()
