"""Manifest immutable external assets and verify every pre-existing tracked file."""

import json, subprocess, hashlib
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5d_assets.json")
    if dest.exists():
        raise FileExistsError(dest)
    baseline = json.loads(Path("reports/m5d_baseline_freeze.json").read_text())
    old = []
    for section in ["tracked_sha256", "checkpoint_sha256"]:
        for p, sha in baseline[section].items():
            if not Path(p).is_file() or file_hash(p) != sha:
                old.append(dict(section=section, path=p))
    external = []
    for root in [
        "runs/m5d_learning",
        "reports/m5d_test_predictions",
        "reports/m5d_val_predictions",
        "/root/gpufree-data/hs-drive-data/raw/m5d_capture",
        "/root/gpufree-data/hs-drive-data/raw/m5d_sealed",
        "/root/gpufree-data/hs-drive-data/processed/m5d_index",
        "/root/gpufree-data/hs-drive-data/processed/m5d_test_index",
    ]:
        for p in sorted(Path(root).rglob("*")):
            if p.is_file():
                external.append(
                    dict(path=str(p), bytes=p.stat().st_size, sha256=file_hash(p))
                )
    source = {
        str(p): file_hash(p)
        for base in [
            "tools",
            "datasets",
            "models",
            "training",
            "diagnostics",
            "tests",
            "configs",
        ]
        for p in Path(base).glob("*m5d*")
        if p.is_file()
    }
    reports = {
        str(p): file_hash(p)
        for p in Path("reports").glob("m5d*")
        if p.is_file()
        and p.name not in ["m5d_assets.json", "m5d_acceptance.json", "m5d_summary.md"]
    }
    logs = {
        str(p): file_hash(p) for p in Path("reports/m5d_logs").glob("*") if p.is_file()
    }
    atomic_json(
        dest,
        dict(
            kind="actual_filesystem_asset_integrity_audit",
            base=baseline["base"],
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            source_sha256=source,
            reports_sha256=reports,
            logs_sha256=logs,
            external_assets=external,
            old_asset_changes=old,
            old_assets_preserved=not old,
            limits="Checkpoint/data/prediction binaries remain on server; Git carries source,reports,figures,tests,logs and manifests. External assets hash each byte; historical external data preservation additionally follows untouched source pipelines and original manifests, no full rehash of every old raw frame. Manifest source_commit precedes final report commit; reports cite this explicitly.",
        ),
    )


if __name__ == "__main__":
    main()
