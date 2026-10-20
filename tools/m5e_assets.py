"""Manifest immutable external assets and verify every pre-existing tracked file."""

import json, subprocess, hashlib
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5e_assets.json")
    if dest.exists():
        raise FileExistsError(dest)
    baseline = json.loads(Path("reports/m5e_baseline_freeze.json").read_text())
    old = []
    for section in ["tracked_sha256", "checkpoint_sha256"]:
        for p, sha in baseline[section].items():
            if not Path(p).is_file() or file_hash(p) != sha:
                old.append(dict(section=section, path=p))
    external = []
    for root in [
        "runs/m5e_learning",
        "reports/m5e_test_predictions",
        "reports/m5e_val_predictions",
        "reports/m5e_witness_attrition_rays",
        "reports/m5e_new_visibility_rays",
        "/root/gpufree-data/hs-drive-data/raw/m5e_capture",
        "/root/gpufree-data/hs-drive-data/raw/m5e_sealed",
        "/root/gpufree-data/hs-drive-data/processed/m5e_index",
        "/root/gpufree-data/hs-drive-data/processed/m5e_test_index",
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
            "simulator",
        ]
        for p in Path(base).glob("*m5e*")
        if p.is_file()
    }
    reports = {
        str(p): file_hash(p)
        for p in Path("reports").rglob("*")
        if p.is_file() and any(part.startswith("m5e") for part in p.parts) and "predictions" not in str(p)
        and p.name not in ["m5e_assets.json", "m5e_acceptance.json", "m5e_summary.md"]
    }
    logs = {
        str(p): file_hash(p) for p in Path("reports/m5e_logs").glob("*") if p.is_file()
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
