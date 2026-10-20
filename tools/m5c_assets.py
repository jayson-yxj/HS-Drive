"""Immutable old-assets check and new artifact inventory, including failed attempts."""

import json, subprocess
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_assets.json")
    if dest.exists():
        raise FileExistsError(dest)
    freeze = json.loads(Path("reports/m5c_baseline_freeze.json").read_text())
    changed = []
    for key in ["tracked_sha256", "checkpoint_sha256"]:
        for path, h in freeze[key].items():
            if not Path(path).exists() or file_hash(path) != h:
                changed.append(path)
    assets = {}
    roots = [
        Path("runs/m5c_learning"),
        Path("reports/m5c_val_predictions"),
        Path("reports/m5c_test_predictions"),
        *Path("/root/gpufree-data/hs-drive-data/raw").glob("m5c*"),
        *Path("/root/gpufree-data/hs-drive-data/processed").glob("m5c*"),
    ]
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                assets[str(p)] = dict(bytes=p.stat().st_size, sha256=file_hash(p))
    reports = {
        str(p): file_hash(p)
        for p in Path("reports").glob("m5c*")
        if p.is_file()
        and p.name not in ["m5c_assets.json", "m5c_acceptance.json", "m5c_summary.md"]
    }
    sources = {
        str(p): file_hash(p)
        for pat in [
            "tools/m5c*.py",
            "datasets/m5c*.py",
            "models/m5c*.py",
            "training/m5c*.py",
            "diagnostics/m5c*.py",
            "tests/test_m5c*.py",
            "configs/m5c*.json",
        ]
        for p in Path(".").glob(pat)
    }
    atomic_json(
        dest,
        dict(
            kind="actual_filesystem_sha256_audit",
            base_git_sha=freeze["base"],
            git_head_at_inventory=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            old_files_checked=len(freeze["tracked_sha256"])
            + len(freeze["checkpoint_sha256"]),
            old_files_changed=changed,
            assets=assets,
            reports_sha256=reports,
            source_sha256=sources,
            source="SHA256 inventory includes failedcapture dirs. Checkpoints/raw data stay server-local, not inline git blobs; inventory supports verification, not remote availability.",
        ),
    )
    print("old_changed", changed, "assets", len(assets), flush=True)


if __name__ == "__main__":
    main()
