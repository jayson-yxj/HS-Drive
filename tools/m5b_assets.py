"""Final immutable asset check and source/data/checkpoint hash manifest."""

import json, subprocess, platform
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def main():
    frozen = json.loads(Path("reports/m5b_asset_freeze.json").read_text())
    bad = []
    for key in ["tracked_sha256", "checkpoint_sha256"]:
        for name, sha in frozen[key].items():
            if not Path(name).exists() or file_hash(name) != sha:
                bad.append(name)
    prior = json.loads(Path(frozen["previous_external"]).read_text())
    for name, sha in prior["new_external_sha256"].items():
        if not Path(name).exists() or file_hash(name) != sha:
            bad.append(name)
    old = json.loads(Path(frozen["previous_data_stat"]).read_text())
    for name, stat in old["data_size_mtime"].items():
        p = Path(name)
        if not p.exists() or [p.stat().st_size, p.stat().st_mtime_ns] != stat:
            bad.append(name)
    hashes = {}
    for base in [
        "/root/gpufree-data/hs-drive-data/raw",
        "/root/gpufree-data/hs-drive-data/processed",
        "runs",
    ]:
        for root in Path(base).glob("m5b*"):
            for p in root.rglob("*"):
                if p.is_file() and "__pycache__" not in str(p):
                    hashes[str(p)] = file_hash(p)
    sources = {
        str(p): file_hash(p)
        for folder in [
            "tools",
            "models",
            "datasets",
            "diagnostics",
            "training",
            "tests",
            "configs",
        ]
        for p in Path(folder).glob("*m5b*")
        if p.is_file()
    }
    atomic_json(
        "reports/m5b_assets.json",
        dict(
            kind="actual_server_asset_audit",
            base=frozen["base"],
            mismatches=bad,
            old_tracked_count=len(frozen["tracked_sha256"]),
            old_checkpoint_count=len(frozen["checkpoint_sha256"]),
            source_sha256=sources,
            external_sha256=hashes,
            python=platform.python_version(),
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        ),
    )
    assert not bad, bad


if __name__ == "__main__":
    main()
