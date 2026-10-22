"""New sealed CARLA positions only after final methods/calibration frozen."""

import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5n_develop import run


def main():
    for p, h in json.loads(Path("reports/m5n_pretest_freeze.json").read_text())[
        "sources"
    ].items():
        assert file_hash(p) == h, p
    run(
        ["tools.m5n_allocate", "--phase", "sealed"],
        "allocate_sealed.log",
        ".venv/bin/python",
    )
    run(
        [
            "tools.m5n_capture",
            "--protocol",
            "configs/m5n_capture_sealed.json",
            "--output",
            "/root/gpufree-data/hs-drive-data/raw/m5n_sealed",
        ],
        "capture_sealed.log",
        ".venv/bin/python",
    )
    run(["tools.m5n_capture_audit"], "capture_audit.log")
    run(
        [
            "tools.m5a_dataset",
            "--raw",
            "/root/gpufree-data/hs-drive-data/raw/m5n_sealed",
            "--output",
            "/root/gpufree-data/hs-drive-data/processed/m5n_sealed_index",
            "--report",
            "runs/m5n_logs/sealed_index_quality.json",
        ],
        "sealed_index.log",
        ".venv/bin/python",
    )
    run(
        [
            "tools.m5g_dataset",
            "--indices",
            "/root/gpufree-data/hs-drive-data/processed/m5n_sealed_index",
            "--output",
            "/root/gpufree-data/hs-drive-data/processed/m5n_sealed_pixels",
            "--splits",
            "test",
        ],
        "sealed_pixels.log",
        ".venv/bin/python",
    )
    run(["tools.m5n_cache", "--split", "sealed"], "cache_sealed.log")
    run(["tools.m5n_eval", "--split", "sealed"], "eval_sealed.log")


if __name__ == "__main__":
    main()
