"""One fresh sealed run after source and validation locks; no parameter selection."""

import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5m_sequence import run


def main():
    freeze = json.loads(Path("reports/m5m_pretest_freeze.json").read_text())
    for p, h in freeze["sources"].items():
        assert file_hash(p) == h, p
    run(
        ["tools.m5m_allocate", "--phase", "sealed"],
        "allocate_sealed.log",
        ".venv/bin/python",
    )
    run(
        [
            "tools.m5m_capture",
            "--protocol",
            "configs/m5m_capture_sealed.json",
            "--output",
            "/root/gpufree-data/hs-drive-data/raw/m5m_sealed",
        ],
        "capture_sealed.log",
        ".venv/bin/python",
    )
    run(
        [
            "tools.m5a_dataset",
            "--raw",
            "/root/gpufree-data/hs-drive-data/raw/m5m_sealed",
            "--output",
            "/root/gpufree-data/hs-drive-data/processed/m5m_sealed_index",
            "--report",
            "runs/m5m_logs/sealed_index_quality.json",
        ],
        "sealed_index.log",
        ".venv/bin/python",
    )
    run(
        [
            "tools.m5g_dataset",
            "--indices",
            "/root/gpufree-data/hs-drive-data/processed/m5m_sealed_index",
            "--output",
            "/root/gpufree-data/hs-drive-data/processed/m5m_sealed_pixels",
            "--splits",
            "test",
        ],
        "sealed_pixels.log",
        ".venv/bin/python",
    )
    run(["tools.m5m_new_eval", "--phase", "sealed"], "sealed_evaluation.log")
    run(["tools.m5m_anatomy_audit", "--phase", "sealed"], "anatomy_sealed.log")


if __name__ == "__main__":
    main()
