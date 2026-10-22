"""Ordered bounded study runner. Each log/artifact is append-only by name."""

import argparse, subprocess
from pathlib import Path


def run(module, log, python=".venv-gpu/bin/python"):
    path = Path("runs/m5m_logs") / log
    with path.open("x") as f:
        subprocess.run(
            [python, "-m", *module], stdout=f, stderr=subprocess.STDOUT, check=True
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    a = p.parse_args()
    if a.phase == "regression":
        for s in ["regression", "test"]:
            run(["tools.m5m_contact_audit", "--split", s], f"contact_{s}.log")
    elif a.phase == "dev":
        run(
            [
                "tools.m5m_capture",
                "--protocol",
                "configs/m5m_capture_dev.json",
                "--output",
                "/root/gpufree-data/hs-drive-data/raw/m5m_dev",
            ],
            "capture_dev.log",
            ".venv/bin/python",
        )
        run(
            [
                "tools.m5a_dataset",
                "--raw",
                "/root/gpufree-data/hs-drive-data/raw/m5m_dev",
                "--output",
                "/root/gpufree-data/hs-drive-data/processed/m5m_dev_index",
                "--report",
                "runs/m5m_logs/dev_index_quality.json",
            ],
            "dev_index.log",
            ".venv/bin/python",
        )
        run(
            [
                "tools.m5g_dataset",
                "--indices",
                "/root/gpufree-data/hs-drive-data/processed/m5m_dev_index",
                "--output",
                "/root/gpufree-data/hs-drive-data/processed/m5m_dev_pixels",
                "--splits",
                "train",
                "val",
            ],
            "dev_pixels.log",
            ".venv/bin/python",
        )


if __name__ == "__main__":
    main()
