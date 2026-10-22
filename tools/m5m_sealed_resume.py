"""Resume after pair-integrity stop; excluded samples preserved before inference."""

from pathlib import Path
from tools.m5m_sequence import run


def main():
    assert Path(
        "/root/gpufree-data/hs-drive-data/processed/m5m_sealed_index/index.json"
    ).exists()
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
    run(["tools.m5m_sealed_exclusions"], "sealed_exclusions.log")
    # GPU inference invoked manually only after shadow completes, avoiding GPU contamination.


if __name__ == "__main__":
    main()
