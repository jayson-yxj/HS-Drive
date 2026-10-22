"""Process successful versioned contact capture, preserving failed attempts."""

from tools.m5m_sequence import run


def main():
    run(
        [
            "tools.m5a_dataset",
            "--raw",
            "/root/gpufree-data/hs-drive-data/raw/m5m_dev_v3",
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
    run(["tools.m5m_new_eval", "--phase", "dev"], "dev_evaluation.log")


if __name__ == "__main__":
    main()
