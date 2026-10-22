"""Sequential data preparation and finite training; logs never overwritten."""

import subprocess
from pathlib import Path


def run(args, log, python=".venv-gpu/bin/python"):
    with Path("runs/m5n_logs", log).open("x") as f:
        subprocess.run(
            [python, "-m", *args], stdout=f, stderr=subprocess.STDOUT, check=True
        )


def main():
    run(["tools.m5n_cache", "--split", "val"], "cache_val.log")
    run(["tools.m5n_train"], "training.log")
    run(["tools.m5n_cache", "--split", "regression"], "cache_regression.log")


if __name__ == "__main__":
    main()
