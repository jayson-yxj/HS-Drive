#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv-gpu
# PyPI 2.8.0 Linux wheel is CUDA 12.8; keep CPU data/simulator envs isolated.
.venv-gpu/bin/python -m pip install -r requirements-train.lock.txt
.venv-gpu/bin/python -m pip install --no-deps -e .
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/python -m tools.check_training_gpu --output reports/m4_gpu_recheck.json
