#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# CPU-only M3 data validation; simulator environment stays separate.
python3.10 -m venv .venv-train
.venv-train/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if [ -f requirements-data.lock.txt ]; then
  .venv-train/bin/python -m pip install -r requirements-data.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
else
  .venv-train/bin/python -m pip install numpy==2.2.6 opencv-python-headless==5.0.0.93 PyYAML==6.0.3 pytest==9.1.1
fi
.venv-train/bin/python -m pip install --no-deps -e .
.venv-train/bin/python -m pip check
