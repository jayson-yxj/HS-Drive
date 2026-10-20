#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
