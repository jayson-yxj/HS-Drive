#!/usr/bin/env bash
set -euo pipefail
# Run only after all train/validation jobs and source review complete.
.venv-gpu/bin/python -m tools.m5i_freeze
.venv/bin/python -m tools.m5f_capture --protocol configs/m5i_capture_test.json --output /root/gpufree-data/hs-drive-data/raw/m5i_sealed > reports/m5i_logs/sealed_capture.log 2>&1
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5i_sealed --output /root/gpufree-data/hs-drive-data/processed/m5i_test_index --report reports/m5i_test_evidence_quality.json > reports/m5i_logs/sealed_evidence.log 2>&1
.venv/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5i_test_index --output /root/gpufree-data/hs-drive-data/processed/m5i_test_pixels --splits test > reports/m5i_logs/sealed_pixels.log 2>&1
.venv-gpu/bin/python -m tools.m5i_capture_audit --test > reports/m5i_logs/sealed_integrity.log 2>&1
.venv-gpu/bin/python -m tools.m5i_test_split_audit > reports/m5i_logs/sealed_split.log 2>&1
