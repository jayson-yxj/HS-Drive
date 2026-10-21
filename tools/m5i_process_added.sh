#!/usr/bin/env bash
set -euo pipefail
# Require complete campaigns; never index partially recorded scenes.
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5i_added_train /root/gpufree-data/hs-drive-data/raw/m5i_added_val --output /root/gpufree-data/hs-drive-data/processed/m5i_added_index --report reports/m5i_added_evidence_quality.json
.venv/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5i_added_index --output /root/gpufree-data/hs-drive-data/processed/m5i_added_pixels --splits train val
.venv-gpu/bin/python -m tools.m5i_data
.venv-gpu/bin/python -m tools.m5i_coverage
.venv-gpu/bin/python -m tools.m5i_split_audit
.venv-gpu/bin/python -m tools.m5f_cache --index /root/gpufree-data/hs-drive-data/processed/m5i_added_pixels --output runs/m5i_added_cache
.venv-gpu/bin/python -m tools.m5i_teacher --root /root/gpufree-data/hs-drive-data/processed/m5i_added_pixels
