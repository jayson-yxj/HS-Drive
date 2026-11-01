#!/usr/bin/env bash
set -eu
M8J12_ROOT=/root/gpufree-data/hs-drive-experiments/m8j12-sparsedrivev2
export PYTHONPATH="$PWD:$M8J12_ROOT/runtime-source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$M8J12_ROOT/env/lib/python3.8/site-packages/torch/lib:/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/cuda-11.8/lib"
exec unshare --net "$M8J12_ROOT/env/bin/python" "$@"
