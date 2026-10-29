#!/usr/bin/env bash
set -eu
M8J11_A=/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime
export PYTHONPATH="$PWD:$M8J11_A/sources/PETR:$M8J11_A/sources/mmdetection3d"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$M8J11_A/env/lib/python3.8/site-packages/torch/lib:$M8J11_A/cuda-11.8/lib"
exec unshare --net "$M8J11_A/env/bin/python" "$@"
