#!/bin/bash
# TACE on the K20c freed by TECE, using the original uncompiled backend.
# Pass --retry-failed once to requeue the preserved first-pass failures.
# On an interrupted retry pass, resume without that flag.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 WYFORMER_CPUSET_CPUS=2,3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec scripts/platforms/iapetus/run.sh bash -c \
    'TACE_USE_OEQ=0 TACE_USE_CUE=0 PYTHONPATH=/workspace/generated/mlip_bias_study/deps/tace exec /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/tace_oam_l --mlip TACE-OAM-L --device cuda --relax-timeout 600 --sync-every 25 "$@"' bash "$@"
