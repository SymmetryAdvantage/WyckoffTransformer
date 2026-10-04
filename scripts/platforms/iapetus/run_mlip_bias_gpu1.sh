#!/bin/bash
# TECE on GPU 1, limited to two of iapetus's six physical CPU cores.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 WYFORMER_CPUSET_CPUS=2,3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec scripts/platforms/iapetus/run.sh bash -c \
    'TACE_USE_OEQ=1 TACE_USE_CUE=0 PYTHONPATH=/workspace/generated/mlip_bias_study/deps/tace /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/tece_oam_rra_1_0 --mlip TECE-OAM-RRA-1.0 --device cuda --relax-timeout 600 --sync-every 25'
