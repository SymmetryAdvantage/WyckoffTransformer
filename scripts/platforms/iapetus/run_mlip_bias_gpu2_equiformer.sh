#!/bin/bash
# EquiformerV3 on GPU 2, limited to two of iapetus's six physical CPU cores.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2 WYFORMER_CPUSET_CPUS=4,5
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec scripts/platforms/iapetus/run.sh bash -c \
    'PYTHONPATH=/workspace/generated/mlip_bias_study/deps/equiformer_v3_src:/workspace/generated/mlip_bias_study/deps/fairchem:/workspace/generated/mlip_bias_study/deps/tace /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/equiformer_v3_dens_oam --mlip EquiformerV3+DeNS-OAM --device cuda --relax-timeout 600 --sync-every 25'
