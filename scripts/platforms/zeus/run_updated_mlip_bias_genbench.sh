#!/bin/bash
# Wait for final recovery evidence; take the shared GPU lock only while scoring.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
gpu=${MLIP_BIAS_GPU:-1}
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo 'Select GPU 0 or 1.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=online WANDB_ENTITY=symmetry-advantage
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MLIP_PREPROCESSOR_N_JOBS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/generated/mlip_bias_study/deps/genbench"
cpus=16-23
[[ "$gpu" == 0 ]] && cpus=8-15
exec taskset -c "$cpus" .venv/bin/python scripts/platforms/zeus/watch_mlip_bias_genbench.py --gpu "$gpu" "$@"
