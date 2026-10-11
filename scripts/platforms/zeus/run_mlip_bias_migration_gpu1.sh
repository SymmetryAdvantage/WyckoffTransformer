#!/bin/bash
# Sequential migration/recovery on an explicitly selected Zeus GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mkdir -p generated/mlip_bias_study
study=generated/mlip_bias_study
gpu=${MLIP_BIAS_GPU:-1}
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo 'Select GPU 0 or 1.' >&2; exit 1; }
exec 9>"$study/zeus_gpu${gpu}.lock"
if [[ "${1:-}" == --wait-for-lock ]]; then
    shift
    flock 9
    # This marker belongs to the old all-arms queue being replaced. Move it
    # only after that writer released its lock and uploaded its active batch.
    if [[ "$gpu" == 1 && -e "$study/zeus_migration/repartition_requested" ]]; then
        mv "$study/zeus_migration/stop" "$study/setup/zeus_migration_partition_stop_20261006"
        mv "$study/zeus_migration/repartition_requested" "$study/setup/zeus_migration_partition_request_20261006"
    fi
else
    flock -n 9 || { echo "GPU $gpu study lock is already held." >&2; exit 1; }
fi
export MLIP_BIAS_GPU="$gpu" CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=online WANDB_ENTITY=symmetry-advantage
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cpus=16-23
[[ "$gpu" == 0 ]] && cpus=8-15
exec taskset -c "$cpus" .venv/bin/python scripts/platforms/zeus/run_mlip_bias_migration_gpu1.py "$@"
