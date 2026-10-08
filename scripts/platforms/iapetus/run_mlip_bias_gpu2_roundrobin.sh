#!/bin/bash
# Alternate eSEN and EquiformerV3 on GPU 2; each needs most of its 2 GiB VRAM.
# Run from the WyFormer checkout, with no other job using GPU 2.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2 WYFORMER_CPUSET_CPUS=4,5
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
while true; do
    esen_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' generated/mlip_bias_study/esen_30m_oam/trials.csv 2>/dev/null || true)
    equiformer_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' generated/mlip_bias_study/equiformer_v3_dens_oam/trials.csv 2>/dev/null || true)
    esen_rows=${esen_rows:-0}
    equiformer_rows=${equiformer_rows:-0}
    if (( esen_rows >= 2698 && equiformer_rows >= 2698 )); then
        break
    fi
    if (( esen_rows < 2698 )); then
        scripts/platforms/iapetus/run.sh bash -c \
            'PYTHONPATH=/workspace/generated/mlip_bias_study/deps/fairchem:/workspace/generated/mlip_bias_study/deps/tace /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/esen_30m_oam --mlip eSEN-30M-OAM --device cuda --relax-timeout 600 --max-trials 10 --sync-every 10'
    fi
    if (( equiformer_rows < 2698 )); then
        scripts/platforms/iapetus/run.sh bash -c \
            'PYTHONPATH=/workspace/generated/mlip_bias_study/deps/equiformer_v3_src:/workspace/generated/mlip_bias_study/deps/fairchem:/workspace/generated/mlip_bias_study/deps/tace /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/equiformer_v3_dens_oam --mlip EquiformerV3+DeNS-OAM --device cuda --relax-timeout 600 --max-trials 10 --sync-every 10'
    fi
done
