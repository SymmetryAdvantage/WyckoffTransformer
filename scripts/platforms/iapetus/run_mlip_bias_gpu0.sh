#!/bin/bash
# Alternate ORB control, Prophet, and NequIP on GPU 0; they exceed VRAM together.
# Run from the WyFormer checkout, with no other job using GPU 0.
set -euo pipefail

if [[ ! -f generated/mlip_bias_study/input/pyxtal.extxyz ]]; then
    echo "Input artifact missing; download it through the study runner first" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export WYFORMER_CPUSET_CPUS=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
while true; do
    orb_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' generated/mlip_bias_study/orb_conserv_inf/trials.csv 2>/dev/null || true)
    prophet_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' generated/mlip_bias_study/prophet_oame_mbd/trials.csv 2>/dev/null || true)
    nequip_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' generated/mlip_bias_study/nequip_oam_xl/trials.csv 2>/dev/null || true)
    orb_rows=${orb_rows:-0}
    prophet_rows=${prophet_rows:-0}
    nequip_rows=${nequip_rows:-0}
    if (( orb_rows >= 2698 && prophet_rows >= 2698 && nequip_rows >= 2698 )); then
        break
    fi
    if (( orb_rows < 2698 )); then
        scripts/platforms/iapetus/run.sh python scripts/run_mlip_bias_relax.py \
            --input generated/mlip_bias_study/input \
            --output generated/mlip_bias_study/orb_conserv_inf \
            --mlip orb_conserv_inf --device cuda --relax-timeout 600 --max-trials 5 --sync-every 5
    fi
    if (( prophet_rows < 2698 )); then
        scripts/platforms/iapetus/run.sh bash -c \
            'PYTHONPATH=/workspace/scripts:/workspace/generated/mlip_bias_study/deps/prophet:/workspace/generated/mlip_bias_study/deps/tace /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/prophet_oame_mbd --mlip Prophet-OAME-MBD --calculator-factory mlip_bias_factories:build_prophet --checkpoint-id https://huggingface.co/kairosmaterial/prophet/resolve/f9a54df874ba52b8b35e7d3f7f348b46e0135563/prophet-oame-mbd.pt --device cuda --relax-timeout 600 --max-trials 5 --sync-every 5'
    fi
    if (( nequip_rows < 2698 )); then
        scripts/platforms/iapetus/run.sh bash -c \
            'PYTHONPATH=/workspace/scripts:/workspace/generated/mlip_bias_study/deps/nequip:/workspace/generated/mlip_bias_study/deps/tace /workspace/.venv/bin/python scripts/run_mlip_bias_relax.py --input generated/mlip_bias_study/input --output generated/mlip_bias_study/nequip_oam_xl --mlip Nequip-OAM-XL --calculator-factory mlip_bias_factories:build_nequip --checkpoint-id nequip.net:mir-group/NequIP-OAM-XL:0.1 --device cuda --relax-timeout 600 --max-trials 5 --sync-every 5'
    fi
done
