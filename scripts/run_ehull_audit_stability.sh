#!/usr/bin/env bash
# Recompute the actual MACE hull energies and SUN scores after the generic
# GenBench runner failed to retain its stability-preprocessor outputs.
# Exactly one 12-process, one-thread-per-process CPU pool is active at a time.
set -euo pipefail

audit_dir="generated/upi73i4k/ehull_conditioning_audit"
results_dir="/home/kna/sun-forest/external/lemat-genbench/results_final"
driver="scripts/recompute_relaxed_sun.py"
python_bin=".venv/bin/python"
out_dir="$audit_dir/stability"
mkdir -p "$out_dir"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

for target in 0 0p025 0p05 0p1 0p2; do
    result_json=$(find "$results_dir" -maxdepth 1 -type f \
        -name "wyformer_upi73i4k_ehull_${target}_mace_single_mlip_*.json" \
        -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    test -n "$result_json"

    "$python_bin" "$driver" --stage preprocess \
        --cif-list "$audit_dir/final_cifs_ehull_${target}.txt" \
        --results-json "$result_json" \
        --energies-csv "$out_dir/ehull_${target}_energies.csv" \
        --structures-pkl "$out_dir/ehull_${target}_processed.pkl" \
        --workers 12

    "$python_bin" "$driver" --stage sun --energy unrelaxed \
        --structures-pkl "$out_dir/ehull_${target}_processed.pkl" \
        --sun-json "$out_dir/ehull_${target}_sun_unrelaxed.json"
    "$python_bin" "$driver" --stage sun --energy relaxed \
        --structures-pkl "$out_dir/ehull_${target}_processed.pkl" \
        --sun-json "$out_dir/ehull_${target}_sun_relaxed.json"
done
