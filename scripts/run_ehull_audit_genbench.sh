#!/usr/bin/env bash
# Score each e_hull target separately.  A benchmark pass uses exactly one
# 20-worker MACE relaxation pool, so cohorts never compete for CPU cores.
set -euo pipefail

audit_dir="generated/upi73i4k/ehull_conditioning_audit"
runner="generated/upi73i4k/genbench/run_genbench_parallel.py"
python_bin=".venv/bin/python"

for target in 0 0p025 0p05 0p1 0p2; do
    "$python_bin" "$runner" 20 -- \
        --cifs "$audit_dir/final_cifs_ehull_${target}.txt" \
        --config single_mlip \
        --name "wyformer_upi73i4k_ehull_${target}" \
        --mlip mace
done
