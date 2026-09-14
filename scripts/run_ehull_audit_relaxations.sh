#!/usr/bin/env bash
# Finish the unrelaxed cohorts in the upi73i4k e_hull-conditioning audit.
# Runs strictly one 20-process CrySPR pool at a time; each worker clamps its
# BLAS/OpenMP/Torch threads to one, so this never exceeds 20 CPU cores.
set -euo pipefail

audit_dir="generated/upi73i4k/ehull_conditioning_audit"
driver="generated/upi73i4k/genbench/relax_remaining.py"
python_bin=".venv/bin/python"

for target in 0p05 0p1 0p2; do
    "$python_bin" "$driver" "$audit_dir/wyckoff_genes_ehull_${target}.json.gz" \
        --output-dir "$audit_dir/cryspr_ehull_${target}" \
        --model MACE-MP-0a-small \
        --model-name MACE-MP-0a-small \
        --workers 20 \
        --n-trials 3 \
        --fmax 0.05
done
