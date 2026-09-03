#!/usr/bin/env bash
# Materialize one lowest-energy CIF per CrySPR-successful gene for each target.
set -euo pipefail

audit_dir="generated/upi73i4k/ehull_conditioning_audit"
collector="generated/upi73i4k/genbench/collect_cifs.py"
python_bin=".venv/bin/python"

for target in 0 0p025 0p05 0p1 0p2; do
    "$python_bin" "$collector" "$audit_dir/cryspr_ehull_${target}" \
        "$audit_dir/final_cifs_ehull_${target}"
done
