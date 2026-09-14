#!/usr/bin/env bash
# Complete the cohorts that do not yet have processed stability outputs.
set -euo pipefail

for target in 0p025 0p05 0p1 0p2; do
    bash scripts/run_monitored_ehull_target.sh "$target"
done
