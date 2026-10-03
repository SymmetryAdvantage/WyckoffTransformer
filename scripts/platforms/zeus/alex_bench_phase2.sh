#!/usr/bin/env bash
# Phase 2 of the alex-mp-20 benchmark study: the rules of engagement at each generator's
# chosen setting.
#
# Broadside and fire-control from one pool per setting; the fire-discipline cell is the
# setting's Phase 1 arm. The pool is sized so that fire-control keeps about the top 10%
# of unique novel genes -- the cut the production run will use. Settings run one after
# another in this lane. Usage:
#   nohup scripts/platforms/zeus/alex_bench_phase2.sh "name|run|flags" ... > log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDGET="${BUDGET:-2000}" ARMS="${ARMS:-broadside fire-control}" CPUS="${CPUS:-10}"
export POOL_SIZE="${POOL_SIZE:-31000}"

for spec in "$@"; do
    IFS='|' read -r name run flags <<< "$spec"
    SETTING="p2_$name" GEN_RUN="$run" GEN_ARGS="$flags" \
        "$HERE/run_alex_bench_setting.sh" || echo "[phase 2] $name FAILED"
done
echo "phase 2 lane done"
