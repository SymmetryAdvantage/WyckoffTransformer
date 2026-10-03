#!/usr/bin/env bash
# Production run of the alex-mp-20 benchmark submission, on zeus.
#
# One arm of the study driver at submission scale -- the same pool sampling, selection,
# DiffCSP++, rattle and ORB evaluation the study measured -- then the assembler, which
# writes the first N structures that pass the start checks in the arm's own order.
#
# The pool is sized so that the selection keeps the same fraction of unique novel genes as
# the study's 2000-of-31000 arms did: the selection fraction is part of what fire-control
# measured, and a sharper or blunter cut would be a different, unmeasured rule.
#
# Environment: SETTING_NAME, GEN_RUN, GEN_ARGS, MODE (default fire-control),
#   N (default 10000), BUDGET (default 10400), POOL_SIZE (default 165000).
set -euo pipefail

: "${SETTING_NAME:?}" "${GEN_RUN:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
MODE="${MODE:-fire-control}"
N="${N:-10000}"
export BUDGET="${BUDGET:-10400}" POOL_SIZE="${POOL_SIZE:-165000}" ARMS="$MODE"
export SETTING="production_$SETTING_NAME" GEN_RUN GEN_ARGS="${GEN_ARGS:-}" CPUS="${CPUS:-20}"

"$HERE/run_alex_bench_setting.sh"

STORE="$(dirname "$("$REPO/.venv/bin/python" -c 'from wyckoff_transformer.paths import runs_root; print(runs_root())')")"
ARM="$STORE/alex_bench/$SETTING/$MODE"
"$REPO/.venv/bin/python" "$REPO/scripts/alex_bench/assemble_submission.py" \
    "$ARM" "$STORE/alex_bench/submission_$SETTING_NAME" --n "$N"
