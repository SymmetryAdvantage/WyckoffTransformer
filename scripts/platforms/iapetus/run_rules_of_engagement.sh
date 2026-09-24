#!/usr/bin/env bash
# Broadside, Fire Discipline and Fire Control over one shared gene pool, on iapetus.
#
# One pool, three selections, one reconstruction budget: the arms differ in what
# they select and in nothing else, and each reconstructs the same 1000 genes, so
# what separates them is the selection rather than how much was spent.
#
# Sequential by necessity, not by preference: the protocol's score stage holds
# the 5.3M-row reference at ~25 GB and this host has 30 GB.
#
# Everything reads models from disk; nothing writes to W&B.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUN="$HERE/run.sh"

BACKBONE="${BACKBONE:-/mnt/hdd/kna/wyformer/runs/unconditional_5x_ehull01-20260915-151250}"
REGRESSOR="${REGRESSOR:-/mnt/hdd/kna/wyformer/runs/min_energy_adamw_wsd-20260912-115957}"
ROOT="${ROOT:-/mnt/hdd/kna/wyformer/roe}"
POOL="$ROOT/pool/wyckoff_genes.json.gz"
BUDGET="${BUDGET:-1000}"
POOL_SIZE="${POOL_SIZE:-10000}"
# Draws per pool gene; generate keeps only formally valid genes, so a guided
# backbone (w = 3 is ~0.905 valid) needs more than the unconditional one's 1.1.
POOL_OVERSAMPLE="${POOL_OVERSAMPLE:-1.1}"
# Extra sampling flags for the pool, e.g.
#   POOL_GEN_ARGS="--condition energy_above_hull=0.05 --guidance-scale 3"
read -r -a POOL_GEN_ARGS <<< "${POOL_GEN_ARGS:-}"

# Two workers on each K20c and one on the 750 Ti; see docs/platforms/iapetus/usage.md.
PROTOCOL_ARGS=(--pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2
               --workers-per-device 1)

mkdir -p "$ROOT/pool"

if [[ ! -f "$POOL" ]]; then
    echo "=== drawing the shared pool of $POOL_SIZE genes ==="
    "$RUN" python -m wyckoff_transformer.cli.generate "$POOL" \
        --model-path "$BACKBONE" \
        --initial-n-samples "$(python3 -c "import math; print(math.ceil($POOL_SIZE * $POOL_OVERSAMPLE))")" \
        --firm-n-samples "$POOL_SIZE" \
        --device cpu "${POOL_GEN_ARGS[@]}"
fi

echo "=== broadside ==="
"$RUN" python -m wyckoff_transformer.roe.cli run broadside \
    --genes "$POOL" --output-dir "$ROOT/broadside" \
    --n-genes $((BUDGET + BUDGET / 10)) --target-engaged "$BUDGET" \
    -- "${PROTOCOL_ARGS[@]}"

echo "=== fire-discipline ==="
"$RUN" python -m wyckoff_transformer.roe.cli run fire-discipline \
    --genes "$POOL" --output-dir "$ROOT/fire-discipline" \
    --n-genes $((BUDGET * 2)) --target-engaged "$BUDGET" \
    -- "${PROTOCOL_ARGS[@]}"

echo "=== fire-control ==="
# No top-up: the whole pool is screened and the budget is spent on the genes with
# the lowest predicted e_hull.  A threshold at 0 keeps ~5% of novel genes, so
# filling a budget of 1000 from one would take tens of thousands of draws.
"$RUN" python -m wyckoff_transformer.roe.cli run fire-control \
    --genes "$POOL" --output-dir "$ROOT/fire-control" \
    --n-genes "$POOL_SIZE" \
    --regressor-path "$REGRESSOR" \
    --energy-select top --energy-top "$BUDGET" \
    -- "${PROTOCOL_ARGS[@]}"

echo "=== done ==="
for mode in broadside fire-discipline fire-control; do
    echo "--- $mode ---"
    "$RUN" python -m wyckoff_transformer.roe.cli report "$ROOT/$mode" || true
done
