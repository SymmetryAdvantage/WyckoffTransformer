#!/usr/bin/env bash
# Sample an alex-mp-20 benchmark submission the way the WyFormer-GeoCSP-*-v2.3
# submissions were made, without the ORB evaluation the study used to choose them.
#
#   genes:     WyFormer (one of three generators) samples a pool of Wyckoff genes
#   filtering: fire-control -- drop duplicates and genes already in alex-mp-20, rank the
#              rest by predicted e_hull and keep the best ~50%
#   GeoCSP:    one 3D structure per kept gene (symmetry-projected diffusion)
#   submit:    check, rattle, keep the first N in predicted-e_hull order, write CIFs
#
# The models are stochastic and floating-point order is not fixed, so a run reproduces the
# recipe, not the published structures. docs/archive/okhotin_submission.md is the record.
#
# Usage: sample_submission.sh LABEL OUT_DIR [N]
#   LABEL  CFG | CON | UC  (WyFormer-GeoCSP-{CFG,CON,UC}-v2.3); N defaults to 10000.
#
# Environment (defaults are the container's layout):
#   WYFORMER_PYTHON   interpreter with wyckoff_transformer     (/opt/wyformer/.venv/bin/python)
#   GEOCSP_PYTHON     interpreter for GeoCSP                   (WYFORMER_PYTHON: one environment)
#   GEOCSP_DIR        GeoCSP source (the DiffCSPNew repository) (/opt/geocsp)
#   GEOCSP_CKPT       GeoCSP weights                (/opt/okhotin/geocsp/geov2_alex_mp20_150e.pt)
#   WYFORMER_RUNS     the four WyFormer run directories         (/opt/okhotin/runs)
#   KEY_TABLE         alex-mp-20 train+val gene keys (/opt/okhotin/refs/alex_mp_20_labelled_train+val_keys.npz)
#   DEVICE            torch device for WyFormer and GeoCSP     (cuda)
#   CPUS              CPU workers for GeoCSP's PyXtal initialisation (all)
set -euo pipefail

LABEL="${1:?LABEL is CFG, CON or UC}"
OUT="${2:?OUT_DIR}"
N="${3:-10000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${WYFORMER_PYTHON:-/opt/wyformer/.venv/bin/python}"
GPY="${GEOCSP_PYTHON:-$PY}"
GEOCSP_DIR="${GEOCSP_DIR:-/opt/geocsp}"
GEOCSP_CKPT="${GEOCSP_CKPT:-/opt/okhotin/geocsp/geov2_alex_mp20_150e.pt}"
export WYFORMER_RUNS="${WYFORMER_RUNS:-/opt/okhotin/runs}"
KEY_TABLE="${KEY_TABLE:-/opt/okhotin/refs/alex_mp_20_labelled_train+val_keys.npz}"
DEVICE="${DEVICE:-cuda}"
CPUS="${CPUS:-$(nproc)}"
REGRESSOR="gene_min_ehull_adamw_wsd-20260929-154926"

# Per label: generator, sampling flags, and the pool that yields ~2 unique novel genes per
# kept gene -- the ~50% fire-control cut, at that generator's measured unique-novel rate
# (CFG 0.583, CON 0.667, UC 0.696 of formally valid genes).
case "$LABEL" in
    CFG) GEN_RUN=ehull_adamw_wsd_5x_cfg-20260929-150414
         GEN_ARGS=(--condition energy_above_hull=0.025 --guidance-scale 6); RATE=0.583 ;;
    CON) GEN_RUN=ehull_adamw_wsd_5x-20260929-143848
         GEN_ARGS=(--condition energy_above_hull=0.025); RATE=0.667 ;;
    UC)  GEN_RUN=uncond_adamw_wsd_5x-20260929-143845
         GEN_ARGS=(); RATE=0.696 ;;
    *)   echo "LABEL must be CFG, CON or UC, not $LABEL" >&2; exit 2 ;;
esac
# 4% headroom over N for GeoCSP failures and the start checks (0.1-0.2% observed).
BUDGET=$("$PY" -c "import math; print(math.ceil($N * 1.04))")
POOL_SIZE=$("$PY" -c "import math; print(math.ceil($BUDGET / 0.5 / $RATE))")

ARM="$OUT/fire-control"
D="$ARM/diffcsp"
mkdir -p "$OUT/pool" "$D"
log() { echo "[$(date '+%F %T')] [WyFormer-GeoCSP-$LABEL-v2.3] $*"; }

if [[ ! -f "$OUT/pool/wyckoff_genes.json.gz" ]]; then
    factor=1.3
    for attempt in 1 2 3; do
        n_draw=$("$PY" -c "import math; print(math.ceil($POOL_SIZE * $factor))")
        log "genes: drawing $n_draw for $POOL_SIZE formally valid genes"
        if "$PY" -m wyckoff_transformer.cli.generate "$OUT/pool/wyckoff_genes.json.gz" \
                --model-path "$WYFORMER_RUNS/$GEN_RUN" --initial-n-samples "$n_draw" \
                --firm-n-samples "$POOL_SIZE" --device "$DEVICE" "${GEN_ARGS[@]}" \
                > "$OUT/pool/generate.log" 2>&1; then
            break
        fi
        factor=$("$PY" -c "print($factor * 2)")
    done
    [[ -f "$OUT/pool/wyckoff_genes.json.gz" ]] || { log "generation failed: $OUT/pool/generate.log"; exit 1; }
    printf '{"label": "WyFormer-GeoCSP-%s-v2.3", "generator": "%s", "gen_args": "%s", "pool_size": %s, "raw_draws": %s}\n' \
        "$LABEL" "$GEN_RUN" "${GEN_ARGS[*]}" "$POOL_SIZE" "$n_draw" > "$OUT/pool/pool.json"
fi

if [[ ! -f "$ARM/engaged_genes.json.gz" ]]; then
    log "filtering: uniqueness and novelty against alex-mp-20, then the top $BUDGET by predicted e_hull"
    "$PY" -m wyckoff_transformer.roe.cli run fire-control \
        --genes "$OUT/pool/wyckoff_genes.json.gz" --output-dir "$ARM" --no-reconstruct \
        --n-genes "$POOL_SIZE" --energy-select top --energy-top "$BUDGET" \
        --regressor-path "$WYFORMER_RUNS/$REGRESSOR" --energy-hull direct \
        --reference-cache cache/alex_mp_20_labelled --reference-splits train,val \
        --screen-backend tensor --key-table "$KEY_TABLE" --device "$DEVICE" \
        > "$ARM/select.log" 2>&1
fi
[[ -f "$ARM/predicted_e_hull.csv" ]] || "$PY" "$HERE/predict_e_hull.py" \
    "$ARM/engaged_genes.json.gz" "$ARM/predicted_e_hull.csv" \
    --regressor-path "$WYFORMER_RUNS/$REGRESSOR" --device "$DEVICE" > "$ARM/predict.log" 2>&1
printf '{"label": "WyFormer-GeoCSP-%s-v2.3", "generator": "%s", "gen_args": "%s", "regressor": "%s", "budget": %s}\n' \
    "$LABEL" "$GEN_RUN" "${GEN_ARGS[*]}" "$REGRESSOR" "$BUDGET" > "$ARM/arm.json"

if [[ ! -f "$D/starts.csv" ]]; then
    log "GeoCSP: $BUDGET structures"
    "$PY" "$HERE/write_benchset.py" "$ARM/engaged_genes.json.gz" "$D/benchset.pkl"
    ( cd "$GEOCSP_DIR" \
      && LOKY_MAX_CPU_COUNT="$CPUS" "$GPY" bench/gen_init.py --benchset "$D/benchset.pkl" \
            --trials 1 --out "$D/inits.pkl" \
      && "$GPY" bench/run_diffusion.py --regime geov2 --ckpt "$GEOCSP_CKPT" \
            --inits "$D/inits.pkl" --batch_size 128 --seed 42 --out "$D/pred.pkl" ) \
        > "$ARM/geocsp.log" 2>&1
    "$GPY" "$HERE/export_predictions.py" "$D/benchset.pkl" "$D/pred.pkl" \
        "$D/starts.extxyz" "$D/starts.tmp.csv" >> "$ARM/geocsp.log" 2>&1
    mv "$D/starts.tmp.csv" "$D/starts.csv"
fi

log "submission: check, rattle and write the first $N"
"$PY" "$HERE/assemble_submission.py" "$ARM" "$OUT/submission" --n "$N"
log "done: $OUT/submission"
