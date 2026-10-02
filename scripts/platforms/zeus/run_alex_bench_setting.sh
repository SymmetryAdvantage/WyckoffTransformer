#!/usr/bin/env bash
# One generator setting of the alex-mp-20 benchmark study, on zeus.
#
# Draws one gene pool, selects each requested rule-of-engagement arm from it, and
# evaluates every arm the same way the submission would be judged:
#   genes -> DiffCSP++ GeoV2 (alex-mp-20) -> rattle -> one unconstrained ORB relaxation
#   -> ORB hull (LeMat-Bulk-MLIP-Hull) and StructureMatcher novelty vs alex-mp-20.
# ORB is evaluation only: nothing it computes feeds a selection.
#
# Environment:
#   SETTING      name of the setting, e.g. cfg_e0p05_w4 (required)
#   GEN_RUN      generator run id in the runs store (required)
#   GEN_ARGS     sampling flags, e.g. "--condition energy_above_hull=0.05 --guidance-scale 4"
#   ARMS         subset of "broadside fire-discipline fire-control" (default: all three)
#   BUDGET       genes handed to DiffCSP++ per arm (default 1000)
#   POOL_SIZE    formally valid genes in the pool (default 2.5 x BUDGET)
#   OVERSAMPLE   raw draws per pool gene for the first attempt (default 1.3)
#   REGRESSOR    gene e_hull predictor run id (fire-control only)
#   GPU          card index for every GPU step (default 1)
#   WORKERS      relaxation workers on that card (default 8)
#   CPUS         CPU cores for PyXtal initialisation and DiffCSP++ data loading (default 20)
#
# Re-running skips every arm whose summary.json exists and every step whose output does.
set -euo pipefail

: "${SETTING:?}" "${GEN_RUN:?}"
GEN_ARGS="${GEN_ARGS:-}"
ARMS="${ARMS:-broadside fire-discipline fire-control}"
BUDGET="${BUDGET:-1000}"
POOL_SIZE="${POOL_SIZE:-$(( BUDGET * 5 / 2 ))}"
OVERSAMPLE="${OVERSAMPLE:-1.3}"
REGRESSOR="${REGRESSOR:-gene_min_ehull_adamw_wsd-20260929-154926}"
GPU="${GPU:-1}"
WORKERS="${WORKERS:-8}"
CPUS="${CPUS:-20}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="$REPO/.venv/bin/python"
DIFFCSP="/home/kna/DiffCSPNew"
DPY="$DIFFCSP/.venv/bin/python"
DIFFCSP_CKPT="$DIFFCSP/runs/alex_mp20_geov2/geov2_alex_mp20_150e.pt"
STORE="$(dirname "$("$PY" -c 'from wyckoff_transformer.paths import runs_root; print(runs_root())')")"
RUNS="$STORE/runs"
BENCH="$STORE/alex_bench"
REFS="$BENCH/refs"
ROOT="$BENCH/$SETTING"
POOL="$ROOT/pool/wyckoff_genes.json.gz"
SUMMARY="$BENCH/summary.csv"

# The alex-mp-20 novelty reference: the cache the generators were trained on, its CIFs,
# and fingerprint / key caches outside the (read-only) cache directory.
REF_ARGS=(--reference-cache cache/alex_mp_20_labelled --reference-splits train,val)
PROTOCOL_REF_ARGS=("${REF_ARGS[@]}" --reference-id-column material_id
                   --lemat-cif-csv data/alex_mp_20_labelled
                   --reference-fingerprint-cache "$REFS/alex_mp_20_labelled_train+val_fps.pkl.gz")
ROE_REF_ARGS=("${REF_ARGS[@]}" --screen-backend tensor
              --key-table "$REFS/alex_mp_20_labelled_train+val_keys.npz")

cd "$REPO"
mkdir -p "$ROOT/pool" "$REFS"
log() { echo "[$(date '+%F %T')] [$SETTING] $*"; }
read -r -a GEN_ARGV <<< "$GEN_ARGS"

if [[ ! -f "$POOL" ]]; then
    # Formal validity falls steeply with guidance at low targets, so a short draw is
    # retried with more oversampling rather than failing the setting.
    factor="$OVERSAMPLE"
    for attempt in 1 2 3 4; do
        n_draw=$("$PY" -c "import math; print(math.ceil($POOL_SIZE * $factor))")
        log "pool: drawing $n_draw for $POOL_SIZE valid genes (attempt $attempt)"
        if CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m wyckoff_transformer.cli.generate "$POOL" \
                --model-path "$RUNS/$GEN_RUN" --initial-n-samples "$n_draw" \
                --firm-n-samples "$POOL_SIZE" --device cuda "${GEN_ARGV[@]}" \
                > "$ROOT/pool/generate.log" 2>&1; then
            break
        fi
        factor=$("$PY" -c "print($factor * 2.5)")
        rm -f "$POOL"
    done
    [[ -f "$POOL" ]] || { log "pool: generation failed, see $ROOT/pool/generate.log"; exit 1; }
    cat > "$ROOT/pool/pool.json" <<EOF
{"generator": "$GEN_RUN", "gen_args": "$GEN_ARGS", "pool_size": $POOL_SIZE, "raw_draws": $n_draw}
EOF
fi

select_arm() {  # $1 = mode, $2 = arm dir
    local mode="$1" dir="$2"
    local extra=()
    case "$mode" in
        broadside)       extra=(--n-genes $(( BUDGET + BUDGET / 10 )) --target-engaged "$BUDGET") ;;
        fire-discipline) extra=(--n-genes $(( BUDGET * 2 )) --target-engaged "$BUDGET") ;;
        fire-control)    extra=(--n-genes "$POOL_SIZE" --energy-select top --energy-top "$BUDGET"
                                --regressor-path "$RUNS/$REGRESSOR" --energy-hull direct
                                --device cuda) ;;
    esac
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m wyckoff_transformer.roe.cli run "$mode" \
        --genes "$POOL" --output-dir "$dir" --no-reconstruct \
        "${ROE_REF_ARGS[@]}" "${extra[@]}" > "$dir/select.log" 2>&1
}

evaluate_arm() {  # $1 = arm dir
    local dir="$1" genes="$1/engaged_genes.json.gz" P="$1/protocol" D="$1/diffcsp"
    local protocol=("$PY" -m wyckoff_transformer.cli.protocol "$genes" --output-dir "$P")
    mkdir -p "$P" "$D"
    [[ -f "$P/screen.json" ]] || "${protocol[@]}" --stage screen "${PROTOCOL_REF_ARGS[@]}" \
        > "$dir/screen.log" 2>&1
    if [[ ! -f "$D/starts.csv" ]]; then
        log "$(basename "$dir"): DiffCSP++"
        "$PY" scripts/alex_bench/write_benchset.py "$genes" "$P" "$D/benchset.pkl"
        ( cd "$DIFFCSP" \
          && LOKY_MAX_CPU_COUNT="$CPUS" "$DPY" bench/gen_init.py --benchset "$D/benchset.pkl" \
                --trials 1 --out "$D/inits.pkl" \
          && CUDA_VISIBLE_DEVICES="$GPU" "$DPY" bench/run_diffusion.py --regime geov2 \
                --ckpt "$DIFFCSP_CKPT" --inits "$D/inits.pkl" --batch_size 128 --seed 42 \
                --out "$D/pred.pkl" ) > "$dir/diffcsp.log" 2>&1
        "$DPY" scripts/alex_bench/export_predictions.py "$D/benchset.pkl" "$D/pred.pkl" \
            "$D/starts.extxyz" "$D/starts.tmp.csv" >> "$dir/diffcsp.log" 2>&1
        mv "$D/starts.tmp.csv" "$D/starts.csv"
    fi
    [[ -f "$P/starts.csv" ]] || "${protocol[@]}" --stage starts --starts "$D/starts.extxyz" \
        --starts-log "$D/starts.csv" > "$dir/starts.log" 2>&1
    if [[ ! -f "$P/structures.csv" ]] || ! grep -q relax_schedule "$P/manifest.json"; then
        log "$(basename "$dir"): relax"
        # --devices names the card; CUDA_VISIBLE_DEVICES must stay unset here, since
        # each worker pins itself from its own device string.
        "${protocol[@]}" --stage relax --relax-schedule single --no-rattle --fmax 0.05 \
            --relax-steps 1000 --devices "cuda:$GPU" --workers-per-device "$WORKERS" \
            --relax-timeout 600 --resume > "$dir/relax.log" 2>&1
    fi
    if [[ ! -f "$P/funnel.json" ]]; then
        log "$(basename "$dir"): score"
        "${protocol[@]}" --stage score "${PROTOCOL_REF_ARGS[@]}" > "$dir/score.log" 2>&1
    fi
}

for mode in $ARMS; do
    dir="$ROOT/$mode"
    if [[ -f "$dir/summary.json" ]]; then
        log "$mode: done already"
        continue
    fi
    mkdir -p "$dir"
    [[ -f "$dir/engaged_genes.json.gz" ]] || { log "$mode: select"; select_arm "$mode" "$dir"; }
    cat > "$dir/arm.json" <<EOF
{"generator": "$GEN_RUN", "gen_args": "$GEN_ARGS", "regressor": "$([[ $mode == fire-control ]] && echo "$REGRESSOR")", "budget": $BUDGET}
EOF
    [[ -f "$dir/predicted_e_hull.csv" ]] || CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
        scripts/alex_bench/predict_e_hull.py "$dir/engaged_genes.json.gz" \
        "$dir/predicted_e_hull.csv" --regressor-path "$RUNS/$REGRESSOR" --device cuda \
        > "$dir/predict.log" 2>&1
    evaluate_arm "$dir"
    # Locked: several settings may run side by side and all append to one summary.
    flock "$SUMMARY.lock" "$PY" scripts/alex_bench/summarise_arm.py "$dir" \
        --setting "$SETTING" --arm "$mode" --summary "$SUMMARY"
done
log "done"
