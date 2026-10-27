#!/bin/bash
# ---------------------------------------------------------------------------
# One fully relaxed gene pool on zeus, for replaying the rules of engagement
# with the fingerprint lookup and with the generator's own surprisal
# (docs/generative_novelty_screen.md). The zeus counterpart of
# scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh, for the models trained on
# lemat_bulk_fixed_*: everything is judged against the LeMat-Bulk-fixed
# reference (gene lookup, structure novelty, PBE hull of the predicted e_hull).
#
# Usage, from the root of a checkout:
#     nohup bash scripts/platforms/zeus/roe_surprisal.sh --backbone cfg_fixed > log 2>&1 &
#     bash scripts/platforms/zeus/roe_surprisal.sh --backbone uncond_fixed --pilot
#
# Steps, each skipped when its output already exists, so a rerun continues:
#   1. pool     -- POOL formally valid genes, in sampling order, duplicates kept
#   2. scores   -- predicted e_hull (wyformer-gene-screen) and the surprisal under
#                  each density the backbone names (wyformer-gene-novelty)
#   3. protocol -- the de novo ranking protocol over every unique gene of the pool
#   4. analysis -- scripts/analyse_roe_surprisal.py, uploaded to W&B unless --pilot
# ---------------------------------------------------------------------------
set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

REPO=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)
cd "$REPO"
PY="$REPO/.venv/bin/python"
RUNS_DIR=$("$PY" -c "from wyckoff_transformer.paths import runs_root; print(runs_root())")

BACKBONE=""
POOL=10000
OVERSAMPLE_OVERRIDE=""
ROOT=""
PILOT=0
DEVICES=${DEVICES:-0,1}
WORKERS_PER_DEVICE=${WORKERS_PER_DEVICE:-5}
PYXTAL_CORES=${PYXTAL_CORES:-20}
AUGMENTATION_SAMPLES=8
PERMUTATION_SAMPLES=64
GENERATION_BATCH=5000
REGRESSOR_RUN=min_energy_adamw_wsd_h7x_xl-20261015-185731
REFERENCE_CACHE=cache/lemat_bulk_fixed_fmax1_stress
LEMAT_CIF_CSV=data/lemat_bulk_fixed_fmax1_stress
HULL_REFERENCE=data/lemat-bulk-fixed/lemat_pbe_ehull.csv.gz

while [ $# -gt 0 ]; do
    case "$1" in
        --backbone)   BACKBONE=${2:?}; shift 2 ;;
        --pool)       POOL=${2:?}; shift 2 ;;
        --oversample) OVERSAMPLE_OVERRIDE=${2:?}; shift 2 ;;
        --root)       ROOT=${2:?}; shift 2 ;;
        --pilot)      PILOT=1; shift ;;
        -h|--help)    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            die "unknown option $1" ;;
    esac
done

# Each surprisal variant is NAME|FLAGS, separated by ';', as on aspire2a.
case "$BACKBONE" in
    cfg_fixed)
        MODEL_RUN=ehull_adamw_wsd_5x_cfg_long-20261015-021519
        GEN_ARGS="--condition energy_above_hull=0.05 --guidance-scale 5"
        # Formal validity at this cell was 61% in the model's protocol run.
        OVERSAMPLE_PERCENT=200
        SURPRISAL_VARIANTS="w5|--condition energy_above_hull=0.05 --guidance-scale 5;w1|--condition energy_above_hull=0.05"
        ;;
    uncond_fixed)
        MODEL_RUN=unconditional_5x_ehull01_h3x_xl_lr7p5e-4-20261018-202012
        GEN_ARGS=""
        OVERSAMPLE_PERCENT=120
        SURPRISAL_VARIANTS="plain|"
        ;;
    *) die "--backbone is cfg_fixed or uncond_fixed" ;;
esac

SUFFIX=""
ANALYSIS_ARGS="--wandb-name roe_surprisal_${BACKBONE}-$(date +%Y%m%d)"
if [ "$PILOT" -eq 1 ]; then
    POOL=200
    SUFFIX="_pilot"
    ANALYSIS_ARGS="--budgets 25,50 --main-budget 50"
fi
[ -n "$OVERSAMPLE_OVERRIDE" ] && OVERSAMPLE_PERCENT=$OVERSAMPLE_OVERRIDE
INITIAL=$((POOL * OVERSAMPLE_PERCENT / 100))
MODEL="$RUNS_DIR/$MODEL_RUN"
REGRESSOR="$RUNS_DIR/$REGRESSOR_RUN"
[ -f "$MODEL/best_model_params.pt" ] || die "no checkpoint in $MODEL"
[ -f "$REGRESSOR/best_model_params.pt" ] || die "no checkpoint in $REGRESSOR"
[ -n "$ROOT" ] || ROOT="$RUNS_DIR/roe_surprisal/${BACKBONE}${SUFFIX}"
IFS=',' read -ra GPUS <<< "$DEVICES"
TORCH_DEVICES=$(printf 'cuda:%s,' "${GPUS[@]}")
TORCH_DEVICES=${TORCH_DEVICES%,}
# Every GPU stays visible; each step names its own device.
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)

COMMIT=$(git rev-parse HEAD)
DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false)
mkdir -p "$ROOT/pool" "$ROOT/scores"
cp "${BASH_SOURCE[0]}" "$ROOT/driver.sh"

echo "=========================================================="
echo "roe surprisal pool on zeus: ${BACKBONE}${SUFFIX}"
echo "commit    : ${COMMIT:0:8} (dirty: $DIRTY)"
echo "date      : $(date -Is)"
echo "root      : $ROOT"
echo "backbone  : $MODEL $GEN_ARGS"
echo "regressor : $REGRESSOR"
echo "reference : $REFERENCE_CACHE, $LEMAT_CIF_CSV, hull $HULL_REFERENCE"
echo "pool      : $POOL genes from $INITIAL draws"
echo "devices   : $TORCH_DEVICES x $WORKERS_PER_DEVICE workers"
echo "=========================================================="

GENES="$ROOT/pool/wyckoff_genes.json.gz"
if [ ! -s "$GENES" ]; then
    echo "--- pool ($(date -Is))"
    started=$SECONDS
    # shellcheck disable=SC2086 # GEN_ARGS is a list of flags
    "$PY" -m wyckoff_transformer.cli.generate "$ROOT/pool/draw_partial.json.gz" \
        --model-path "$MODEL" --initial-n-samples "$INITIAL" --firm-n-samples "$POOL" \
        --device "cuda:${GPUS[0]}" --generation-batch "$GENERATION_BATCH" $GEN_ARGS
    mv "$ROOT/pool/draw_partial.json.gz" "$GENES"
    cat > "$ROOT/pool/pool_manifest.json" <<EOF
{"model": "$MODEL", "generation_args": "$GEN_ARGS", "initial_n_samples": $INITIAL,
 "firm_n_samples": $POOL, "seconds": $((SECONDS - started)), "commit": "$COMMIT",
 "dirty": $DIRTY, "host": "$(hostname)", "regressor": "$REGRESSOR",
 "reference_cache": "$REFERENCE_CACHE", "lemat_cif_csv": "$LEMAT_CIF_CSV",
 "hull_reference": "$HULL_REFERENCE", "date": "$(date -Is)"}
EOF
fi

echo "--- scores ($(date -Is))"
pids=()
slot=0
if [ ! -s "$ROOT/scores/gene_screen.csv" ]; then
    ( "$PY" -m wyckoff_transformer.cli.gene_screen "$GENES" \
          --regressor-path "$REGRESSOR" --reference "$HULL_REFERENCE" \
          --augmentation-samples "$AUGMENTATION_SAMPLES" \
          --device "cuda:${GPUS[0]}" --out "$ROOT/scores/gene_screen.tmp.csv" \
      && mv "$ROOT/scores/gene_screen.tmp.csv" "$ROOT/scores/gene_screen.csv" ) &
    pids+=($!)
fi
IFS=';' read -ra variants <<< "$SURPRISAL_VARIANTS"
for variant in "${variants[@]}"; do
    name=${variant%%|*}
    args=${variant#*|}
    out="$ROOT/scores/gene_novelty_${name}.csv"
    [ -s "$out" ] && continue
    slot=$((slot + 1))
    gpu=${GPUS[$((slot % ${#GPUS[@]}))]}
    # shellcheck disable=SC2086 # args is a list of flags
    ( "$PY" -m wyckoff_transformer.cli.gene_novelty "$GENES" \
          --model-path "$MODEL" $args --permutation-samples "$PERMUTATION_SAMPLES" \
          --batch-size 2500 --seed 0 --device "cuda:$gpu" --out "$out.tmp.csv" \
      && mv "$out.tmp.csv" "$out" ) &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[ "$failed" -eq 0 ] || die "a gene-scoring step failed (see above)"

PROTOCOL="$ROOT/protocol"
devices_list=""
for g in "${GPUS[@]}"; do devices_list+="cuda:$g,"; done
PROTOCOL_ARGS=(--devices "${devices_list%,}" --workers-per-device "$WORKERS_PER_DEVICE"
               --pyxtal-cores "$PYXTAL_CORES"
               --reference-cache "$REFERENCE_CACHE" --lemat-cif-csv "$LEMAT_CIF_CSV")
for stage in screen generate relax score; do
    case "$stage" in
        screen)   [ -f "$PROTOCOL/screen.json" ] && continue ;;
        generate) [ -f "$PROTOCOL/pyxtal.csv" ] && continue ;;
        score)    [ -f "$PROTOCOL/funnel.json" ] && continue ;;
    esac
    echo "--- protocol $stage ($(date -Is))"
    OMP_NUM_THREADS=1 "$PY" -m wyckoff_transformer.cli.protocol "$GENES" \
        --output-dir "$PROTOCOL" --stage "$stage" "${PROTOCOL_ARGS[@]}"
done

if [ ! -f "$ROOT/analysis/report.json" ]; then
    echo "--- analysis ($(date -Is))"
    # shellcheck disable=SC2086 # ANALYSIS_ARGS is a list of flags
    WANDB_ENTITY=symmetry-advantage "$PY" scripts/analyse_roe_surprisal.py "$ROOT" \
        --workers 20 $ANALYSIS_ARGS
fi
echo "done ($(date -Is))"
