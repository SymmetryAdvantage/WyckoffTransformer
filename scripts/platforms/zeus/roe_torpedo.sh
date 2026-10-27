#!/bin/bash
# ---------------------------------------------------------------------------
# A torpedo run on zeus: one pool, two paired arms. The zeus counterpart of
# scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh, for the models trained on
# lemat_bulk_fixed_*: the residuals, the screen, the PBE hull and the protocol's
# novelty reference are all LeMat-Bulk-fixed.
#
# Usage, from the root of a checkout:
#     nohup bash scripts/platforms/zeus/roe_torpedo.sh > log 2>&1 &
#     bash scripts/platforms/zeus/roe_torpedo.sh --pilot
#
# Steps, each skipped when its output already exists, so a rerun continues:
#   1. residuals      -- the regressor's honest residuals and the known genes' DFT energies
#   2. pool           -- N_TARGETS ternaries from the backbone's own system prior,
#                        each closed under its binaries, POOL genes in all
#   3. arm joint      -- ranked on the joint predicted hull, residual-corrected energies
#   4. arm reference  -- ranked on the DFT hull, raw energies
#   5. report each arm, and log everything to one W&B run unless --pilot
# See docs/rules_of_engagement.md#torpedo-run.
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

BACKBONE_RUN=chemsys_sg_uncond_adanmw_wsd_long-20261015-044354
REGRESSOR_RUN=min_energy_adamw_wsd_h7x_xl-20261015-185731
DATASET=lemat_bulk_fixed_fmax1_stress
REFERENCE_CACHE=cache/lemat_bulk_fixed_fmax1_stress
LEMAT_CIF_CSV=data/lemat_bulk_fixed_fmax1_stress
HULL_REFERENCE=data/lemat-bulk-fixed/lemat_pbe_ehull.csv.gz
N_TARGETS=50
POOL=20000
BUDGET=1000
SEED=0
DEVICES=${DEVICES:-0,1}
WORKERS_PER_DEVICE=${WORKERS_PER_DEVICE:-5}
PYXTAL_CORES=${PYXTAL_CORES:-20}
AUGMENTATION_SAMPLES=8
RESIDUAL_MAX_ROWS=""
ROOT=""
UPLOAD=1
SUFFIX=""

while [ $# -gt 0 ]; do
    case "$1" in
        --root)      ROOT=${2:?}; shift 2 ;;
        --pilot)     N_TARGETS=3; POOL=300; BUDGET=10; RESIDUAL_MAX_ROWS=50000
                     UPLOAD=0; SUFFIX="_pilot"; shift ;;
        --no-upload) UPLOAD=0; shift ;;
        -h|--help)   sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           die "unknown option $1" ;;
    esac
done

BACKBONE="$RUNS_DIR/$BACKBONE_RUN"
REGRESSOR="$RUNS_DIR/$REGRESSOR_RUN"
[ -f "$BACKBONE/system_prior.npz" ] || die "no system_prior.npz in $BACKBONE"
[ -f "$BACKBONE/best_model_params.pt" ] || die "no checkpoint in $BACKBONE"
[ -f "$REGRESSOR/best_model_params.pt" ] || die "no checkpoint in $REGRESSOR"
[ -n "$ROOT" ] || ROOT="$RUNS_DIR/roe/torpedo_${BACKBONE_RUN%%-*}${SUFFIX}"
NAME="roe_torpedo_${BACKBONE_RUN%%-*}-$(date +%Y%m%d)${SUFFIX}"
IFS=',' read -ra GPUS <<< "$DEVICES"
devices_list=""
for g in "${GPUS[@]}"; do devices_list+="cuda:$g,"; done
devices_list=${devices_list%,}
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)

COMMIT=$(git rev-parse HEAD)
DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false)
mkdir -p "$ROOT"
cp "${BASH_SOURCE[0]}" "$ROOT/driver.sh"

echo "=========================================================="
echo "torpedo run on zeus: $NAME"
echo "commit    : ${COMMIT:0:8} (dirty: $DIRTY)"
echo "date      : $(date -Is)"
echo "root      : $ROOT"
echo "backbone  : $BACKBONE"
echo "regressor : $REGRESSOR"
echo "reference : $REFERENCE_CACHE, $LEMAT_CIF_CSV, hull $HULL_REFERENCE"
echo "targets   : $N_TARGETS, pool $POOL, budget $BUDGET per arm"
echo "=========================================================="

if [ ! -f "$ROOT/residuals/residual_validation.json" ]; then
    echo "--- residuals ($(date -Is))"
    "$PY" -m wyckoff_transformer.gene_energy_residuals \
        --regressor-path "$REGRESSOR" --out-dir "$ROOT/residuals" --dataset "$DATASET" \
        --device "cuda:${GPUS[0]}" --augmentation-samples "$AUGMENTATION_SAMPLES" \
        ${RESIDUAL_MAX_ROWS:+--max-rows "$RESIDUAL_MAX_ROWS"}
fi

if [ ! -f "$ROOT/pool/wyckoff_genes.json.gz" ]; then
    echo "--- pool ($(date -Is))"
    "$PY" -m wyckoff_transformer.roe.cli draw \
        --model-path "$BACKBONE" --system-prior "$BACKBONE/system_prior.npz" \
        --closure --n-targets "$N_TARGETS" --target-arity 3 --closure-min-arity 2 \
        --sampler-seed "$SEED" --n-genes "$POOL" --allow-fewer \
        --output-dir "$ROOT/pool" --device "cuda:${GPUS[0]}"
fi

PROTOCOL_ARGS=(--devices "$devices_list" --workers-per-device "$WORKERS_PER_DEVICE"
               --pyxtal-cores "$PYXTAL_CORES"
               --reference-cache "$REFERENCE_CACHE" --lemat-cif-csv "$LEMAT_CIF_CSV")
for arm in joint reference; do
    if [ -f "$ROOT/$arm/protocol/funnel.json" ] && [ -f "$ROOT/$arm/engagement.json" ]; then
        echo "--- arm $arm already reconstructed"
        continue
    fi
    case "$arm" in
        joint)     hull=joint;     basis=corrected ;;
        reference) hull=reference; basis=raw ;;
    esac
    echo "--- arm $arm ($(date -Is))"
    OMP_NUM_THREADS=1 "$PY" -m wyckoff_transformer.roe.cli run torpedo-run \
        --genes "$ROOT/pool/wyckoff_genes.json.gz" \
        --system-plan "$ROOT/pool/system_plan.json" \
        --reference-cache "$REFERENCE_CACHE" --hull-reference "$HULL_REFERENCE" \
        --regressor-path "$REGRESSOR" --residuals "$ROOT/residuals" \
        --energy-select rank --energy-hull "$hull" --energy-basis "$basis" \
        --augmentation-samples "$AUGMENTATION_SAMPLES" \
        --n-genes "$POOL" --target-engaged "$BUDGET" --max-rounds 1 \
        --output-dir "$ROOT/$arm" --device "cuda:${GPUS[0]}" \
        -- "${PROTOCOL_ARGS[@]}"
    "$PY" -m wyckoff_transformer.roe.cli report "$ROOT/$arm" || true
done

if [ "$UPLOAD" -eq 1 ]; then
    echo "--- upload ($(date -Is))"
    WANDB_ENTITY=symmetry-advantage "$PY" -m wyckoff_transformer.roe.cli upload "$ROOT" \
        --name "$NAME" --arms joint reference \
        --config "commit=$COMMIT" --config "dirty=$DIRTY" --config "host=$(hostname)" \
        --config "backbone=$BACKBONE" --config "regressor=$REGRESSOR" \
        --config "dataset=$DATASET" --config "reference_cache=$REFERENCE_CACHE" \
        --config "lemat_cif_csv=$LEMAT_CIF_CSV" --config "hull_reference=$HULL_REFERENCE" \
        --config "n_targets=$N_TARGETS" --config "pool=$POOL" \
        --config "budget=$BUDGET" --config "seed=$SEED"
fi
echo "done ($(date -Is))"
