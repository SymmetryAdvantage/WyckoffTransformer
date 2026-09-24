#!/bin/bash
# ---------------------------------------------------------------------------
# A torpedo run on a full 4-GPU ASPIRE 2A node: one pool, two paired arms.
#
# Usage:
#     bash scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh [options]
#     bash scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh --pilot
#     bash scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh --dry-run
#
# What the job does, each step skipped when its output already exists, so a
# resubmission after the 24 h ceiling continues rather than restarts:
#   1. residuals  -- the gene-energy regressor's honest residuals and the known
#                    genes' DFT energies (wyckoff_transformer.gene_energy_residuals)
#   2. pool       -- N_TARGETS ternaries drawn from the backbone's own system prior,
#                    each closed under its binaries, POOL genes in all
#   3. arm joint  -- torpedo-run ranking on the joint predicted hull, residual-
#                    corrected energies
#   4. arm reference -- torpedo-run ranking on the DFT hull, raw energies: the
#                    selection the earlier modes used, the ablation
#   5. report each arm, and log everything to one W&B run
# Each arm reconstructs BUDGET genes with wyformer-protocol's own defaults.
# See docs/rules_of_engagement.md#torpedo-run.
#
# Resources follow protocol_wandb_in_pbs.sh: 4x A100-40GB, 64 cores, 440 GB;
# 4 relaxation workers per GPU is the measured saturation knee.
# ---------------------------------------------------------------------------
#PBS -N wyf_torpedo
#PBS -q ai
#PBS -P 11001786
#PBS -l select=1:ngpus=4:ncpus=64:mem=440gb
#PBS -l walltime=23:59:59
#PBS -j oe
#PBS -o /scratch/users/nus/kna/WyFormer/logs/

set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

# ===========================================================================
# Inside the PBS job
# ===========================================================================
run_job_payload() {
    [ -f "${JOB_SPEC:?no JOB_SPEC provided to PBS job}" ] || die "spec file not found: $JOB_SPEC"
    # shellcheck source=/dev/null
    . "$JOB_SPEC"
    cd "$REPO"

    echo "=========================================================="
    echo "torpedo run on a full 4-GPU node"
    echo "pbs job   : ${PBS_JOBID:-<interactive>}   node: $(hostname)"
    echo "commit    : ${COMMIT:0:8} (${BRANCH})"
    echo "date      : $(date -Is)"
    echo "root      : $ROOT"
    echo "backbone  : $BACKBONE"
    echo "regressor : $REGRESSOR"
    echo "targets   : $N_TARGETS, pool $POOL, budget $BUDGET per arm"
    echo "=========================================================="
    nvidia-smi || true

    if ! command -v singularity >/dev/null 2>&1; then
        type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
        module load singularity
    fi
    local RUN=(bash "$REPO/scripts/platforms/aspire2a/run_in_singularity.sh")
    "${RUN[@]}" python -c "import orb_models" \
        || die "orb_models is not importable in the venv (docs/platforms/aspire2a/environment.md)"
    mkdir -p "$ROOT"

    if [ ! -f "$ROOT/residuals/residual_validation.json" ]; then
        echo "--- residuals ($(date -Is))"
        "${RUN[@]}" python -m wyckoff_transformer.gene_energy_residuals \
            --regressor-path "$REGRESSOR" --out-dir "$ROOT/residuals" --device cuda:0 \
            --augmentation-samples "$AUGMENTATION_SAMPLES" \
            ${RESIDUAL_MAX_ROWS:+--max-rows "$RESIDUAL_MAX_ROWS"}
    fi

    if [ ! -f "$ROOT/pool/wyckoff_genes.json.gz" ]; then
        echo "--- pool ($(date -Is))"
        "${RUN[@]}" python -m wyckoff_transformer.roe.cli draw \
            --model-path "$BACKBONE" --system-prior "$BACKBONE/system_prior.npz" \
            --closure --n-targets "$N_TARGETS" --target-arity 3 --closure-min-arity 2 \
            --sampler-seed "$SEED" --n-genes "$POOL" --allow-fewer \
            --output-dir "$ROOT/pool" --device cuda:0
    fi

    local PROTOCOL_ARGS=(--devices cuda:0,cuda:1,cuda:2,cuda:3
                         --workers-per-device "$WORKERS_PER_DEVICE"
                         --pyxtal-cores "$PYXTAL_CORES")
    local arm hull basis
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
        "${RUN[@]}" python -m wyckoff_transformer.roe.cli run torpedo-run \
            --genes "$ROOT/pool/wyckoff_genes.json.gz" \
            --system-plan "$ROOT/pool/system_plan.json" \
            --regressor-path "$REGRESSOR" --residuals "$ROOT/residuals" \
            --energy-select rank --energy-hull "$hull" --energy-basis "$basis" \
            --augmentation-samples "$AUGMENTATION_SAMPLES" \
            --n-genes "$POOL" --target-engaged "$BUDGET" --max-rounds 1 \
            --output-dir "$ROOT/$arm" --device cuda:0 \
            -- "${PROTOCOL_ARGS[@]}"
        "${RUN[@]}" python -m wyckoff_transformer.roe.cli report "$ROOT/$arm" || true
    done

    if [ "$UPLOAD" -eq 1 ]; then
        echo "--- upload ($(date -Is))"
        "${RUN[@]}" python -m wyckoff_transformer.roe.cli upload "$ROOT" \
            --name "$WANDB_NAME" --arms joint reference \
            --config "commit=$COMMIT" --config "branch=$BRANCH" \
            --config "backbone=$BACKBONE" --config "regressor=$REGRESSOR" \
            --config "n_targets=$N_TARGETS" --config "pool=$POOL" \
            --config "budget=$BUDGET" --config "seed=$SEED" \
            --config "pbs_job=${PBS_JOBID:-}"
    fi
    echo "done ($(date -Is))"
}

if [ -n "${PBS_JOBID:-}" ] && [ -n "${JOB_SPEC:-}" ]; then
    run_job_payload
    exit 0
fi

# ===========================================================================
# Submission
# ===========================================================================
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
REPO_DIR=$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)
# shellcheck source=scripts/wyformer_paths.sh
. "$REPO_DIR/scripts/wyformer_paths.sh"
LOGS_DIR=${WYFORMER_LOGS:-/scratch/users/nus/kna/WyFormer/logs}
RUNS_DIR=$(wyformer_path WYFORMER_RUNS "$REPO_DIR/runs") || exit 1
QSUB=/opt/pbs/bin/qsub
[ -x "$QSUB" ] || QSUB=$(command -v qsub || echo "qsub")

BACKBONE_RUN=chemsys_sg_uncond_adanmw_wsd-20260921-220745
REGRESSOR_RUN=min_energy_adamw_wsd-20260924-102431
N_TARGETS=50
POOL=20000
BUDGET=1000
SEED=0
WORKERS_PER_DEVICE=4
PYXTAL_CORES=60
# Each prediction averages this many equivalent Wyckoff descriptions: one draws a
# random one, and the two arms would then score the same gene differently.
AUGMENTATION_SAMPLES=8
WALLTIME="23:59:59"
NAME=""
ROOT=""
RESIDUAL_MAX_ROWS=""
UPLOAD=1
PILOT=0
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
    sed -n '2,25p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
    --backbone RUN        chemical-system-conditioned generator (default: $BACKBONE_RUN)
    --regressor RUN       gene-energy regressor (default: $REGRESSOR_RUN)
    --n-targets N         ternary targets (default: $N_TARGETS)
    --pool N              genes drawn, over all targets (default: $POOL)
    --budget N            reconstructions per arm (default: $BUDGET)
    --seed N              sampler seed (default: $SEED)
    --root DIR            output root (default: \$WYFORMER_RUNS/roe/torpedo_<backbone>[_pilot])
    --name NAME           W&B run name (default: roe_torpedo_<date>[_pilot])
    --pilot               3 targets, 300 genes, 10 per arm, 01:59:00 walltime (aidev)
    --no-upload           skip the W&B run
    --allow-dirty         submit with uncommitted changes
    --dry-run, -n         print, do not submit
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --backbone)    BACKBONE_RUN=${2:?}; shift 2 ;;
        --regressor)   REGRESSOR_RUN=${2:?}; shift 2 ;;
        --n-targets)   N_TARGETS=${2:?}; shift 2 ;;
        --pool)        POOL=${2:?}; shift 2 ;;
        --budget)      BUDGET=${2:?}; shift 2 ;;
        --seed)        SEED=${2:?}; shift 2 ;;
        --root)        ROOT=${2:?}; shift 2 ;;
        --name)        NAME=${2:?}; shift 2 ;;
        --pilot)       PILOT=1; shift ;;
        --no-upload)   UPLOAD=0; shift ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --dry-run|-n)  DRY_RUN=1; shift ;;
        -h|--help)     usage ;;
        *)             die "unknown option $1 (see --help)" ;;
    esac
done

SUFFIX=""
if [ "$PILOT" -eq 1 ]; then
    WALLTIME="01:59:00"
    N_TARGETS=3; POOL=300; BUDGET=10; RESIDUAL_MAX_ROWS=50000; SUFFIX="_pilot"
fi
BACKBONE="$RUNS_DIR/$BACKBONE_RUN"
REGRESSOR="$RUNS_DIR/$REGRESSOR_RUN"
[ -f "$BACKBONE/system_prior.npz" ] || die "no system_prior.npz in $BACKBONE"
[ -f "$BACKBONE/best_model_params.pt" ] || die "no checkpoint in $BACKBONE"
[ -f "$REGRESSOR/best_model_params.pt" ] || die "no checkpoint in $REGRESSOR"
[ -n "$ROOT" ] || ROOT="$RUNS_DIR/roe/torpedo_${BACKBONE_RUN%%-*}${SUFFIX}"
[ -n "$NAME" ] || NAME="roe_torpedo_${BACKBONE_RUN%%-*}-$(date +%Y%m%d)${SUFFIX}"

case "$REPO_DIR" in
    */.claude/worktrees/*) die "$REPO_DIR is a Claude Code worktree; use create_worktree.sh" ;;
esac
BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$ALLOW_DIRTY" -eq 0 ] && [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
    die "Uncommitted changes in $REPO_DIR. Commit first or pass --allow-dirty."
fi

mkdir -p "$LOGS_DIR" "$RUNS_DIR/.jobspec"
SPEC="$RUNS_DIR/.jobspec/torpedo-$(date +%Y%m%d-%H%M%S)${SUFFIX}.sh"
{
    echo "# Generated by scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh on $(date -Is)"
    for var in REPO_DIR BRANCH COMMIT ROOT BACKBONE REGRESSOR N_TARGETS POOL BUDGET SEED \
               WORKERS_PER_DEVICE PYXTAL_CORES AUGMENTATION_SAMPLES RESIDUAL_MAX_ROWS UPLOAD; do
        printf '%s=%q\n' "$var" "${!var}"
    done
    printf 'REPO=%q\n' "$REPO_DIR"
    printf 'WANDB_NAME=%q\n' "$NAME"
} > "$SPEC"

QSUB_ARGS=(-N "wyf_torpedo${SUFFIX}" -q ai -P 11001786
           -l "select=1:ngpus=4:ncpus=64:mem=440gb" -l "walltime=$WALLTIME"
           -j oe -o "$LOGS_DIR/" -v "JOB_SPEC=$SPEC")

echo "=========================================================="
echo "torpedo run      : $NAME"
echo "branch           : $BRANCH @ ${COMMIT:0:8}"
echo "root             : $ROOT"
echo "backbone         : $BACKBONE"
echo "regressor        : $REGRESSOR"
echo "targets/pool/arm : $N_TARGETS / $POOL / $BUDGET"
echo "walltime         : $WALLTIME"
echo "spec             : $SPEC"
echo "=========================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "$QSUB ${QSUB_ARGS[*]} $SCRIPT_PATH"
    cat "$SPEC"
    rm -f "$SPEC"
    exit 0
fi
"$QSUB" "${QSUB_ARGS[@]}" "$SCRIPT_PATH"
