#!/bin/bash
# ---------------------------------------------------------------------------
# One fully relaxed gene pool on a 4-GPU ASPIRE 2A node, for replaying the
# rules of engagement with the fingerprint lookup and with the generator's own
# surprisal (docs/generative_novelty_screen.md).
#
# Usage:
#     bash scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh --backbone cfg
#     bash scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh --backbone uncond
#     bash scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh --backbone cfg --pilot
#     bash scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh --backbone cfg --dry-run
#
# What the job does, each step skipped when its output already exists, so a
# resubmission after the 24 h ceiling continues rather than restarts:
#   1. pool     -- POOL formally valid genes from the backbone, in sampling order,
#                  duplicates kept (wyformer-generate)
#   2. scores   -- per gene, in parallel on three GPUs: the predicted e_hull
#                  (wyformer-gene-screen) and the surprisal under each density
#                  the backbone names (wyformer-gene-novelty)
#   3. protocol -- the de novo ranking protocol over EVERY unique gene of the
#                  pool: screen, generate, relax, score
#   4. analysis -- scripts/analyse_roe_surprisal.py, which uploads the pool, the
#                  scores, the protocol outputs and the report to one W&B run
# Every arm -- broadside, fire-discipline and fire-control, with the lookup or
# with the surprisal band -- is a selection from one relaxed pool, scored
# against the same relaxation outcomes.
#
# Resources follow protocol_wandb_in_pbs.sh: 4x A100-40GB, 64 cores, 440 GB;
# 4 relaxation workers per GPU is the measured saturation knee.
# ---------------------------------------------------------------------------
#PBS -N wyf_roe_surprisal
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
    echo "roe surprisal pool on a full 4-GPU node"
    echo "pbs job   : ${PBS_JOBID:-<interactive>}   node: $(hostname)"
    echo "commit    : ${COMMIT:0:8} (${BRANCH})"
    echo "date      : $(date -Is)"
    echo "root      : $ROOT"
    echo "backbone  : $MODEL"
    echo "sampling  : ${GEN_ARGS:-<none>}"
    echo "regressor : $REGRESSOR"
    echo "pool      : $POOL genes from $INITIAL draws"
    echo "surprisal : $SURPRISAL_VARIANTS"
    echo "=========================================================="
    nvidia-smi || true

    if ! command -v singularity >/dev/null 2>&1; then
        type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
        module load singularity
    fi
    local RUN=(bash "$REPO/scripts/platforms/aspire2a/run_in_singularity.sh")
    "${RUN[@]}" python -c "import orb_models" \
        || die "orb_models is not importable in the venv (docs/platforms/aspire2a/environment.md)"
    mkdir -p "$ROOT/pool" "$ROOT/scores"

    local GENES="$ROOT/pool/wyckoff_genes.json.gz"
    if [ ! -s "$GENES" ]; then
        echo "--- pool ($(date -Is))"
        local started=$SECONDS
        # wyformer-generate insists on exactly the suffixes .json.gz, so the
        # temporary name may not carry a .tmp of its own.
        # shellcheck disable=SC2086 # GEN_ARGS is a list of flags
        "${RUN[@]}" python -m wyckoff_transformer.cli.generate "$ROOT/pool/draw_partial.json.gz" \
            --model-path "$MODEL" --initial-n-samples "$INITIAL" --firm-n-samples "$POOL" \
            --device cuda:0 $GEN_ARGS
        mv "$ROOT/pool/draw_partial.json.gz" "$GENES"
        cat > "$ROOT/pool/pool_manifest.json" <<EOF
{"model": "$MODEL", "generation_args": "$GEN_ARGS", "initial_n_samples": $INITIAL,
 "firm_n_samples": $POOL, "seconds": $((SECONDS - started)), "commit": "$COMMIT",
 "branch": "$BRANCH", "pbs_job": "${PBS_JOBID:-}", "date": "$(date -Is)"}
EOF
    fi

    echo "--- scores ($(date -Is))"
    local pids=() gpu=1 variant name args out
    if [ ! -s "$ROOT/scores/gene_screen.csv" ]; then
        ( "${RUN[@]}" python -m wyckoff_transformer.cli.gene_screen "$GENES" \
              --regressor-path "$REGRESSOR" --augmentation-samples "$AUGMENTATION_SAMPLES" \
              --device cuda:0 --out "$ROOT/scores/gene_screen.tmp.csv" \
          && mv "$ROOT/scores/gene_screen.tmp.csv" "$ROOT/scores/gene_screen.csv" ) &
        pids+=($!)
    fi
    IFS=';' read -ra variants <<< "$SURPRISAL_VARIANTS"
    for variant in "${variants[@]}"; do
        name=${variant%%|*}
        args=${variant#*|}
        out="$ROOT/scores/gene_novelty_${name}.csv"
        [ -s "$out" ] && continue
        # shellcheck disable=SC2086 # args is a list of flags
        ( "${RUN[@]}" python -m wyckoff_transformer.cli.gene_novelty "$GENES" \
              --model-path "$MODEL" $args --permutation-samples "$PERMUTATION_SAMPLES" \
              --batch-size 2500 --seed 0 --device "cuda:$gpu" --out "$out.tmp.csv" \
          && mv "$out.tmp.csv" "$out" ) &
        pids+=($!)
        gpu=$((gpu + 1))
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [ "$failed" -eq 0 ] || die "a gene-scoring step failed (see above)"

    local PROTOCOL="$ROOT/protocol"
    local PROTOCOL_ARGS=(--devices cuda:0,cuda:1,cuda:2,cuda:3
                         --workers-per-device "$WORKERS_PER_DEVICE"
                         --pyxtal-cores "$PYXTAL_CORES")
    local stage
    for stage in screen generate relax score; do
        case "$stage" in
            screen)   [ -f "$PROTOCOL/screen.json" ] && continue ;;
            generate) [ -f "$PROTOCOL/pyxtal.csv" ] && continue ;;
            score)    [ -f "$PROTOCOL/funnel.json" ] && continue ;;
        esac
        echo "--- protocol $stage ($(date -Is))"
        "${RUN[@]}" python -m wyckoff_transformer.cli.protocol "$GENES" \
            --output-dir "$PROTOCOL" --stage "$stage" "${PROTOCOL_ARGS[@]}"
    done

    # The replay is minutes of CPU and the relaxation is not, so a failure here
    # does not fail the job: rerun the script by hand on the finished pool.
    if [ ! -f "$ROOT/analysis/report.json" ]; then
        echo "--- analysis ($(date -Is))"
        # shellcheck disable=SC2086 # ANALYSIS_ARGS is a list of flags
        "${RUN[@]}" python scripts/analyse_roe_surprisal.py "$ROOT" --workers 60 \
            $ANALYSIS_ARGS || echo "warning: the analysis failed; the pool is complete" >&2
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

BACKBONE=""
# The xl energy predictor, frozen on 2026-10-08 while its chain was still in its
# last link (val MAE 0.0253): the chain overwrites best_model_params.pt in place.
REGRESSOR_DIR=roe_surprisal/models/min_energy_adamw_wsd_h7x_xl-20260930-030055.snapshot-20261008
POOL=10000
WORKERS_PER_DEVICE=4
PYXTAL_CORES=60
# Each prediction averages this many equivalent Wyckoff descriptions, as the
# torpedo run did; one draws a random one.
AUGMENTATION_SAMPLES=8
# The ranking is stable from 32 (docs/generative_novelty_screen.md); 64 as there.
PERMUTATION_SAMPLES=64
WALLTIME="23:59:59"
ROOT=""
PILOT=0
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
    sed -n '2,29p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
    --backbone cfg|uncond  cfg: ehull_adamw_wsd_5x_cfg_cont at e_hull=0.05, w=5;
                           uncond: unconditional_5x_ehull01_h3x_xl
    --pool N               formally valid genes in the pool (default: $POOL)
    --root DIR             output root (default: \$WYFORMER_RUNS/roe_surprisal/<backbone>[_pilot])
    --pilot                200 genes, 01:59:00 walltime
    --allow-dirty          submit with uncommitted changes
    --dry-run, -n          print, do not submit
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --backbone)    BACKBONE=${2:?}; shift 2 ;;
        --pool)        POOL=${2:?}; shift 2 ;;
        --root)        ROOT=${2:?}; shift 2 ;;
        --pilot)       PILOT=1; shift ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --dry-run|-n)  DRY_RUN=1; shift ;;
        -h|--help)     usage ;;
        *)             die "unknown option $1 (see --help)" ;;
    esac
done

# Each surprisal variant is NAME|FLAGS, separated by ';'. The pool is scored under
# the density it was drawn from, and the CFG one also under the conditional model.
case "$BACKBONE" in
    cfg)
        MODEL_RUN=ehull_adamw_wsd_5x_cfg_cont-20260930-071120
        # Best cell of the parent run in docs/archive/cfg_ehull_guidance_grid_20261002.md.
        GEN_ARGS="--condition energy_above_hull=0.05 --guidance-scale 5"
        # Formal validity was 61.7% at this cell for the parent.
        OVERSAMPLE_PERCENT=200
        SURPRISAL_VARIANTS="w5|--condition energy_above_hull=0.05 --guidance-scale 5;w1|--condition energy_above_hull=0.05"
        ;;
    uncond)
        MODEL_RUN=unconditional_5x_ehull01_h3x_xl-20260930-033512
        GEN_ARGS=""
        OVERSAMPLE_PERCENT=120
        SURPRISAL_VARIANTS="plain|"
        ;;
    *) die "--backbone is cfg or uncond" ;;
esac

SUFFIX=""
ANALYSIS_ARGS="--wandb-name roe_surprisal_${BACKBONE}-$(date +%Y%m%d)"
if [ "$PILOT" -eq 1 ]; then
    WALLTIME="01:59:00"
    POOL=200
    SUFFIX="_pilot"
    # A pilot proves the pipeline; it is not a result, so it is not uploaded.
    ANALYSIS_ARGS="--budgets 25,50 --main-budget 50"
fi
INITIAL=$((POOL * OVERSAMPLE_PERCENT / 100))
MODEL="$RUNS_DIR/$MODEL_RUN"
REGRESSOR="$RUNS_DIR/$REGRESSOR_DIR"
[ -f "$MODEL/best_model_params.pt" ] || die "no checkpoint in $MODEL"
[ -f "$REGRESSOR/best_model_params.pt" ] || die "no checkpoint in $REGRESSOR"
[ -n "$ROOT" ] || ROOT="$RUNS_DIR/roe_surprisal/${BACKBONE}${SUFFIX}"

case "$REPO_DIR" in
    */.claude/worktrees/*) die "$REPO_DIR is a Claude Code worktree; use create_worktree.sh" ;;
esac
BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$ALLOW_DIRTY" -eq 0 ] && [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
    die "Uncommitted changes in $REPO_DIR. Commit first or pass --allow-dirty."
fi

mkdir -p "$LOGS_DIR" "$RUNS_DIR/.jobspec"
SPEC="$RUNS_DIR/.jobspec/roe-surprisal-${BACKBONE}-$(date +%Y%m%d-%H%M%S)${SUFFIX}.sh"
{
    echo "# Generated by scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh on $(date -Is)"
    for var in REPO_DIR BRANCH COMMIT ROOT MODEL GEN_ARGS INITIAL POOL REGRESSOR \
               SURPRISAL_VARIANTS AUGMENTATION_SAMPLES PERMUTATION_SAMPLES \
               WORKERS_PER_DEVICE PYXTAL_CORES ANALYSIS_ARGS; do
        printf '%s=%q\n' "$var" "${!var}"
    done
    printf 'REPO=%q\n' "$REPO_DIR"
} > "$SPEC"

QSUB_ARGS=(-N "wyf_roe_surprisal_${BACKBONE}${SUFFIX}" -q ai -P 11001786
           -l "select=1:ngpus=4:ncpus=64:mem=440gb" -l "walltime=$WALLTIME"
           -j oe -o "$LOGS_DIR/" -v "JOB_SPEC=$SPEC")

echo "=========================================================="
echo "roe surprisal pool : $BACKBONE${SUFFIX}"
echo "branch             : $BRANCH @ ${COMMIT:0:8}"
echo "root               : $ROOT"
echo "backbone           : $MODEL ${GEN_ARGS}"
echo "regressor          : $REGRESSOR"
echo "pool               : $POOL from $INITIAL draws"
echo "walltime           : $WALLTIME"
echo "spec               : $SPEC"
echo "=========================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "$QSUB ${QSUB_ARGS[*]} $SCRIPT_PATH"
    cat "$SPEC"
    rm -f "$SPEC"
    exit 0
fi
"$QSUB" "${QSUB_ARGS[@]}" "$SCRIPT_PATH"
