#!/bin/bash
# ---------------------------------------------------------------------------
# Launch wyformer-protocol-wandb on ASPIRE 2A via PBS utilizing a full 4-GPU node.
#
# Usage:
#     bash scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh <run-id> [options]
#
# Example:
#     bash scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh chemsys_e_hull_sg_adamw_wsd-20260916-201605
#     bash scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh v910c1fy --condition energy_above_hull=0
#     bash scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh v910c1fy --pilot
#
# HARDWARE & SCHEDULING (FULL 4-GPU NODE)
#
# ASPIRE 2A 4-GPU nodes (both in the AI partition pbs102 and Cray EX pbs101)
# are equipped with:
#   - 4x NVIDIA A100-SXM4-40GB
#   - 1x AMD EPYC 7713 64-core processor (64 physical cores, 128 SMT threads)
#   - 462 GB RAM (node_pool=4gpu)
#
# PBS resource request:
#   select=1:ngpus=4:ncpus=64:mem=440gb
#
# Queue routing:
#   Submitted to -q ai:
#     - walltime in (02:00:00, 24:00:00] with 4 GPUs routes to `aiq3`.
#     - walltime <= 02:00:00 (e.g. with --pilot) routes to `aidev`.
#
# WORKER CONCURRENCY
#
# Empirical saturation benchmark on the A100-SXM4-40GB GPU:
#   - W=1 worker/GPU: 7.7 struct/min (mean GPU util 4.3%, peak mem 837 MB)
#   - W=2 workers/GPU: 17.7 struct/min (2.30x speedup, 115% efficiency, peak mem 1.6 GB)
#   - W=4 workers/GPU: 22.0 struct/min (2.85x speedup, 71.4% efficiency, peak mem 3.3 GB)
#   - W=6 workers/GPU: 22.5 struct/min (2.91x speedup, 48.6% efficiency, peak mem 4.8 GB)
#   - W>=8 workers/GPU: throughput plateaus or drops due to CUDA kernel contention.
#
# Default workers per device is set to 4 (yielding 16 total relaxation workers
# across the 4 GPUs), operating right at the saturation knee with low contention
# and high efficiency.
# PyXtal structure generation uses 60 CPU cores (--pyxtal-cores 60).
# ---------------------------------------------------------------------------
#PBS -N wyf_protocol
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
# Execution mode inside PBS job
# ===========================================================================
run_job_payload() {
    [ -f "${JOB_SPEC:?no JOB_SPEC provided to PBS job}" ] || die "spec file not found: $JOB_SPEC"
    # shellcheck source=/dev/null
    . "$JOB_SPEC"
    SKIP_GENERATE=${SKIP_GENERATE:-0}

    cd "$REPO"
    mkdir -p "$LOGS_DIR"

    echo "=========================================================="
    echo "wyformer-protocol-wandb on full 4-GPU node"
    echo "pbs job   : ${PBS_JOBID:-<interactive>}   node: $(hostname)"
    echo "commit    : ${COMMIT:0:8} (${BRANCH})"
    echo "date      : $(date -Is)"
    echo "run id    : $RUN_ID"
    echo "output dir: $OUTPUT_DIR"
    echo "devices   : $DEVICES"
    echo "workers/dev: $WORKERS_PER_DEVICE (total relaxation workers: $(( 4 * WORKERS_PER_DEVICE )))"
    echo "pyxtal cores: $PYXTAL_CORES"
    echo "=========================================================="
    nvidia-smi || true

    # Singularity setup
    if ! command -v singularity >/dev/null 2>&1; then
        type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
        module load singularity
    fi

    # Verify container
    [ -f "$SIF" ] || die "Singularity image not found at $SIF"

    # Verify relax extra is installed in venv
    if ! bash "$REPO/scripts/platforms/aspire2a/run_in_singularity.sh" python -c "import orb_models" >/dev/null 2>&1; then
        die "orb_models is not importable in venv. Install relax extra first (docs/platforms/aspire2a/environment.md)."
    fi

    # Assemble wyformer-protocol-wandb command
    local -a CMD=(
        wyformer-protocol-wandb
        "$RUN_ID"
        --output-dir "$OUTPUT_DIR"
        --devices "$DEVICES"
        --workers-per-device "$WORKERS_PER_DEVICE"
        --pyxtal-cores "$PYXTAL_CORES"
        --gen-device "$GEN_DEVICE"
    )

    if [ -n "$N_GENES" ]; then
        CMD+=(--n-genes "$N_GENES")
    fi
    if [ -n "$TEMPERATURE" ]; then
        CMD+=(--temperature "$TEMPERATURE")
    fi
    if [ -n "$STAGES" ]; then
        CMD+=(--stages "$STAGES")
    fi
    if [ -n "$MLIP" ]; then
        CMD+=(--mlip "$MLIP")
    fi
    if [ -n "$PRERELAX_MLIP" ]; then
        CMD+=(--prerelax-mlip "$PRERELAX_MLIP")
    fi
    if [ -n "$SYSTEM_PRIOR" ]; then
        CMD+=(--system-prior "$SYSTEM_PRIOR")
    fi
    if [ -n "$LEMAT_CIF_CSV" ]; then
        CMD+=(--lemat-cif-csv "$LEMAT_CIF_CSV")
    fi
    if [ -n "$FROM_ARTIFACT" ]; then
        if [ "$FROM_ARTIFACT" = "__CONST__" ]; then
            CMD+=(--from-artifact)
        else
            CMD+=(--from-artifact "$FROM_ARTIFACT")
        fi
    fi
    if [ "$UPLOAD" -eq 0 ]; then
        CMD+=(--no-upload)
    fi
    if [ "$SKIP_GENERATE" -eq 1 ]; then
        CMD+=(--skip-generate)
    fi

    if [ ${#CONDITIONS[@]} -gt 0 ]; then
        for cond in "${CONDITIONS[@]}"; do
            CMD+=(--condition "$cond")
        done
    fi
    if [ -n "$CONDITION_VALUE" ]; then
        CMD+=(--condition-value "$CONDITION_VALUE")
    fi

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "Invoking: ${CMD[*]}"
    echo "----------------------------------------------------------"
    t_start=$(date +%s)

    bash "$REPO/scripts/platforms/aspire2a/run_in_singularity.sh" "${CMD[@]}"
    rc=$?

    t_end=$(date +%s)
    duration=$(( t_end - t_start ))
    echo "----------------------------------------------------------"
    echo "Completed with exit code $rc in ${duration}s"
    exit $rc
}

# If running inside PBS, execute the job payload
if [ -n "${PBS_JOBID:-}" ] && [ -n "${JOB_SPEC:-}" ]; then
    run_job_payload
    exit 0
fi

# ===========================================================================
# CLI submission mode (on login node or interactive session)
# ===========================================================================

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
REPO_DIR=$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)

# shellcheck source=scripts/wyformer_paths.sh
. "$REPO_DIR/scripts/wyformer_paths.sh"
LOGS_DIR=${WYFORMER_LOGS:-/scratch/users/nus/kna/WyFormer/logs}
RUNS_DIR=$(wyformer_path WYFORMER_RUNS "$REPO_DIR/runs") || exit 1
QSUB=/opt/pbs/bin/qsub
[ -x "$QSUB" ] || QSUB=$(command -v qsub || echo "qsub")

# Default values
SIF=${SIF:-/home/users/nus/kna/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif}
QUEUE="ai"
PROJECT="11001786"
WALLTIME="23:59:59"
NGPUS=4
NCPUS=64
MEM="440gb"

DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
WORKERS_PER_DEVICE=4
PYXTAL_CORES=60
GEN_DEVICE="cuda:0"

OUTPUT_DIR=""
N_GENES=""
TEMPERATURE=""
STAGES="screen,generate,relax,score"
MLIP="orb_conserv_inf"
PRERELAX_MLIP=""
SYSTEM_PRIOR=""
LEMAT_CIF_CSV=""
FROM_ARTIFACT=""
SKIP_GENERATE=0
UPLOAD=1
PILOT=0
DRY_RUN=0
ALLOW_DIRTY=0

declare -a CONDITIONS=()
declare -a EXTRA_ARGS=()
declare -a POSITIONAL=()

usage() {
    cat <<EOF
Usage:
    bash scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh <run-id> [options] [-- extra args]

Arguments:
    <run-id>                   W&B run id to evaluate (e.g. chemsys_e_hull_sg_adamw_wsd-20260916-201605)

Options:
    --output-dir DIR           Output directory (default: generated/<run-id>/protocol)
    --n-genes N                Number of genes to generate and evaluate (default: 1000)
    --condition NAME=VALUE     Condition target, e.g. energy_above_hull=0 (repeatable)
    --condition-value VAL      Shorthand for condition value
    --temperature T            Sampling temperature (default: 1.0)
    --system-prior PATH        Path to system_prior.npz
    --lemat-cif-csv PATH       Path to LeMat CIF export or dataset splits directory
    --stages STAGES            Comma-separated stages (default: screen,generate,relax,score)
    --mlip MLIP                Scoring MLIP (default: orb_conserv_inf)
    --prerelax-mlip MLIP       Pre-relaxation MLIP (e.g. nep89)
    --from-artifact [VERSION]  Re-score from existing W&B protocol artifact (default: latest)
    --skip-generate            Reuse existing wyckoff_genes.json.gz in output dir
    --no-upload                Skip write-back to W&B
    --pilot                    Short test (2h walltime, routes to aidev)
    --workers-per-gpu N        Relaxation workers per GPU (default: 4, based on saturation benchmark)
    --pyxtal-cores N           CPU processes for PyXtal structure generation (default: 60)
    --devices DEVS             CUDA devices (default: cuda:0,cuda:1,cuda:2,cuda:3)
    --walltime HH:MM:SS        PBS walltime (default: 23:59:59)
    --name NAME                PBS job name (default: wyf_proto_<run_id>)
    --allow-dirty              Allow submission with uncommitted git changes
    --dry-run, -n              Print submission commands without submitting
    -h, --help                 Show this help message

EOF
    exit 0
}

JOB_NAME=""

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir)       OUTPUT_DIR=${2:?--output-dir needs a path}; shift 2 ;;
        --n-genes)          N_GENES=${2:?--n-genes needs an integer}; shift 2 ;;
        --condition)        CONDITIONS+=("${2:?--condition needs NAME=VALUE}"); shift 2 ;;
        --condition-value)  CONDITION_VALUE=${2:?--condition-value needs a value}; shift 2 ;;
        --temperature)      TEMPERATURE=${2:?--temperature needs a float}; shift 2 ;;
        --system-prior)     SYSTEM_PRIOR=${2:?--system-prior needs a path}; shift 2 ;;
        --lemat-cif-csv)    LEMAT_CIF_CSV=${2:?--lemat-cif-csv needs a path}; shift 2 ;;
        --stages)           STAGES=${2:?--stages needs a value}; shift 2 ;;
        --mlip)             MLIP=${2:?--mlip needs a value}; shift 2 ;;
        --prerelax-mlip)    PRERELAX_MLIP=${2:?--prerelax-mlip needs a value}; shift 2 ;;
        --from-artifact)
            if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
                FROM_ARTIFACT="$2"; shift 2
            else
                FROM_ARTIFACT="__CONST__"; shift 1
            fi
            ;;
        --skip-generate)    SKIP_GENERATE=1; shift ;;
        --no-upload)        UPLOAD=0; shift ;;
        --pilot)            PILOT=1; shift ;;
        --workers-per-gpu)  WORKERS_PER_DEVICE=${2:?--workers-per-gpu needs an integer}; shift 2 ;;
        --pyxtal-cores)     PYXTAL_CORES=${2:?--pyxtal-cores needs an integer}; shift 2 ;;
        --devices)          DEVICES=${2:?--devices needs a string}; shift 2 ;;
        --walltime)         WALLTIME=${2:?--walltime needs HH:MM:SS}; shift 2 ;;
        --name)             JOB_NAME=${2:?--name needs a string}; shift 2 ;;
        --queue|-q)         QUEUE=${2:?--queue needs a value}; shift 2 ;;
        --project|-P)       PROJECT=${2:?--project needs a value}; shift 2 ;;
        --allow-dirty)      ALLOW_DIRTY=1; shift ;;
        --dry-run|-n)       DRY_RUN=1; shift ;;
        -h|--help)          usage ;;
        --)                 shift; EXTRA_ARGS+=("$@"); break ;;
        -*)                 die "unknown option $1 (see --help)" ;;
        *)                  POSITIONAL+=("$1"); shift ;;
    esac
done

[ ${#POSITIONAL[@]} -eq 1 ] || die "expected exactly 1 argument: <run-id>. Run with --help for usage."
RUN_ID="${POSITIONAL[0]}"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="generated/$RUN_ID/protocol"
fi

if [ "$PILOT" -eq 1 ]; then
    WALLTIME="01:59:00"   # Routes to aidev queue (<= 2h)
    [ -n "$N_GENES" ] || N_GENES=20
fi

# Sanity checks on git status
case "$REPO_DIR" in
    */.claude/worktrees/*)
        die "$REPO_DIR is a Claude Code worktree which may be deleted. Use scripts/platforms/aspire2a/create_worktree.sh instead."
        ;;
esac

BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")

if [ "$ALLOW_DIRTY" -eq 0 ]; then
    if [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
        die "Uncommitted changes exist in $REPO_DIR. Commit before running or pass --allow-dirty."
    fi
fi

# Name the job if not specified
if [ -z "$JOB_NAME" ]; then
    JOB_NAME=$(printf 'wyf_proto_%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-48)
fi

mkdir -p "$LOGS_DIR" "$RUNS_DIR/.jobspec"
SPEC="$RUNS_DIR/.jobspec/protocol-${RUN_ID}-$(date +%Y%m%d-%H%M%S).sh"

# Write out job specification
{
    echo "# Generated by scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh on $(date -Is)"
    printf 'REPO=%q\n'                "$REPO_DIR"
    printf 'BRANCH=%q\n'              "$BRANCH"
    printf 'COMMIT=%q\n'              "$COMMIT"
    printf 'LOGS_DIR=%q\n'            "$LOGS_DIR"
    printf 'RUNS_DIR=%q\n'            "$RUNS_DIR"
    printf 'SIF=%q\n'                 "$SIF"
    printf 'RUN_ID=%q\n'              "$RUN_ID"
    printf 'OUTPUT_DIR=%q\n'          "$OUTPUT_DIR"
    printf 'DEVICES=%q\n'             "$DEVICES"
    printf 'WORKERS_PER_DEVICE=%q\n'  "$WORKERS_PER_DEVICE"
    printf 'PYXTAL_CORES=%q\n'        "$PYXTAL_CORES"
    printf 'GEN_DEVICE=%q\n'          "$GEN_DEVICE"
    printf 'N_GENES=%q\n'             "$N_GENES"
    printf 'TEMPERATURE=%q\n'         "$TEMPERATURE"
    printf 'STAGES=%q\n'              "$STAGES"
    printf 'MLIP=%q\n'                "$MLIP"
    printf 'PRERELAX_MLIP=%q\n'       "$PRERELAX_MLIP"
    printf 'SYSTEM_PRIOR=%q\n'        "$SYSTEM_PRIOR"
    printf 'LEMAT_CIF_CSV=%q\n'       "$LEMAT_CIF_CSV"
    printf 'FROM_ARTIFACT=%q\n'       "$FROM_ARTIFACT"
    printf 'UPLOAD=%q\n'              "$UPLOAD"
    printf 'SKIP_GENERATE=%q\n'       "$SKIP_GENERATE"
    printf 'CONDITION_VALUE=%q\n'     "${CONDITION_VALUE:-}"

    if [ ${#CONDITIONS[@]} -gt 0 ]; then
        printf 'CONDITIONS=('; printf ' %q' "${CONDITIONS[@]}"; printf ' )\n'
    else
        printf 'CONDITIONS=()\n'
    fi

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        printf 'EXTRA_ARGS=('; printf ' %q' "${EXTRA_ARGS[@]}"; printf ' )\n'
    else
        printf 'EXTRA_ARGS=()\n'
    fi
} > "$SPEC"

QSUB_ARGS=(
    -N "$JOB_NAME"
    -q "$QUEUE"
    -P "$PROJECT"
    -l "select=1:ngpus=$NGPUS:ncpus=$NCPUS:mem=$MEM"
    -l "walltime=$WALLTIME"
    -j oe
    -o "$LOGS_DIR/"
    -v "JOB_SPEC=$SPEC"
)

echo "=========================================================="
echo "Submitting wyformer-protocol-wandb to ASPIRE 2A PBS"
echo "run id      : $RUN_ID"
echo "output dir  : $OUTPUT_DIR"
echo "branch      : $BRANCH @ ${COMMIT:0:8}"
echo "resources   : select=1:ngpus=$NGPUS:ncpus=$NCPUS:mem=$MEM"
echo "queue       : $QUEUE (routes to $([ "$PILOT" -eq 1 ] && echo "aidev" || echo "aiq3"))"
echo "walltime    : $WALLTIME"
echo "job name    : $JOB_NAME"
echo "workers/gpu : $WORKERS_PER_DEVICE (total relaxation workers: $(( NGPUS * WORKERS_PER_DEVICE )))"
echo "pyxtal cores: $PYXTAL_CORES"
echo "spec file   : $SPEC"
echo "=========================================================="

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] Would execute:"
    echo "  $QSUB ${QSUB_ARGS[*]} \"$SCRIPT_PATH\""
    exit 0
fi

JOB_ID=$("$QSUB" "${QSUB_ARGS[@]}" "$SCRIPT_PATH")
echo "Submitted PBS job: $JOB_ID"
echo "Log file: $LOGS_DIR/${JOB_ID}.OU"
