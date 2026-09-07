#!/bin/bash
# ---------------------------------------------------------------------------
# Universal PBS launcher for WyFormer training: takes a model config and a
# dataset, and runs the whole training to completion on the AI partition,
# chaining itself across as many 24 h jobs as it needs.
#
#     bash scripts/train_in_pb.sh yamls/models/lemat_bulk_fmax1/gene_min_energy_adamw_wsd.yaml lemat_bulk_fmax1
#
# This is the generalisation of scripts/train_ehull_5x.pbs and
# scripts/train_ehull_ssops.pbs: everything those two hardcoded (config,
# dataset, tokeniser, W&B id file, job name) is now derived from the two
# arguments, and the parts that were argued once there are kept verbatim.
#
# ONE FILE, TWO ROLES
#
# Run from a shell it SUBMITS; run by PBS it IS the job. Which one it is doing
# is read off $JOB_SPEC, which only a qsub'd link has. The submitter resolves everything that needs the
# repo (config, tokeniser, cache paths, resource flags), writes it to a spec
# file under runs/.jobspec/, and passes only that file's path down through
# `qsub -v`. Every link of the chain sources the same spec, so a job resubmitted
# 20 hours later is submitted with exactly the resources the first one was, and
# nothing has to survive PBS's comma-separated -v quoting.
#
# WHY IT CHAINS
#
# A full config is typically far more epochs than fits in the queue's 24 h
# ceiling. Each link:
#   * pins one W&B run id (persisted in runs/.<dataset>__<config>.runid),
#   * resumes from runs/<run-id>/last_checkpoint.pt (optimiser + scheduler +
#     loader + RNG state),
#   * runs train.py under `timeout` so it stops ~30 min before the wall, hands
#     control back cleanly, and re-qsubs itself,
#   * stops when train.py exits 0 (all epochs + the config's evaluation done),
#     when the attempt cap is hit, or when an attempt crashes without getting
#     anywhere (see the crash test near the bottom).
#
# WHAT IT SETS UP BEFORE TRAINING
#
#   * the site_symmetry_ops engineers and their lookup table, when the tokeniser
#     or the model asks for them (generated package data, gitignored, ~1 min,
#     idempotent);
#   * the tensor cache for the config's tokeniser, when it is missing -- one
#     pass over the dataset under pandarallel, sized to the CPUs the job owns
#     rather than the node's core count, which is what keeps it from being
#     OOM-killed. What it costs comes out of this link's wall clock, so the
#     training timeout is what is left of the budget after it.
#
# QUEUE
#
# aiq1 is `from_route_only` -- you cannot `qsub -q aiq1` directly. Jobs go to the
# `ai` routing queue, which dispatches by (ngpus, walltime):
#   aiq1: ngpus=1          walltime in (02:00:01, 24:00:00]   <- the default here
#   aiq2: ngpus in [2,3]   aiq3: ngpus=4   aiq4: ngpus>=5      (same walltime band)
#   aidev: walltime <= 2h  ailong: walltime > 24h
# So `-q ai` + select=1:ngpus=1 + a walltime in that band is what lands in aiq1.
# (g1, on the separate pbs101 server, is an equivalent <=24 h fallback --
# `--queue g1@pbs101` from a login node; not reachable from an AI-partition
# session.)
#
# W&B: live to symmetry-advantage/WyckoffTransformer (entity/project pinned in
# scripts/train.py). Auth is ~/.netrc, visible in the container via the automatic
# $HOME bind. For offline + `wandb sync`, submit with --offline.
#
# USAGE
#
#   bash scripts/train_in_pb.sh [options] <model-config.yaml> <dataset>
#
#   --run-id ID        continue this W&B run id (default: the one pinned for
#                      this config+dataset, else a new one)
#   --fresh            start a new W&B run id, forgetting the pinned one
#   --pilot            3-epoch smoke test: own run id, no chaining, 2 h wall
#   --name NAME        PBS job name          (default: wyf_<config stem>)
#   --queue Q          (default: ai)         --project P    (default: 11001786)
#   --walltime HH:MM:SS (default: 23:59:59)  --ngpus N      (default: 1)
#   --ncpus N (default: 16)                  --mem SIZE     (default: 64gb)
#   --max-attempts N   links in the chain    (default: 8)
#   --job-budget SEC   seconds of the wall a link may use
#                      (default: walltime - 30 min)
#   --tokenise-timeout SEC  ceiling on the one-off cache build (default: 14400)
#   --device DEV       train.py device       (default: cuda)
#   --sif PATH         container image
#   --offline          run W&B offline (sync the run dir afterwards)
#   --allow-duplicate  submit even though a chain for this config+dataset is
#                      still queued or running
#   --train-arg ARG    extra argument for train.py, repeatable
#                      (e.g. --train-arg --production --train-arg --no-test)
#   --dry-run          print the qsub command and the spec, submit nothing
#
# Resubmitting the same config+dataset after a stop continues the pinned run
# from its last checkpoint with a fresh attempt budget. To abandon a run and
# start over, submit with --fresh.
# ---------------------------------------------------------------------------
#PBS -j oe

set -u

# ===========================================================================
# shared helpers
# ===========================================================================

die() { echo "error: $*" >&2; exit 1; }

# ===========================================================================
# submitter
# ===========================================================================

submit_mode() {
    local SCRIPT
    SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
    local REPO
    REPO=$(cd "$(dirname "$SCRIPT")/.." && pwd)

    # --- defaults ----------------------------------------------------------
    local SIF=${SIF:-/home/users/nus/kna/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif}
    local QUEUE=ai PROJECT=11001786
    local WALLTIME=23:59:59 NGPUS=1 NCPUS=16 MEM=64gb
    local MAX_ATTEMPTS=8 JOB_BUDGET= TOKENISE_TIMEOUT=14400
    local DEVICE=cuda JOB_NAME= RUN_ID= FRESH=0 PILOT=0 OFFLINE=0 DRY_RUN=0 ALLOW_DUP=0
    local -a TRAIN_EXTRA=()
    local -a POSITIONAL=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --run-id)           RUN_ID=${2:?--run-id needs a value}; shift 2 ;;
            --fresh)            FRESH=1; shift ;;
            --pilot)            PILOT=1; shift ;;
            --name)             JOB_NAME=${2:?--name needs a value}; shift 2 ;;
            --queue|-q)         QUEUE=${2:?--queue needs a value}; shift 2 ;;
            --project|-P)       PROJECT=${2:?--project needs a value}; shift 2 ;;
            --walltime)         WALLTIME=${2:?--walltime needs a value}; shift 2 ;;
            --ngpus)            NGPUS=${2:?--ngpus needs a value}; shift 2 ;;
            --ncpus)            NCPUS=${2:?--ncpus needs a value}; shift 2 ;;
            --mem)              MEM=${2:?--mem needs a value}; shift 2 ;;
            --max-attempts)     MAX_ATTEMPTS=${2:?--max-attempts needs a value}; shift 2 ;;
            --job-budget)       JOB_BUDGET=${2:?--job-budget needs a value}; shift 2 ;;
            --tokenise-timeout) TOKENISE_TIMEOUT=${2:?--tokenise-timeout needs a value}; shift 2 ;;
            --device)           DEVICE=${2:?--device needs a value}; shift 2 ;;
            --sif)              SIF=${2:?--sif needs a value}; shift 2 ;;
            --offline)          OFFLINE=1; shift ;;
            --allow-duplicate)  ALLOW_DUP=1; shift ;;
            --train-arg)        TRAIN_EXTRA+=("${2:?--train-arg needs a value}"); shift 2 ;;
            --dry-run|-n)       DRY_RUN=1; shift ;;
            -h|--help)          sed -n '3,/^# ---/p' "$SCRIPT" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
            --)                 shift; POSITIONAL+=("$@"); break ;;
            -*)                 die "unknown option $1 (see --help)" ;;
            *)                  POSITIONAL+=("$1"); shift ;;
        esac
    done

    [ ${#POSITIONAL[@]} -eq 2 ] || die "expected <model-config.yaml> <dataset>, got ${#POSITIONAL[@]} argument(s); see --help"

    local n v
    for n in MAX_ATTEMPTS TOKENISE_TIMEOUT NGPUS NCPUS JOB_BUDGET; do
        v=${!n}
        case "$v" in
            '') ;;                                  # only JOB_BUDGET, meaning "derive it"
            *[!0-9]*|0) die "--$(printf '%s' "$n" | tr 'A-Z_' 'a-z-') wants a positive integer, got '$v'" ;;
        esac
    done
    local CONFIG_IN=${POSITIONAL[0]} DATASET=${POSITIONAL[1]}

    # --- validate the config and derive everything hanging off it ----------
    [ -f "$CONFIG_IN" ] || die "no such config: $CONFIG_IN"
    local CONFIG_ABS
    CONFIG_ABS=$(readlink -f "$CONFIG_IN")
    # Keep it repo-relative so the job's command line reads like the docs and does
    # not depend on where it was submitted from.
    local CONFIG=${CONFIG_ABS#"$REPO"/}
    [ "$CONFIG" != "$CONFIG_ABS" ] || die "config must live inside $REPO"
    local CONFIG_STEM
    CONFIG_STEM=$(basename "$CONFIG" .yaml)

    local TOKENISER
    TOKENISER=$(read_tokeniser_name "$CONFIG_ABS") || exit 1
    [ -n "$TOKENISER" ] || die "no tokeniser.name in $CONFIG"
    local TOKENISER_YAML="$REPO/yamls/tokenisers/$TOKENISER.yaml"
    [ -f "$TOKENISER_YAML" ] || die "config asks for tokeniser '$TOKENISER', but $TOKENISER_YAML does not exist"

    local DATA_PKL="$REPO/cache/$DATASET/data.pkl.gz"
    local TENSOR_CACHE="$REPO/cache/$DATASET/tensors/$TOKENISER.safetensors"
    if [ ! -f "$TENSOR_CACHE" ] && [ ! -f "$DATA_PKL" ]; then
        die "neither the tensor cache ($TENSOR_CACHE) nor the raw dataset cache ($DATA_PKL) exists -- there is nothing to train on; cache the dataset first (scripts/cache_a_dataset.py)"
    fi
    if [ ! -f "$TENSOR_CACHE" ]; then
        echo "note: tensor cache for '$TOKENISER' is missing; the first link will build it from $DATA_PKL"
    fi

    # The ops lookup table is read at model construction time, so a missing file is a
    # hard failure rather than a silent fallback. Comments are stripped before the
    # grep; a false positive only costs the ~1 min idempotent build.
    local NEEDS_OPS_TABLE=0
    if sed 's/#.*//' "$TOKENISER_YAML" "$CONFIG_ABS" | grep -q 'site_symmetry_ops'; then
        NEEDS_OPS_TABLE=1
    fi

    # --- names, ids, budgets ------------------------------------------------
    local KEY
    KEY=$(printf '%s__%s' "$DATASET" "$CONFIG_STEM" | tr -c 'A-Za-z0-9_.-' '_')
    local RUNID_FILE JOBID_FILE

    if [ "$PILOT" -eq 1 ]; then
        # A pilot keeps its own key, so it shares neither the spec nor the pinned id
        # with the real chain -- submitting one must not reach into a chain in flight.
        KEY="$KEY.pilot"
        TRAIN_EXTRA+=(--pilot)
        MAX_ATTEMPTS=1
        [ -n "$RUN_ID" ] || RUN_ID="pilot-$(sanitise_id "$CONFIG_STEM")-$(date +%Y%m%d-%H%M%S)"
        if [ "$WALLTIME" = 23:59:59 ]; then
            WALLTIME=01:59:00      # under 2 h routes to aidev, which turns pilots around fast
        fi
    fi

    # Empty for a pilot: a smoke test must neither adopt nor pin a run id.
    RUNID_FILE=
    [ "$PILOT" -eq 1 ] || RUNID_FILE="$REPO/runs/.$KEY.runid"
    JOBID_FILE="$REPO/runs/.$KEY.jobid"

    mkdir -p "$REPO/logs" "$REPO/runs" "$REPO/runs/.jobspec"

    if [ "$FRESH" -eq 1 ]; then
        [ -n "$RUN_ID" ] && die "--fresh and --run-id contradict each other"
        if [ -n "$RUNID_FILE" ] && [ -f "$RUNID_FILE" ]; then
            echo "note: dropping the pinned run id $(cat "$RUNID_FILE") for $KEY"
            rm -f "$RUNID_FILE"
        fi
    elif [ -z "$RUN_ID" ] && [ -n "$RUNID_FILE" ] && [ -f "$RUNID_FILE" ]; then
        RUN_ID=$(cat "$RUNID_FILE")
        if [ -f "$REPO/runs/$RUN_ID/COMPLETED" ]; then
            die "$KEY is pinned to $RUN_ID, which is already COMPLETED; submit with --fresh to start a new run"
        fi
        if [ -f "$REPO/runs/$RUN_ID/last_checkpoint.pt" ]; then
            echo "continuing the pinned run $RUN_ID from its last checkpoint"
        else
            # Nothing to resume, and the id has already been logged to; the job mints a
            # new one rather than opening a second run on top of the first.
            echo "pinned run $RUN_ID has no checkpoint -- the job will start over under a fresh id"
        fi
    fi
    # Left empty, the job invents the id on the node and pins it there.

    local WALL_SECONDS
    WALL_SECONDS=$(walltime_seconds "$WALLTIME") || die "cannot parse walltime '$WALLTIME' (expected HH:MM:SS)"
    if [ -z "$JOB_BUDGET" ]; then
        # Hand the node back this far before the wall so the last checkpoint flushes,
        # W&B uploads, and the next link is queued while we still own it.
        JOB_BUDGET=$(( WALL_SECONDS - 1800 ))
    fi
    [ "$JOB_BUDGET" -gt 0 ] || die "job budget ($JOB_BUDGET s) is not positive; raise --walltime"
    [ "$JOB_BUDGET" -lt "$WALL_SECONDS" ] || die "job budget ($JOB_BUDGET s) must be under the walltime ($WALL_SECONDS s)"

    # PBS Pro takes job names up to 236 characters; 40 keeps `qstat` and the log file
    # names readable while still telling two configs of a family apart.
    [ -n "$JOB_NAME" ] || JOB_NAME=$(printf 'wyf_%s' "$CONFIG_STEM" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-40)

    # --- is a chain for this config+dataset already in flight? ---------------
    # Two live chains on one run directory would take turns overwriting each other's
    # checkpoints, and the damage only shows up in the loss curve. The id of the job
    # each link queues is kept in JOBID_FILE; qstat knows it only while it is still
    # queued, held or running.
    if [ -f "$JOBID_FILE" ] && [ "$ALLOW_DUP" -eq 0 ]; then
        local live_job
        live_job=$(cat "$JOBID_FILE")
        if "${QSUB%qsub}qstat" "$live_job" >/dev/null 2>&1; then
            die "$KEY is already in the queue as $live_job -- 'qdel $live_job' to stop it, or submit with --allow-duplicate if you really want both"
        fi
    fi

    # --- the spec every link of the chain sources ---------------------------
    # Timestamped, so a resubmission cannot rewrite the parameters a chain already in
    # flight will read on its next link; .latest.sh points at the newest for reading.
    local SPEC="$REPO/runs/.jobspec/$KEY-$(date +%Y%m%d-%H%M%S).sh"
    local -a QSUB_ARGS=(
        -N "$JOB_NAME"
        -q "$QUEUE"
        -P "$PROJECT"
        -l "select=1:ngpus=$NGPUS:ncpus=$NCPUS:mem=$MEM"
        -l "walltime=$WALLTIME"
        -j oe
        -o "$REPO/logs/"
    )

    {
        echo "# Generated by scripts/train_in_pb.sh on $(date -Is); sourced by every link of the chain."
        echo "# Regenerated on each submission -- edit the submission, not this file."
        printf 'REPO=%q\n'             "$REPO"
        printf 'SCRIPT=%q\n'           "$SCRIPT"
        printf 'SIF=%q\n'              "$SIF"
        printf 'CONFIG=%q\n'           "$CONFIG"
        printf 'DATASET=%q\n'          "$DATASET"
        printf 'CONFIG_STEM=%q\n'      "$CONFIG_STEM"
        printf 'TOKENISER=%q\n'        "$TOKENISER"
        printf 'NEEDS_OPS_TABLE=%q\n'  "$NEEDS_OPS_TABLE"
        printf 'RUNID_FILE=%q\n'       "$RUNID_FILE"
        printf 'JOBID_FILE=%q\n'       "$JOBID_FILE"
        printf 'MAX_ATTEMPTS=%q\n'     "$MAX_ATTEMPTS"
        printf 'JOB_BUDGET=%q\n'       "$JOB_BUDGET"
        printf 'TOKENISE_TIMEOUT=%q\n' "$TOKENISE_TIMEOUT"
        printf 'DEVICE=%q\n'           "$DEVICE"
        printf 'NCPUS_REQUESTED=%q\n'  "$NCPUS"
        printf 'WANDB_OFFLINE=%q\n'    "$OFFLINE"
        # printf reprints its format for every argument and once for none, so an empty
        # array has to be written out rather than expanded -- otherwise it becomes a
        # one-element array holding '', which train.py would see as an empty argument.
        if [ ${#TRAIN_EXTRA[@]} -eq 0 ]; then
            printf 'TRAIN_EXTRA=()\n'
        else
            printf 'TRAIN_EXTRA=('; printf ' %q' "${TRAIN_EXTRA[@]}"; printf ' )\n'
        fi
        printf 'QSUB_ARGS=('; printf ' %q' "${QSUB_ARGS[@]}"; printf ' )\n'
    } > "$SPEC"
    ln -sfn "$(basename "$SPEC")" "$REPO/runs/.jobspec/$KEY.latest.sh"

    local -a SUBMIT=("$QSUB" "${QSUB_ARGS[@]}" -v "JOB_SPEC=$SPEC,ATTEMPT=1${RUN_ID:+,RUN_ID=$RUN_ID}" "$SCRIPT")

    echo "config    : $CONFIG"
    echo "dataset   : $DATASET"
    echo "tokeniser : $TOKENISER"
    echo "run id    : ${RUN_ID:-<assigned by the first link>}"
    echo "spec      : $SPEC"
    echo "job name  : $JOB_NAME   queue: $QUEUE   walltime: $WALLTIME (budget ${JOB_BUDGET}s)   attempts: $MAX_ATTEMPTS"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "--- spec ---"
        cat "$SPEC"
        echo "--- qsub (not submitted) ---"
        printf '%q ' "${SUBMIT[@]}"; echo
        return 0
    fi

    [ -x "$QSUB" ] || command -v qsub >/dev/null 2>&1 || die "qsub not found -- submit from a login node"
    local JOB_ID
    JOB_ID=$("${SUBMIT[@]}") || die "qsub failed"
    echo "$JOB_ID" > "$JOBID_FILE"
    echo "submitted : $JOB_ID"
    echo "log       : $REPO/logs/$JOB_NAME.o${JOB_ID%%.*}"
}

# Read tokeniser.name out of a model config. PyYAML where it is importable, and a
# block-scoped awk fallback for shells that have neither -- the field is a plain
# scalar under a top-level `tokeniser:` key in every model config.
read_tokeniser_name() {
    local config=$1 name
    if name=$(python3 -c '
import sys, yaml
with open(sys.argv[1]) as f:
    print(yaml.safe_load(f)["tokeniser"]["name"])
' "$config" 2>/dev/null) && [ -n "$name" ]; then
        printf '%s\n' "$name"
        return 0
    fi
    awk '
        /^[^[:space:]#]/ { in_block = ($0 ~ /^tokeniser:/) }
        in_block && $1 == "name:" { gsub(/^[[:space:]]*name:[[:space:]]*/, ""); gsub(/["\x27]/, ""); sub(/[[:space:]]*#.*/, ""); print; exit }
    ' "$config"
}

sanitise_id() { printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-'; }

# Has anything been logged to W&B under this run id? Its run directory is named
# <mode>-<timestamp>-<id>, and survives both online and offline runs.
wandb_dir_exists() {
    local d
    for d in "$1"/wandb/*-"$2"; do
        [ -e "$d" ] && return 0
    done
    return 1
}

walltime_seconds() {
    printf '%s' "$1" | awk -F: '
        { for (i = 1; i <= NF; i++) if ($i !~ /^[0-9]+$/) exit 1 }
        NF == 3 { print $1 * 3600 + $2 * 60 + $3; exit }
        NF == 2 { print $1 * 60 + $2; exit }
        NF == 1 { print $1; exit }
        { exit 1 }
    '
}

# ===========================================================================
# the job itself
# ===========================================================================

job_mode() {
    local JOB_START
    JOB_START=$(date +%s)

    [ -f "$JOB_SPEC" ] || die "job spec $JOB_SPEC is gone -- resubmit from a shell to regenerate it"
    # shellcheck disable=SC1090
    source "$JOB_SPEC"

    local ATTEMPT=${ATTEMPT:-1}
    local MIN_TRAIN_SECONDS=1800   # below this, chain instead of starting a stub epoch
    # Pandarallel sizes its pool from the machine's *physical cores*, not the cgroup: on a
    # 128-core node it forks 128 workers over a multi-GB frame and the job is OOM-killed
    # (rc 137). PBS exports NCPUS as what the job actually owns.
    local TOKENISE_JOBS=${NCPUS:-$NCPUS_REQUESTED}

    cd "$REPO" || die "cannot cd to $REPO"
    mkdir -p logs runs

    # --- the W&B run id this chain is pinned to -----------------------------
    if [ -n "${RUN_ID:-}" ]; then
        :                                       # passed down the chain, or by the submitter
    elif [ -n "$RUNID_FILE" ] && [ -f "$RUNID_FILE" ]; then
        RUN_ID=$(cat "$RUNID_FILE")
    else
        RUN_ID="$(sanitise_id "$CONFIG_STEM")-$(date +%Y%m%d-%H%M%S)"
    fi
    local RUN_DIR="$REPO/runs/$RUN_ID"

    if [ -f "$RUN_DIR/COMPLETED" ]; then
        echo "run $RUN_ID is already marked COMPLETED -- nothing to do"
        exit 0
    fi

    # --- resume that id, or mint a new one ----------------------------------
    # An id is resumable only while a checkpoint stands behind it. Without one there is
    # nothing to continue, and reusing the id would be worse than useless: wandb.init
    # without resume "always starts a new run", so a second link under the same id opens
    # a second run on top of the first and their histories interleave. So: resume when
    # there is a checkpoint; otherwise, if anything was already logged under this id (a
    # run directory, or a wandb run directory carrying it), it is spent -- mint a fresh
    # one and re-pin. A first link, whose id nothing has touched yet, keeps it.
    local -a RESUME_ARGS=()
    local RESUME_NOTE="fresh run"
    if [ -f "$RUN_DIR/last_checkpoint.pt" ]; then
        RESUME_ARGS=(--resume "$RUN_ID")
        RESUME_NOTE="resuming from $RUN_DIR/last_checkpoint.pt"
    elif [ -d "$RUN_DIR" ] || wandb_dir_exists "$REPO" "$RUN_ID"; then
        local spent=$RUN_ID
        RUN_ID="$(sanitise_id "$CONFIG_STEM")-$(date +%Y%m%d-%H%M%S)"
        RESUME_NOTE="$spent got nowhere (no checkpoint) -> starting over as $RUN_ID"
        # It holds no checkpoint, and train.py refuses to mkdir over an existing dir.
        rm -rf "$RUN_DIR"
        RUN_DIR="$REPO/runs/$RUN_ID"
    fi
    [ -n "$RUNID_FILE" ] && echo "$RUN_ID" > "$RUNID_FILE"

    echo "=========================================================="
    echo "attempt   : $ATTEMPT / $MAX_ATTEMPTS"
    echo "config    : $CONFIG"
    echo "dataset   : $DATASET"
    echo "tokeniser : $TOKENISER"
    echo "wandb id  : $RUN_ID"
    echo "run dir   : $RUN_DIR"
    echo "state     : $RESUME_NOTE"
    echo "pbs job   : ${PBS_JOBID:-<interactive>}   node: $(hostname)"
    echo "start     : $(date -Is)"
    echo "=========================================================="
    nvidia-smi || true

    # --- environment for the container -------------------------------------
    # Put singularity on PATH without relying on the module system in a batch shell.
    if ! command -v singularity >/dev/null 2>&1; then
        export PATH="/app/apps/singularity/sup/squashfuse/0.6.1/bin:/app/apps/singularity/3.10.0/bin:$PATH"
    fi
    command -v singularity >/dev/null 2>&1 || { source /etc/profile.d/modules.sh 2>/dev/null && module load singularity; }
    singularity --version

    export REPO_DIR="$REPO" SIF
    export SINGULARITYENV_WANDB_RUN_ID="$RUN_ID"
    export SINGULARITYENV_WANDB__SERVICE_WAIT=300
    export SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
    export SINGULARITYENV_OMP_NUM_THREADS=${NCPUS:-$NCPUS_REQUESTED}
    if [ "$WANDB_OFFLINE" -eq 1 ]; then
        export SINGULARITYENV_WANDB_MODE=offline   # `wandb sync $RUN_DIR` afterwards
    fi

    # `timeout` execs its argument, so every call below goes through the launcher
    # script directly rather than a shell function it could not exec.
    local -a IN_CONTAINER=(bash "$REPO/scripts/run_in_singularity.sh")

    # --- one-off: the operations engineer and its lookup table --------------
    local OPS_TABLE="$REPO/src/wyckoff_transformer/engineers/site_symmetry_ops_id_table.json"
    if [ "$NEEDS_OPS_TABLE" -eq 1 ] && [ ! -f "$OPS_TABLE" ]; then
        echo "site_symmetry_ops_id table missing -> building the engineers ($(date -Is))"
        "${IN_CONTAINER[@]}" python -c "
from wyckoff_transformer.preprocess_wychoffs import (
    build_site_symmetry_ops_engineer, build_site_symmetry_ops_id_engineer)
build_site_symmetry_ops_engineer()
build_site_symmetry_ops_id_engineer()
print('engineers built')
" || die "failed to build the site_symmetry_ops engineers"
    fi

    # --- one-off: the tensor cache for this tokeniser -----------------------
    local TENSOR_CACHE="$REPO/cache/$DATASET/tensors/$TOKENISER.safetensors"
    if [ ! -f "$TENSOR_CACHE" ]; then
        echo "tensor cache missing -> tokenising $DATASET with $TOKENISER ($(date -Is))"
        echo "this is a one-off pass over the whole dataset; later links skip it"
        local tok_rc
        timeout --signal=TERM --kill-after=180 "$TOKENISE_TIMEOUT" \
            "${IN_CONTAINER[@]}" python scripts/tokenise_a_dataset.py \
                "$DATASET" "yamls/tokenisers/$TOKENISER.yaml" --new-tokenizer --n-jobs "$TOKENISE_JOBS"
        tok_rc=$?
        if [ "$tok_rc" -ne 0 ] || [ ! -f "$TENSOR_CACHE" ]; then
            # A half-written cache would poison every later link, so clear it.
            rm -f "$TENSOR_CACHE"
            die "tokenisation failed (rc $tok_rc) -> hard stop, nothing to train on"
        fi
        echo "tensor cache built: $(du -h "$TENSOR_CACHE" | cut -f1)   ($(date -Is))"
    fi

    # --- what is left of this job's budget for training ---------------------
    local ELAPSED=$(( $(date +%s) - JOB_START ))
    local TRAIN_TIMEOUT=$(( JOB_BUDGET - ELAPSED ))
    echo "spent $ELAPSED s so far; $TRAIN_TIMEOUT s left for training this link"

    if [ "$TRAIN_TIMEOUT" -lt "$MIN_TRAIN_SECONDS" ]; then
        echo "too little of the wall left to train usefully -> chaining now"
        resubmit "$RUN_ID" "$ATTEMPT" || exit 1
        exit 0
    fi

    # The mark a crash below is judged against: the mtime of the checkpoint this attempt
    # starts from, or 0 when there is nothing to resume.
    local CKPT_MTIME_BEFORE
    CKPT_MTIME_BEFORE=$(stat -c %Y "$RUN_DIR/last_checkpoint.pt" 2>/dev/null || echo 0)

    # --- train --------------------------------------------------------------
    local rc
    timeout --signal=TERM --kill-after=180 "$TRAIN_TIMEOUT" \
        "${IN_CONTAINER[@]}" python scripts/train.py "$CONFIG" "$DATASET" "$DEVICE" \
            --run-path "$REPO/runs" \
            ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"} \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    rc=$?

    echo "----------------------------------------------------------"
    echo "train.py exit code: $rc   ($(date -Is))"

    # --- completion / chaining ---------------------------------------------
    if [ "$rc" -eq 0 ]; then
        # The marker is what keeps a later submission from re-running a finished run,
        # so it has to land even if the run directory is somewhere unexpected.
        mkdir -p "$RUN_DIR"
        touch "$RUN_DIR/COMPLETED"
        echo "TRAINING COMPLETE for $RUN_ID"
        exit 0
    fi

    # rc 124 = our timeout fired (expected once per link): the link had its whole slot,
    # so chain it. Any other non-zero is a crash, and what makes chaining a crash safe is
    # that the attempt got somewhere first -- the existence of a checkpoint does not say
    # that, because a resume that dies on its first line is still holding the one the
    # previous link wrote. Requiring a NEW checkpoint is what keeps a deterministic
    # failure from spending the whole attempt budget in a few minutes, which is how both
    # ehull chains ended: eight links, each dying in the same place on the resume.
    if [ "$rc" -ne 124 ]; then
        local ckpt_mtime_after
        ckpt_mtime_after=$(stat -c %Y "$RUN_DIR/last_checkpoint.pt" 2>/dev/null || echo 0)
        if [ "$ckpt_mtime_after" -le "$CKPT_MTIME_BEFORE" ]; then
            echo "crashed without writing a checkpoint -> hard stop (see log above)" >&2
            echo "once the cause is fixed, resubmitting this config+dataset resumes $RUN_ID with a fresh budget" >&2
            exit "$rc"
        fi
        echo "crashed, but this attempt wrote a checkpoint -> continuing from it"
    fi

    resubmit "$RUN_ID" "$ATTEMPT" || exit "$rc"
}

# Queue the next link of the chain. Returns non-zero when the cap is reached, so the
# caller can exit with the failure it was already carrying.
resubmit() {
    local run_id=$1 attempt=$2
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "attempt cap ($MAX_ATTEMPTS) reached -> stopping; resubmit the same config+dataset to continue" >&2
        return 1
    fi
    echo "resubmitting to continue from checkpoint (attempt $((attempt + 1)))"
    local next_id
    next_id=$("$QSUB" "${QSUB_ARGS[@]}" \
        -v "JOB_SPEC=$JOB_SPEC,RUN_ID=$run_id,ATTEMPT=$((attempt + 1))" \
        "$SCRIPT") || { echo "qsub failed -- the chain stops here" >&2; return 1; }
    echo "$next_id" > "$JOBID_FILE"
    echo "next link : $next_id"
}

QSUB=${QSUB:-/opt/pbs/bin/qsub}
[ -x "$QSUB" ] || QSUB=qsub

# Which role this invocation plays: JOB_SPEC is set only by the qsub'd links (the
# submitter passes it through -v), so an interactive PBS session -- where
# PBS_ENVIRONMENT is set but no spec is -- still submits rather than trying to train.
if [ -n "${JOB_SPEC:-}" ]; then
    job_mode
else
    submit_mode "$@"
fi
