#!/bin/bash
# Run a multi-day WyFormer training on zeus to completion, resuming it after a crash.
#
#     nohup scripts/platforms/zeus/train_supervised.sh <config.yaml> <dataset> <gpu> [run-id] \
#         > <log> 2>&1 &
#
# zeus has no scheduler, so nothing restarts a run that dies -- a colleague's job taking
# the card's memory, a driver fault, a reboot. This loop is the zeus counterpart of the
# PBS chain in scripts/platforms/aspire2a/train_in_pbs.sh, reduced to what a single host
# needs: one W&B run id, and `scripts/train.py --resume` from the run's own
# last_checkpoint.pt whenever an attempt exits non-zero.
#
#   <gpu>     the nvidia-smi index; the job sees it as its only card and trains on `cuda`.
#   [run-id]  default <config stem>-<YYYYmmdd-HHMMSS>, the id scheme train_in_pbs.sh uses.
#             Pass an existing id to continue a run this script started earlier.
#
# It refuses, rather than guesses, in the two situations where guessing loses work or
# fabricates a run:
#   * an attempt died before its first checkpoint, but W&B already holds the id -- starting
#     it again would log a second training into the same run;
#   * the checkpoint epoch did not move across MAX_STALLED consecutive failed attempts, i.e.
#     a crash loop.
#
# Environment: MAX_ATTEMPTS (default 30), MAX_STALLED (default 3), RETRY_DELAY_S (300).
# Docs: docs/platforms/zeus/usage.md
set -uo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    sed -n '2,5p' "$0" >&2
    exit 64
fi
config=$1
dataset=$2
gpu=$3
stem=$(basename "${config%.*}")
run_id=${4:-${stem}-$(date +%Y%m%d-%H%M%S)}
max_attempts=${MAX_ATTEMPTS:-30}
max_stalled=${MAX_STALLED:-3}
retry_delay=${RETRY_DELAY_S:-300}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root" || exit 1
# shellcheck source=scripts/wyformer_paths.sh
. scripts/wyformer_paths.sh
runs_dir=$(wyformer_path WYFORMER_RUNS "$repo_root/runs") || exit 1
checkpoint="$runs_dir/$run_id/last_checkpoint.pt"

log() { echo "[train_supervised $(date '+%F %T')] $*"; }

checkpoint_epoch() {
    [ -f "$checkpoint" ] || { echo -1; return; }
    .venv/bin/python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['epoch'])" \
        "$checkpoint" 2>/dev/null || echo -1
}

log "run id $run_id: $config on $dataset, GPU $gpu; checkpoint $checkpoint"
stalled=0
last_epoch=$(checkpoint_epoch)
for attempt in $(seq 1 "$max_attempts"); do
    if [ "$last_epoch" -ge 0 ]; then
        log "attempt $attempt: resuming from epoch $last_epoch"
        CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/train.py "$config" "$dataset" cuda \
            --resume "$run_id"
    else
        probe_status=1
        if [ "$attempt" -gt 1 ]; then
            .venv/bin/python -m wyckoff_transformer.cli.resume_probe "$run_id"
            probe_status=$?
        fi
        if [ "$probe_status" -eq 0 ]; then
            log "attempt $attempt: no local checkpoint, but W&B holds one; resuming from it"
            CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/train.py "$config" "$dataset" cuda \
                --resume "$run_id"
        elif [ "$attempt" -gt 1 ]; then
            log "attempt $((attempt - 1)) died before writing a checkpoint, and run $run_id" \
                "may already exist on W&B (probe status $probe_status). Refusing to start it" \
                "over; delete or rename the W&B run and relaunch."
            exit 2
        else
            log "attempt $attempt: starting run $run_id"
            # The name as well as the id, so the run reads the same in the W&B UI as the
            # PBS chain's do.
            WANDB_RUN_ID=$run_id WANDB_NAME=$run_id CUDA_VISIBLE_DEVICES=$gpu \
                .venv/bin/python scripts/train.py "$config" "$dataset" cuda
        fi
    fi
    status=$?
    if [ "$status" -eq 0 ]; then
        log "run $run_id finished"
        exit 0
    fi
    epoch=$(checkpoint_epoch)
    if [ "$epoch" -le "$last_epoch" ]; then
        stalled=$((stalled + 1))
    else
        stalled=0
    fi
    log "attempt $attempt exited with status $status at checkpoint epoch $epoch" \
        "(previous $last_epoch; $stalled stalled)"
    if [ "$stalled" -ge "$max_stalled" ]; then
        log "no progress across $stalled failed attempts; giving up"
        exit 3
    fi
    last_epoch=$epoch
    sleep "$retry_delay"
done
log "attempt budget of $max_attempts exhausted"
exit 4
