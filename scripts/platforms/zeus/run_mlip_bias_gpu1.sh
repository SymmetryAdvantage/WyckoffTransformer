#!/bin/bash
# Finish EquFlashV2, then run PET-OAM-XL on Zeus GPU 1.
# See docs/platforms/zeus/mlip_bias.md for the isolated package overlays.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
study=generated/mlip_bias_study
mkdir -p "$study"
exec 9>"$study/zeus_gpu1.lock"
flock -n 9 || { echo "A Zeus GPU 1 study worker already holds the lock." >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=1 WANDB_ENTITY=symmetry-advantage WANDB_MODE=online
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
batch_trials=${MLIP_BIAS_BATCH_TRIALS:-25}
[[ "$batch_trials" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid batch size" >&2; exit 1; }
for overlay in "$study/deps/GGNN/GGNN" "$study/deps/fairchem/fairchem" "$study/deps/pet/upet"; do
    [[ -d "$overlay" ]] || { echo "Missing model overlay: $overlay" >&2; exit 1; }
done

trial_rows() {
    if [[ -f "$1/trials.csv" ]]; then
        # Exception messages may span physical lines inside quoted CSV fields.
        # Count actual records so those errors cannot mark an arm complete early.
        .venv/bin/python -c 'import csv, sys; rows = list(csv.DictReader(open(sys.argv[1], newline=""))); print(len(rows), sum(r["status"] == "failed" and "TimeoutError" in r["error"] and float(r["seconds"]) < 1200 for r in rows))' "$1/trials.csv"
    else
        echo '0 0'
    fi
}

for arm in equflashv2 pet_oam_xl; do
    if [[ "$arm" == equflashv2 ]]; then
        model=EquFlashV2
        overlay="$repo_root/$study/deps/GGNN:$repo_root/$study/deps/fairchem"
        workers=${EQUFLASH_WORKERS:-4}
        cpus=16-23
    else
        model=PET-OAM-XL
        overlay="$repo_root/$study/deps/pet"
        workers=${PET_WORKERS:-2}
        cpus=20-23
    fi
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid worker count" >&2; exit 1; }
    read -r rows old_timeouts < <(trial_rows "$study/$arm")
    [[ "$rows" =~ ^[0-9]+$ ]] || { echo "Invalid trial count for $arm" >&2; exit 1; }
    [[ "$old_timeouts" =~ ^[0-9]+$ ]] || { echo "Invalid timeout count for $arm" >&2; exit 1; }
    retry_args=()
    if (( old_timeouts > 0 )); then
        retry_args=(--retry-timeouts)
    fi
    while (( rows < 2698 || ${#retry_args[@]} > 0 )); do
        # Finish the active batch and its upload before honouring a stop marker.
        [[ ! -e "$study/zeus_gpu1.stop" ]] || exit 0
        PYTHONPATH="$overlay" taskset -c "$cpus" .venv/bin/python scripts/run_mlip_bias_relax.py \
            --input "$study/input" --output "$study/$arm" --mlip "$model" \
            --device cuda --workers "$workers" --relax-timeout 1200 "${retry_args[@]}" \
            --max-trials "$batch_trials" --sync-every 25
        retry_args=()
        read -r rows old_timeouts < <(trial_rows "$study/$arm")
        [[ "$rows" =~ ^[0-9]+$ ]] || { echo "Invalid trial count for $arm" >&2; exit 1; }
    done
    echo "$model has a complete 2698-trial first-pass ledger."
done
