#!/usr/bin/env bash
# Run one audit target under a deliberately conservative, observable CPU budget.
set -euo pipefail

target=${1:?usage: run_monitored_ehull_target.sh TARGET}
audit_dir="generated/upi73i4k/ehull_conditioning_audit"
out_dir="$audit_dir/stability"
driver="scripts/recompute_relaxed_sun.py"
python_bin=".venv/bin/python"
results_dir="/home/kna/sun-forest/external/lemat-genbench/results_final"
cpu_set="23"  # one physical core on this host
monitor_log="$out_dir/cpu_monitor_${target}.csv"

mkdir -p "$out_dir"
result_json=$(find "$results_dir" -maxdepth 1 -type f \
    -name "wyformer_upi73i4k_ehull_${target}_mace_single_mlip_*.json" \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
test -n "$result_json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
printf 'timestamp,target,pid,nlwp,pcpu,psr\n' > "$monitor_log"

taskset -c "$cpu_set" "$python_bin" "$driver" --stage preprocess \
    --cif-list "$audit_dir/final_cifs_ehull_${target}.txt" \
    --results-json "$result_json" \
    --energies-csv "$out_dir/ehull_${target}_energies.csv" \
    --structures-pkl "$out_dir/ehull_${target}_processed.pkl" \
    --hull-cache-dir ".cache/lemat_hull" \
    --workers 1 &
audit_pid=$!
trap 'kill -TERM "$audit_pid" 2>/dev/null || true' INT TERM EXIT

while kill -0 "$audit_pid" 2>/dev/null; do
    fields=$(ps -o nlwp=,pcpu=,psr= -p "$audit_pid" | xargs)
    set -- $fields
    nlwp=${1:-0}
    printf '%s,%s,%s,%s,%s,%s\n' "$(date -Is)" "$target" "$audit_pid" \
        "$nlwp" "${2:-0}" "${3:-}" >> "$monitor_log"
    # MACE maintains several inactive runtime/helper threads.  Enforce both a
    # conservative thread ceiling and a one-core CPU-usage ceiling instead of
    # mistaking those idle helpers for parallel compute.
    if [ "$nlwp" -gt 16 ] || awk "BEGIN {exit !(${2:-0} > 105)}"; then
        echo "safety stop: PID $audit_pid reached $nlwp threads / ${2:-0}% CPU" >&2
        kill -TERM "$audit_pid"
        wait "$audit_pid" || true
        exit 1
    fi
    sleep 10
done
wait "$audit_pid"
trap - INT TERM EXIT

# SUN reads the already-processed structures; retain the same one-core affinity
# even though this stage normally uses little CPU.
for energy in unrelaxed relaxed; do
    taskset -c "$cpu_set" "$python_bin" "$driver" --stage sun --energy "$energy" \
        --structures-pkl "$out_dir/ehull_${target}_processed.pkl" \
        --sun-json "$out_dir/ehull_${target}_sun_${energy}.json"
done
