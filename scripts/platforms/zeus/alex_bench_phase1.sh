#!/usr/bin/env bash
# Phase 1 of the alex-mp-20 benchmark study: which sampling setting per generator.
#
# Fire-discipline only -- it removes duplicate and known genes and nothing else, so it
# compares settings without letting the energy predictor in -- at BUDGET genes per
# setting. Two lanes share GPU 1, so one setting's DiffCSP++ overlaps the other's
# relaxation and scoring. Usage:
#   nohup scripts/platforms/zeus/alex_bench_phase1.sh > $STORE/alex_bench/phase1.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDGET="${BUDGET:-2000}" ARMS="fire-discipline" CPUS="${CPUS:-10}"

UNCOND=uncond_adamw_wsd_5x-20260929-143845
COND=ehull_adamw_wsd_5x-20260929-143848
CFG=ehull_adamw_wsd_5x_cfg-20260929-150414

# name|run|sampling flags
SETTINGS=(
    "uncond_t1|$UNCOND|"
    "cond_e0|$COND|--condition energy_above_hull=0.0"
    "cond_e0p025|$COND|--condition energy_above_hull=0.025"
    "cond_e0p05|$COND|--condition energy_above_hull=0.05"
)
for target in 0.0 0.025 0.05; do
    for w in 2 4 6; do
        tag="${target/./p}"
        [[ "$target" == "0.0" ]] && tag="0"
        SETTINGS+=("cfg_e${tag}_w${w}|$CFG|--condition energy_above_hull=$target --guidance-scale $w")
    done
done

lane() {  # every second setting, starting at $1
    local i
    for (( i = $1; i < ${#SETTINGS[@]}; i += 2 )); do
        IFS='|' read -r name run flags <<< "${SETTINGS[$i]}"
        SETTING="$name" GEN_RUN="$run" GEN_ARGS="$flags" \
            "$HERE/run_alex_bench_setting.sh" || echo "[lane $1] $name FAILED"
    done
}

lane 0 &
lane 1 &
wait
echo "phase 1 done"
