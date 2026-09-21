#!/bin/bash
# The classifier-free guidance study: the same guidance sweep at two conditioning targets,
# one after the other so that they never contend for the GPU.
#
#     nohup scripts/run_guidance_study.sh <cfg-run-id> <base-run-id> <output-root> > <log> 2>&1 &
#
# WHY TWO TARGETS
#
# `energy_above_hull = 0` is off-support: about 3% of the training rows sit there, and a
# cohort drawn at it comes out bimodal -- a spike of memorised on-hull genes plus a fatter
# unstable tail than the model produces at 0.05 (measured 2026-09-21, see
# docs/negative_data_strategy.md and the e_hull = 0.05 protocol note). At 0.05 the
# conditioned model is statistically indistinguishable from one trained only on the
# e_hull <= 0.1 slice. So 0.05 is where guidance should be judged, and 0 is where it might
# be *needed*: guidance extrapolates away from the unconditional distribution rather than
# partitioning it, which is the one lever here that could make an off-support target work.
#
# The 0.05 sweep runs first, and relaxes one more scale, because it is the primary result.
# Every arm is relaxed on the same GPU of the same host: protocol arms are not comparable
# across machines (iapetus reads ~0.025-0.030 eV/atom above aspire2a on the same structures).
set -uo pipefail

if [ $# -ne 3 ]; then
    sed -n '2,5p' "$0" >&2
    exit 64
fi
cfg_run=$1
base_run=$2
root=$3

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root" || exit 1

echo "[guidance_study $(date '+%F %T')] primary sweep at e_hull = 0.05"
CONDITION=energy_above_hull=0.05 ARM_PREFIX=guidance-c0p05 RELAX_SCALES="${RELAX_SCALES_005:-1 2 3}" \
    scripts/run_guidance_sweep.sh "$cfg_run" "$base_run" "$root/target0p05"

echo "[guidance_study $(date '+%F %T')] secondary sweep at e_hull = 0 (the off-support target)"
CONDITION=energy_above_hull=0 ARM_PREFIX=guidance RELAX_SCALES="${RELAX_SCALES_000:-1 2}" \
    scripts/run_guidance_sweep.sh "$cfg_run" "$base_run" "$root"

echo "[guidance_study $(date '+%F %T')] both sweeps done"
