#!/bin/bash
# Queue one serial OOM/timeout recovery pass after both normal Zeus arms.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
study=generated/mlip_bias_study
mkdir -p "$study"
exec 9>"$study/zeus_gpu1.lock"
echo "Waiting for the normal GPU 1 launcher to finish both arms."
flock 9
[[ ! -e "$study/zeus_gpu1.stop" ]] || exit 0

export CUDA_VISIBLE_DEVICES=1 WANDB_ENTITY=symmetry-advantage WANDB_MODE=online
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
plan="$study/zeus_gpu1_recovery_plan.json"

# Record the final normal ledgers before removing any failed rows. This plan
# also lets an interrupted recovery resume without requiring complete ledgers.
.venv/bin/python - "$plan" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_mlip_bias_relax import is_oom_or_timeout, load_rows, write_json

plan = Path(sys.argv[1])
if not plan.exists():
    arms = {}
    for arm in ("equflashv2", "pet_oam_xl"):
        rows = load_rows(plan.parent / arm / "trials.csv")
        if len(rows) != 2698:
            raise SystemExit(f"Recovery requires both normal arms complete: {arm} has {len(rows)} rows")
        arms[arm] = {"normal_trial_count": len(rows), "complete": False,
                     "original_failures": [row for row in rows.values() if is_oom_or_timeout(row)]}
    write_json(plan, {"date_utc": datetime.now(timezone.utc).isoformat(),
                      "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "gpu": 1, "workers": 1, "relax_timeout_s": 3600, "arms": arms})
PY

.venv/bin/python scripts/upload_mlip_bias_compatibility.py \
    --artifact-name mlip-bias-zeus-recovery-plan-20261005 --file "$plan"

for arm in equflashv2 pet_oam_xl; do
    [[ ! -e "$study/zeus_gpu1.stop" ]] || exit 0
    read -r complete candidates < <(.venv/bin/python - "$plan" "$arm" <<'PY'
import json, sys
arm = json.load(open(sys.argv[1]))["arms"][sys.argv[2]]
print(int(arm["complete"]), len(arm["original_failures"]))
PY
    )
    if (( complete || candidates == 0 )); then
        echo "Skipping $arm: recovery complete or no OOM/timeout failures."
        continue
    fi
    if [[ "$arm" == equflashv2 ]]; then
        model=EquFlashV2
        overlay="$repo_root/$study/deps/GGNN:$repo_root/$study/deps/fairchem"
        cpus=16-23
    else
        model=PET-OAM-XL
        overlay="$repo_root/$study/deps/pet"
        cpus=20-23
    fi
    echo "Recovering $model: $candidates candidate(s), one worker, 3600 seconds per trial."
    PYTHONPATH="$overlay" taskset -c "$cpus" .venv/bin/python scripts/run_mlip_bias_relax.py \
        --input "$study/input" --output "$study/$arm" --mlip "$model" \
        --device cuda --workers 1 --relax-timeout 3600 --retry-oom-timeouts --sync-every 1
    .venv/bin/python - "$plan" "$arm" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from scripts.run_mlip_bias_relax import load_rows, write_json
plan = Path(sys.argv[1])
saved = json.loads(plan.read_text())
rows = load_rows(plan.parent / sys.argv[2] / "trials.csv")
if len(rows) != 2698:
    raise SystemExit("Recovery did not restore the full trial ledger")
saved["arms"][sys.argv[2]].update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
write_json(plan, saved)
PY
    .venv/bin/python scripts/upload_mlip_bias_compatibility.py \
        --artifact-name mlip-bias-zeus-recovery-plan-20261005 --file "$plan"
done
echo "GPU 1 serial OOM/timeout recovery finished."
