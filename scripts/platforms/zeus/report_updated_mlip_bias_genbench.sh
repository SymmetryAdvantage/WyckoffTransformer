#!/bin/bash
# Wait for all nine finished scoring attempts, then benchmark the global shared mask.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
namespace=${1:-recovery-20261006}
[[ "$namespace" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid namespace.' >&2; exit 1; }
index="generated/mlip_bias_study/metrics/$namespace/scoring_source_index.json"
while ! .venv/bin/python - "$index" <<'PY'
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('watcher','scripts/platforms/zeus/watch_mlip_bias_genbench.py')
watcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)
path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(1)
data = json.loads(path.read_text())
sys.exit(0 if set(data['arms']) == set(watcher.ARMS) else 1)
PY
do
    echo "$(date -Is) Waiting for all nine scoring source entries."
    sleep 30
done
exec 8>"${index%/*}/report.lock"
flock -n 8 || { echo 'Another reporter owns this namespace.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="" WANDB_MODE=online WANDB_ENTITY=symmetry-advantage
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/generated/mlip_bias_study/deps/genbench"
taskset -c 8-11 .venv/bin/python scripts/compute_mlip_bias_common_genbench.py --source-index "$index"
exec taskset -c 8-11 .venv/bin/python scripts/report_mlip_bias_genbench.py --source-index "$index"
