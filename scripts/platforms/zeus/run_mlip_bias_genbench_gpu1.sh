#!/bin/bash
# Score both tracks of each completed arm, sequentially, on Zeus GPU 1.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
study=generated/mlip_bias_study
exec 9>"$study/zeus_gpu1.lock"
if [[ ${MLIP_BIAS_SCORING_WAIT_FOR_LOCK:-0} == 1 ]]; then
    flock 9
else
    flock -n 9 || { echo 'Another study process holds GPU 1.' >&2; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=1 WANDB_ENTITY=symmetry-advantage WANDB_MODE=online
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MLIP_PREPROCESSOR_N_JOBS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_root/$study/deps/genbench"
scoring_workers=${MLIP_BIAS_SCORING_WORKERS:-4}
[[ "$scoring_workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid scoring worker count.' >&2; exit 1; }
[[ $(git -C "$study/deps/lemat_genbench" rev-parse HEAD) == fbf1ba4855934acb8fe87135315504899064080d ]]
if (( $# )); then
    arms=("$@")
else
    arms=(equflashv2 pet_oam_xl tece_oam_rra_1_0 equiformerv3_dens_oam)
fi
for arm in "${arms[@]}"; do
    case "$arm" in
        equflashv2) artifact=mlip-bias-9b5e07e1c31040ee:v116 ;;
        pet_oam_xl) artifact=mlip-bias-a4bbbd2f9825a3a3:v130 ;;
        tece_oam_rra_1_0) artifact=mlip-bias-55f04b6c21bc18d4:v111 ;;
        equiformerv3_dens_oam) artifact=mlip-bias-756e18ed10a3f423:v268 ;;
        esen_30m_oam) artifact=mlip-bias-e18262dd1ae93acd:v271 ;;
        tace_oam_l_first_pass) artifact=mlip-bias-8eb9d70c27c24afd:v108 ;;
        *) echo "No pinned completed artifact for $arm" >&2; exit 1 ;;
    esac
    for track in free fixed_symmetry; do
        [[ ! -e "$study/zeus_genbench_gpu1.stop" ]] || exit 0
        output="$study/metrics/$arm/$track"
        if [[ -f "$output/metrics.json" ]] && .venv/bin/python -c '
import json,sys,hashlib
from pathlib import Path
output,arm,artifact=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
metrics=json.loads((output/"metrics.json").read_text())
identity=json.loads((output/"checkpoint_manifest.json").read_text())
selected=json.loads((arm/("selected_"+metrics["track"]+".json")).read_text())
hashes={gene:hashlib.sha256((arm/row["cif"]).read_bytes()).hexdigest() for gene,row in selected.items()}
sys.exit(0 if metrics["final_results"] and metrics["arm_artifact"]==artifact and hashes==identity["selected_cif_hashes"] else 1)
' "$output" "$study/$arm" "symmetry-advantage/WyckoffTransformer/$artifact"; then
            echo "Keeping completed $arm $track metrics with unchanged inputs."
            continue
        fi
        reuse_args=()
        if [[ "$track" == fixed_symmetry ]]; then
            reuse_args=(--reuse-scores-from "$study/metrics/$arm/free")
        fi
        mkdir -p "$output"
        echo "$(date -Is) Scoring $arm $track from $artifact"
        taskset -c 16-23 .venv/bin/python scripts/run_mlip_bias_genbench.py \
            --arm "$study/$arm" --track "$track" \
            --workers "$scoring_workers" \
            --arm-artifact "symmetry-advantage/WyckoffTransformer/$artifact" \
            --lemat-root "$study/deps/lemat_genbench" --output "$output" "${reuse_args[@]}" \
            > "$output/scoring.log" 2>&1
        .venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["arm_manifest"]["mlip"],p["track"],"selected",p["selected_genes"],"coverage",p["single_point_coverage"],"final",p["final_results"])' "$output/metrics.json"
    done
done
echo "$(date -Is) Completed requested arms and both tracks."
