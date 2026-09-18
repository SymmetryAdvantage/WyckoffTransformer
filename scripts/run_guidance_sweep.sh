#!/bin/bash
# The classifier-free guidance study, end to end: cohorts and screens at every guidance
# scale, the full protocol at a pre-registered few, one table, and everything in W&B.
#
#     nohup scripts/run_guidance_sweep.sh <cfg-run-id> <base-run-id> <output-dir> > <log> 2>&1 &
#
# It waits for both W&B runs to finish training, so it can be started long before they
# do. Every arm is named with --arm (guidance-w<scale>), so neither run's headline
# protocol/ numbers or protocol_<run-id> artifact is touched. Each step is skipped when
# its output already exists, and the protocol stages resume their own logs, so running
# it again continues where it stopped. See docs/classifier_free_guidance.md.
#
# Environment:
#   SCREEN_SCALES  guidance scales screened on the CFG run   (default "0 1 1.5 2 3 5")
#   RELAX_SCALES   of those, the ones relaxed and scored      (default "1 2 3")
#   CONDITION      the generation target                      (default energy_above_hull=0)
#   GPU            nvidia-smi index to relax on               (default 1)
#   WORKERS        relaxation workers on that GPU             (default 6)
#   CPU_CORES      PyXtal cores                               (default 16)
#   POLL_S         seconds between W&B state checks           (default 900)
set -uo pipefail

if [ $# -ne 3 ]; then
    sed -n '2,6p' "$0" >&2
    exit 64
fi
cfg_run=$1
base_run=$2
out=$3
screen_scales=${SCREEN_SCALES:-"0 1 1.5 2 3 5"}
relax_scales=${RELAX_SCALES:-"1 2 3"}
condition=${CONDITION:-energy_above_hull=0}
gpu=${GPU:-1}
workers=${WORKERS:-6}
cpu_cores=${CPU_CORES:-16}
poll=${POLL_S:-900}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root" || exit 1
export WANDB_ENTITY=${WANDB_ENTITY:-symmetry-advantage}

log() { echo "[guidance_sweep $(date '+%F %T')] $*"; }

# 1.5 -> w1p5: an arm name may not contain a dot.
arm_name() { echo "guidance-w${1//./p}"; }

# run_state <run>: "<W&B state> <last validation epoch> <configured epochs>".
run_state() {
    .venv/bin/python -c "
import sys, wandb
from wyckoff_transformer import WANDB_PROJECT, wandb_run_path
run = wandb.Api().run(wandb_run_path(sys.argv[1], '$WANDB_ENTITY', WANDB_PROJECT))
print(run.state, run.summary.get('epoch', -1), run.config['optimisation']['epochs'])" "$1" \
        2>/dev/null || echo "unknown -1 0"
}

wait_finished() {
    local run=$1 state epoch epochs
    while true; do
        read -r state epoch epochs <<< "$(run_state "$run")"
        # Both: a PBS link stopped by its timeout can leave the run marked finished
        # mid-training, and the last validation is logged at epoch epochs - 1.
        if [ "$state" = finished ] && [ "$epoch" -ge $((epochs - 1)) ]; then
            log "$run has finished training ($epoch of $epochs epochs)"
            return 0
        fi
        log "$run is $state at epoch $epoch of $epochs"
        sleep "$poll"
    done
}

# oversample <scale>: how many genes to draw per gene kept.
#
# The cohort is truncated to the first --n-genes *formally valid* ones, and guidance costs
# validity: on the epoch-500 checkpoint of ehull_adamw_wsd_5x_cfg-20260916-055000 it ran
# 0.78 at w=0, 0.67 at w=1, 0.50 at w=2, 0.37 at w=3 and 0.18 at w=5. A mature model starts
# far higher (~0.95), so these are upper bounds on the cost, but a fixed 1.4 would still
# leave the high-w arms short of a cohort -- and an arm that falls short is not comparable.
oversample() {
    .venv/bin/python -c "
import sys
w = float(sys.argv[1])
print(max(1.4, min(6.0, 1.4 * (1 + 0.8 * max(0.0, w - 1)))))" "$1"
}

# screen_arm <run> <scale> <dir>: draw the cohort and screen it, on the CPU.
screen_arm() {
    local run=$1 scale=$2 dir=$3
    if [ -f "$dir/screen.json" ]; then
        log "$dir: screened already"
        return 0
    fi
    local generate=()
    if [ -f "$dir/wyckoff_genes.json.gz" ]; then
        # Sampling again would replace the cohort; screen the one that is there.
        generate=(--skip-generate)
    fi
    local over
    over=$(oversample "$scale")
    log "$dir: cohort at guidance scale $scale (oversample $over), then screen"
    CUDA_VISIBLE_DEVICES="" .venv/bin/wyformer-protocol-wandb "$run" --output-dir "$dir" \
        --condition "$condition" --guidance-scale "$scale" --arm "$(arm_name "$scale")" \
        --oversample "$over" --gen-device cpu "${generate[@]}" --stages screen --no-upload
}

# relax_arm <run> <scale> <dir>: PyXtal, relax on the GPU, score, upload under the arm.
relax_arm() {
    local run=$1 scale=$2 dir=$3
    if [ -f "$dir/funnel.json" ]; then
        log "$dir: scored already"
        return 0
    fi
    log "$dir: generate, relax, score"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 .venv/bin/wyformer-protocol-wandb "$run" \
        --output-dir "$dir" --condition "$condition" --guidance-scale "$scale" \
        --arm "$(arm_name "$scale")" --skip-generate --stages generate,relax,score \
        --pyxtal-cores "$cpu_cores" --devices cuda:0 --workers-per-device "$workers"
}

# refresh_checkpoint <run>: make the next screen fetch this run's *final* weights.
#
# ensure_run_files skips a file that is already in runs/<id>/, so a best_model_params.pt
# downloaded from the same run earlier -- mid-training, for a diagnosis -- would be used
# silently in place of the finished one. That only applies to a run trained elsewhere; the
# CFG run's own directory holds the weights its training wrote, which are authoritative.
refresh_checkpoint() {
    local run=$1 dir stamp
    dir="$(wyformer_path WYFORMER_RUNS "$repo_root/runs")/$run"
    stamp="$dir/.checkpoint_refreshed"
    if [ -f "$stamp" ] || [ ! -f "$dir/best_model_params.pt" ]; then
        return 0
    fi
    log "$run: setting aside the local best_model_params.pt so the final one is fetched"
    mv "$dir/best_model_params.pt" "$dir/best_model_params.superseded-$(date +%Y%m%d-%H%M%S).pt"
    touch "$stamp"
}

base_dir="$out/$base_run/w1"
cfg_dir() { echo "$out/$cfg_run/w$1"; }

wait_finished "$base_run"
refresh_checkpoint "$base_run"
screen_arm "$base_run" 1 "$base_dir" || log "WARNING: base screen failed"
wait_finished "$cfg_run"
for scale in $screen_scales; do
    screen_arm "$cfg_run" "$scale" "$(cfg_dir "$scale")" || log "WARNING: screen w=$scale failed"
done

relax_arm "$base_run" 1 "$base_dir" || log "WARNING: base relax failed"
for scale in $relax_scales; do
    relax_arm "$cfg_run" "$scale" "$(cfg_dir "$scale")" || log "WARNING: relax w=$scale failed"
done

arms=(--arm "base=$base_dir")
for scale in $screen_scales; do
    [ -f "$(cfg_dir "$scale")/screen.json" ] && arms+=(--arm "cfg-w$scale=$(cfg_dir "$scale")")
done
tables="$out/$cfg_run/tables"
.venv/bin/python scripts/analyse_guidance_sweep.py table --reference base "${arms[@]}" \
    --output-dir "$tables" || { log "table failed"; exit 1; }

# The relaxed arms are artifacts of their own already; this carries the table and the
# screen-only arms, which exist nowhere else.
.venv/bin/python - "$cfg_run" "$out" "$tables" "$base_run" <<'PY'
import sys
from pathlib import Path
import wandb
from wyckoff_transformer import WANDB_PROJECT
from wyckoff_transformer.paths import wandb_dir

cfg_run, out, tables, base_run = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
run = wandb.init(dir=wandb_dir(), project=WANDB_PROJECT, id=cfg_run, resume="must")
artifact = wandb.Artifact(f"guidance_sweep_{cfg_run}", type="protocol_sweep",
                          metadata={"base_run": base_run})
artifact.add_dir(str(tables), name="tables")
for run_id in (cfg_run, base_run):
    for arm in sorted((out / run_id).glob("w*")):
        for name in ("wyckoff_genes.json.gz", "screen.json", "manifest.json", "funnel.json"):
            if (arm / name).is_file():
                artifact.add_file(str(arm / name), name=f"{run_id}/{arm.name}/{name}")
run.log_artifact(artifact)
run.finish()
PY
log "done: $tables"
