#!/bin/bash
# Run a command inside the WyFormer container venv on luna.
#
#   scripts/platforms/luna/run.sh python scripts/train.py <model.yaml> mp_20 cuda
#   scripts/platforms/luna/run.sh wyformer-generate out.json.gz --hf-model ...
#   scripts/platforms/luna/run.sh pytest
#   scripts/platforms/luna/run.sh bash          # interactive shell in the venv
#
# luna's GPUs are shared, and GPU 2 is faulty (uncorrectable ECC + pending row
# remap), so always pick a device explicitly:
#   CUDA_VISIBLE_DEVICES=4 scripts/platforms/luna/run.sh python scripts/train.py ...
# Inside the container that device is then cuda:0.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [[ ! -e "$WYFORMER_CONTAINER" ]]; then
    echo "error: container $WYFORMER_CONTAINER not found; run build_image.sh first" >&2
    exit 1
fi
if [[ ! -d "$WYFORMER_VENV" ]]; then
    echo "error: venv $WYFORMER_VENV not found; run build_venv.sh first" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

# --nv exposes the host driver and its libraries.
# The venv's bin goes first on PATH so `python`, `pytest` and the wyformer-*
# entry points resolve to it rather than to the container's system python.
# ~/.local/bin goes last so `uv` is available inside without shadowing anything.
env_args=(
    --env "VIRTUAL_ENV=$WYFORMER_VENV"
    --env "PATH=$WYFORMER_VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
    # CUDA's default device order is FASTEST_FIRST, which is a heuristic and not
    # guaranteed to match nvidia-smi's PCI ordering. They coincide today because
    # all eight cards are identical L40S; pin it so CUDA_VISIBLE_DEVICES=N keeps
    # meaning nvidia-smi's GPU N even if that stops being true.
    --env "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
)

# Forward these only when actually set. Passing CUDA_VISIBLE_DEVICES="" does not
# mean "no preference", it means *no GPUs are visible*, and torch.cuda.is_available()
# silently becomes False.
for var in CUDA_VISIBLE_DEVICES WANDB_MODE WANDB_ENTITY WANDB_API_KEY HF_TOKEN; do
    if [[ -n "${!var:-}" ]]; then
        env_args+=(--env "$var=${!var}")
    fi
done

exec apptainer exec --nv "${env_args[@]}" --pwd "$WYFORMER_REPO" \
    "$WYFORMER_CONTAINER" "$@"
