#!/bin/bash
# Run a command inside the WyFormer container venv on iapetus.
#
#   scripts/platforms/iapetus/run.sh python -m pytest
#   scripts/platforms/iapetus/run.sh python scripts/train.py <model.yaml> mp_20 cuda --pilot
#   scripts/platforms/iapetus/run.sh wyformer-protocol-wandb <run-id> \
#       --output-dir generated/<run-id>/protocol \
#       --pyxtal-cores 10 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2
#   scripts/platforms/iapetus/run.sh            # interactive shell in the venv
#
# iapetus has no scheduler and its GPUs are shared, so check `nvidia-smi` before
# a long job. The relaxation stages take their cards from --devices; for
# anything that picks a device itself, select one explicitly:
#   CUDA_VISIBLE_DEVICES=0 scripts/platforms/iapetus/run.sh python scripts/train.py ...
# Inside the container that device is then cuda:0.
#
# WHY THE MOUNTS
#
# The image builds its own /home/kna rather than inheriting the host's, so
# without them the container has neither the host's caches -- and re-downloads
# the 102 MB ORB checkpoint and the LeMat-Bulk hull parquet on every invocation
# -- nor its W&B credentials, which fail as `No API key configured`.
#
# The data store, cache, runs and W&B directories are outside the checkout (see
# docs/data_store.md). The container cannot read the host's
# ~/.config/wyformer/paths.env, so they are resolved here, passed in the
# environment tier, and mounted at their host paths so the values mean the same
# inside.
#
# In a git worktree `.git` is a file naming the main checkout's .git directory by
# its host path, so that directory is mounted too, read-only, or git (and W&B's
# commit capture) fails inside the container.
set -euo pipefail

WYFORMER_IMAGE="${WYFORMER_IMAGE:-pytorch:2.14.0-cuda11.8-py312-universal}"
WYFORMER_REPO="${WYFORMER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
WYFORMER_VENV="${WYFORMER_VENV:-$WYFORMER_REPO/.venv}"

if [[ -f "$WYFORMER_REPO/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$WYFORMER_REPO/.env"
    set +a
fi

if ! docker image inspect "$WYFORMER_IMAGE" >/dev/null 2>&1; then
    echo "error: image $WYFORMER_IMAGE not found; see docs/platforms/iapetus/environment.md" >&2
    exit 1
fi
# build_venv.sh runs through here before the venv exists.
if [[ ! -d "$WYFORMER_VENV" && -z "${WYFORMER_BUILDING_VENV:-}" ]]; then
    echo "error: venv $WYFORMER_VENV not found; see docs/platforms/iapetus/environment.md" >&2
    exit 1
fi

# After a host reboot, /dev/nvidia-uvm is not created until nvidia-modprobe runs.
# Without it, NVIDIA container toolkit cannot pass UVM to the container and
# CUDA initialization in PyTorch fails with "CUDA unknown error".
if [[ ! -e /dev/nvidia-uvm ]] && command -v nvidia-modprobe >/dev/null 2>&1; then
    nvidia-modprobe -c 0 -u || true
fi

# The image's entrypoint prepends /opt/venv312/bin to PATH *after* anything we
# set, so a bare `python` is the image's interpreter, which cannot see the
# project venv's packages -- and since torch does live in the image, that fails
# far from the cause, as `ModuleNotFoundError: No module named 'sklearn'`.
# Resolve the command against the venv here instead of relying on PATH order.
# The `wyformer-*` entry points are immune either way: their shebang names the
# venv interpreter outright.
if [[ $# -eq 0 ]]; then
    set -- bash -c 'export PATH="/workspace/.venv/bin:$PATH"; exec bash'
# -L as well as -x: .venv/bin/python is a symlink to the image's interpreter,
# a path that does not exist on the host, so -x alone is false for it.
elif [[ "$1" != /* && ( -x "$WYFORMER_VENV/bin/$1" || -L "$WYFORMER_VENV/bin/$1" ) ]]; then
    set -- "/workspace/.venv/bin/$1" "${@:2}"
fi

docker_args=(
    --rm
    --runtime=nvidia
    --env NVIDIA_VISIBLE_DEVICES=all
    # Shared memory for DataLoader workers; the default 64 MB is not enough.
    --ipc=host
    --volume "$WYFORMER_REPO:/workspace"
    --workdir /workspace
    # Marks the venv as active for anything that inspects it. PATH is not set
    # here on purpose: the entrypoint would override it (see the argv rewrite
    # below).
    --env "VIRTUAL_ENV=/workspace/.venv"
)

# The host's caches and credentials, at the same paths the container's own
# $HOME uses. See WHY THE MOUNTS above.
if [[ -d "$HOME/.cache" ]]; then
    docker_args+=(--volume "$HOME/.cache:$HOME/.cache")
fi
if [[ -f "$HOME/.netrc" ]]; then
    docker_args+=(--volume "$HOME/.netrc:$HOME/.netrc:ro")
fi

# The stores, at their host paths. Created first: docker would make a missing
# mount point root-owned.
. "$WYFORMER_REPO/scripts/wyformer_paths.sh"
declare -A mounted=()
for key in WYFORMER_DATA WYFORMER_CACHE WYFORMER_RUNS WANDB_DIR; do
    dir=$(wyformer_path "$key") || exit 1
    if [[ -z "$dir" ]]; then
        echo "error: $key is not configured; see docs/platforms/iapetus/environment.md" >&2
        exit 1
    fi
    mkdir -p "$dir"
    docker_args+=(--env "$key=$dir")
    if [[ -z "${mounted[$dir]:-}" ]]; then
        docker_args+=(--volume "$dir:$dir")
        mounted[$dir]=1
    fi
done

git_common_dir=$(git -C "$WYFORMER_REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [[ -n "$git_common_dir" && "$git_common_dir" != "$WYFORMER_REPO/.git" ]]; then
    docker_args+=(--volume "$git_common_dir:$git_common_dir:ro")
fi

# Extra bind mounts, space-separated docker --volume specs; build_venv.sh uses
# this for the Warp wheel.
for spec in ${WYFORMER_EXTRA_MOUNTS:-}; do
    docker_args+=(--volume "$spec")
done

# Forward these only when actually set. Passing CUDA_VISIBLE_DEVICES="" does not
# mean "no preference", it means *no GPUs are visible*, and
# torch.cuda.is_available() silently becomes False.
for var in CUDA_VISIBLE_DEVICES WANDB_MODE WANDB_ENTITY WANDB_API_KEY HF_TOKEN; do
    if [[ -n "${!var:-}" ]]; then
        docker_args+=(--env "$var=${!var}")
    fi
done

# A TTY only when there is one to attach: without the guard every background or
# piped invocation fails with "the input device is not a TTY".
if [[ -t 0 && -t 1 ]]; then
    docker_args+=(--interactive --tty)
fi

exec docker run "${docker_args[@]}" "$WYFORMER_IMAGE" "$@"
