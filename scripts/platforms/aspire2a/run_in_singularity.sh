#!/usr/bin/env bash
# Launch a command against the WyFormer venv inside the base PyTorch container.
#
#   scripts/platforms/aspire2a/run_in_singularity.sh python scripts/train.py \
#       yamls/models/lemat_bulk_fmax1/gene_min_energy_adamw_wsd.yaml lemat_bulk_fmax1_stress cuda
#
# The main venv at /home/project/11001786/WyFormer/WyckoffTransformer/.venv is built by
# scripts/build_singularity_venv.sh and inherits the container's CUDA-enabled Torch via
# --system-site-packages, so it is only valid *inside* this image.
#
# Worktrees (located in /home/users/nus/kna/scratch/WyFormer/worktrees/<name>) reuse this
# same venv: REPO_DIR defaults to the enclosing repository/worktree root, and PYTHONPATH
# points to $REPO_DIR/src so the worktree's own code is loaded rather than the main
# checkout's src recorded in the shared venv's editable install.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
MAIN_REPO="/home/project/11001786/WyFormer/WyckoffTransformer"
SIF=${SIF:-/home/users/nus/kna/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif}

# Determine which venv to use: if REPO_DIR has a .venv (directory or symlink), use it;
# otherwise fall back to the shared venv in the main repository.
VENV_DIR=${VENV_DIR:-}
if [ -z "$VENV_DIR" ]; then
    if [ -d "$REPO_DIR/.venv" ]; then
        VENV_DIR="$REPO_DIR/.venv"
    elif [ -d "$MAIN_REPO/.venv" ]; then
        VENV_DIR="$MAIN_REPO/.venv"
    else
        VENV_DIR="$MAIN_REPO/.venv"
    fi
fi

# SINGULARITY_NO_EVAL keeps `singularity run` from re-parsing the arg vector
# through a shell (it otherwise chokes on parentheses in -c snippets).
export SINGULARITY_NO_EVAL=1

# The module system is not always initialised in a batch shell, so source it first.
if ! command -v singularity >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load singularity
fi

# /scratch and $HOME are bound automatically by ASPIRE 2A Singularity config.
# /home/project and /data/projects are NOT bound automatically and MUST be explicitly bound
# so the repository, data store, and cache are accessible inside the container.
BINDS="$REPO_DIR"
[ -d /home/project ] && BINDS="$BINDS,/home/project"
[ -d /data/projects ] && BINDS="$BINDS,/data/projects"
[ -d /raid ] && BINDS="$BINDS,/raid"
[ -n "${EXTRA_BIND:-}" ] && BINDS="$BINDS,$EXTRA_BIND"

# Ensure current checkout's src is prioritized on PYTHONPATH so worktrees
# import their own code rather than the main repo's src recorded in the shared venv.
CONTAINER_PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec singularity run --nv \
    --bind "$BINDS" \
    --env "VIRTUAL_ENV=$VENV_DIR" \
    --env "PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "PYTHONPATH=$CONTAINER_PYTHONPATH" \
    "$SIF" "$@"
