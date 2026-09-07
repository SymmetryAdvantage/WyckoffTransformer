#!/usr/bin/env bash
# Launch a command against the WyFormer venv inside the base PyTorch container.
#
#   scripts/run_in_singularity.sh python scripts/train.py \
#       yamls/models/lemat_bulk_ehull/ehull_adamw_wsd_5x.yaml lemat_bulk_ehull cuda
#
# The venv at $REPO_DIR/.venv is built by scripts/build_singularity_venv.sh and
# inherits the container's CUDA-enabled Torch via --system-site-packages, so it is
# only valid *inside* this image.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/scratch/users/nus/kna/WyckoffTransformer}
SIF=${SIF:-/home/users/nus/kna/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif}

# SINGULARITY_NO_EVAL keeps `singularity run` from re-parsing the arg vector
# through a shell (it otherwise chokes on parentheses in -c snippets).
export SINGULARITY_NO_EVAL=1

# /scratch and $HOME are bound automatically; add the node-local job scratch (where
# throwaway configs live) and anything the caller asks for via EXTRA_BIND.
BINDS="$REPO_DIR"
[ -d /raid ] && BINDS="$BINDS,/raid"
[ -n "${EXTRA_BIND:-}" ] && BINDS="$BINDS,$EXTRA_BIND"

exec singularity run --nv \
    --bind "$BINDS" \
    --env "VIRTUAL_ENV=$REPO_DIR/.venv" \
    --env "PATH=$REPO_DIR/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "$SIF" "$@"
