#!/bin/bash
# Shared settings for the WyFormer Apptainer environment on luna.
# Source this; do not execute it.

# The container image tag. See build_image.sh for why this exact tag.
export WYFORMER_IMAGE_TAG="${WYFORMER_IMAGE_TAG:-pytorch/pytorch:2.14.0-cuda12.6-cudnn9-devel}"

# The built container. This is a *sandbox directory*, not a .sif -- see
# build_image.sh. apptainer exec accepts either, so this path works both ways
# if a .sif is ever produced.
export WYFORMER_CONTAINER="${WYFORMER_CONTAINER:-$HOME/containers/pytorch-2.14.0-cuda12.6}"

# The repository checkout, i.e. the parent of scripts/platforms/luna.
WYFORMER_REPO_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WYFORMER_REPO="${WYFORMER_REPO:-$WYFORMER_REPO_DEFAULT}"

# The venv lives in the repo but is built by, and only usable inside, the
# container: its interpreter points at the container's /usr/bin/python.
export WYFORMER_VENV="${WYFORMER_VENV:-$WYFORMER_REPO/.venv-luna}"

# /home is a local RAID array and is bind-mounted into the container by
# default, so the checkout and the caches need no explicit --bind.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/.apptainer_tmp}"

# The container image ships its own uv, which is what the build normally uses.
# This is only a fallback for an image that does not have one.
export UV="${UV:-$HOME/.local/bin/uv}"
