#!/bin/bash
# Build (or rebuild) the checkout's .venv inside the iapetus container, and link
# CLAUDE.local.md. Run it from the checkout it is for -- the main one or a worktree:
#
#   scripts/platforms/iapetus/build_venv.sh
#
# Every checkout needs its own venv: the editable install and the scripts'
# shebangs name /workspace, which run.sh binds to the checkout it lives in, so
# a venv only ever imports its own checkout's src.
#
# The uv cache is the host's ~/.cache/uv, which run.sh mounts, so a worktree's
# build downloads nothing the main checkout's already fetched.
#
# Afterwards, installs the CPU-only Warp wheel that ORB needs on this driver,
# when the wheel is present, and fails if Triton is importable.
# Docs: docs/platforms/iapetus/environment.md
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
warp_wheels=${WYFORMER_WARP_WHEELS:-/mnt/hdd/kna/pytorch-research/archive/dist-warp}
warp_wheel=warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl

warp_step=true
if [[ -f "$warp_wheels/$warp_wheel" ]]; then
    export WYFORMER_EXTRA_MOUNTS="$warp_wheels:/opt/dist-warp:ro"
    warp_step="UV_LINK_MODE=copy uv pip install --python .venv/bin/python --no-deps --force-reinstall /opt/dist-warp/$warp_wheel"
else
    echo "warning: $warp_wheels/$warp_wheel not found; ORB will need the CPU-only Warp wheel installed by hand" >&2
fi

WYFORMER_BUILDING_VENV=1 exec "$repo/scripts/platforms/iapetus/run.sh" bash -c "
    set -euo pipefail
    REPO_DIR=/workspace WYFORMER_PLATFORM=iapetus UV=/usr/local/bin/uv \
        BASE_PYTHON=/opt/venv312/bin/python UV_CACHE_DIR=$HOME/.cache/uv \
        bash scripts/build_singularity_venv.sh
    $warp_step
    # The K20c is below Triton's minimum compute capability: Triton must not be
    # importable here, from the venv or the image.
    .venv/bin/python -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec(\"triton\") is not None and \"error: triton is installed; uninstall it -- see docs/platforms/iapetus/environment.md\")'
"
