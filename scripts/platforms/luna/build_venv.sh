#!/bin/bash
# Build the WyFormer venv *inside* the container on luna.
#
# Two things make this different from a plain `uv sync`:
#
# 1. The venv reuses the container's torch instead of installing its own.
#    That matters for more than disk. `uv pip install` inspects only the venv's
#    own site-packages -- it ignores `include-system-site-packages` -- so left
#    alone it resolves torch from PyPI and installs torch 2.14.0 with **CUDA 13**
#    wheels, which luna's CUDA 12.4 driver cannot run. reuse_container_packages.py
#    copies the container CUDA stack's *.dist-info metadata into the venv (which
#    is what uv reads to decide a package is installed) and writes
#    container-constraints.txt pinning those versions.
#
# 2. Versions for everything else come from the project's uv.lock, via
#    `uv export`, with the CUDA stack filtered out. Without this, a fresh
#    resolution picks the newest of everything and quietly degrades the result:
#    pandas 3.0.5 is chosen, matminer 0.10.1 requires pandas<3, so matminer is
#    backtracked to 0.8.0, which imports scipy.special.sph_harm -- removed in
#    scipy 1.17 -- and the test suite fails to collect.
#
#    `uv sync` is not used because it manages the venv strictly and would
#    replace the container's torch with the lock's CPU torch 2.11.0.
#
# The venv's interpreter points at the container's /usr/bin/python, so the venv
# only works inside the container. Use run.sh to enter it.
#
# The cdvae extra is deliberately not installed: it pulls torch-scatter and
# torch-sparse, which compile against torch and take a long time. Add it with
#   scripts/platforms/luna/run.sh uv pip install -e ".[cdvae]"
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRAS="${WYFORMER_EXTRAS:-dev,relax,research}"

if [[ ! -e "$WYFORMER_CONTAINER" ]]; then
    echo "error: container $WYFORMER_CONTAINER not found; run build_image.sh first" >&2
    exit 1
fi

rm -rf "$WYFORMER_VENV"

apptainer exec --nv "$WYFORMER_CONTAINER" bash -euo pipefail -c '
    REPO="$1"; HOST_UV="$2"; VENV="$3"; PLATFORM_DIR="$4"; EXTRAS="$5"
    cd "$REPO"

    # The PyTorch image ships uv. Prefer it, so the build needs nothing from the
    # host; fall back to a host install if a future image tag drops it.
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    elif [[ -x "$HOST_UV" ]]; then
        UV="$HOST_UV"
    else
        echo "error: no uv in the container and none at $HOST_UV" >&2
        echo "install one with:" >&2
        echo "  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=\$HOME/.local/bin sh" >&2
        exit 1
    fi
    echo "==> Using uv at $UV ($("$UV" --version))"

    # Hardlinking from the uv cache into the container-visible venv is not
    # reliable across the bind boundary.
    export UV_LINK_MODE=copy

    EXTRA_FLAGS=()
    for extra in ${EXTRAS//,/ }; do EXTRA_FLAGS+=(--extra "$extra"); done

    echo "==> Creating venv at $VENV with the container python"
    "$UV" venv --system-site-packages --python "$(command -v python)" "$VENV"

    echo
    echo "==> Registering the container CUDA stack so uv treats it as installed"
    "$VENV/bin/python" "$PLATFORM_DIR/reuse_container_packages.py" --roots torch \
        --constraints-out "$VENV/container-constraints.txt"

    echo
    echo "==> Deriving the remaining versions from the project uv.lock"
    "$UV" export --frozen --no-hashes --no-header --no-annotate --no-emit-project \
        "${EXTRA_FLAGS[@]}" -o "$VENV/lock-constraints.raw"
    # Drop the CUDA stack: those come from the container, and the lock pins the
    # CPU torch 2.11.0. Anchored so torch-ema and torchmetrics are kept.
    grep -vE "^(torch|triton|pytorch-triton)==|^(nvidia|cuda)-[A-Za-z0-9._-]*==" \
        "$VENV/lock-constraints.raw" > "$VENV/lock-constraints.txt"
    rm -f "$VENV/lock-constraints.raw"
    echo "    $(grep -c . "$VENV/lock-constraints.txt") versions pinned from uv.lock"

    echo
    echo "==> Installing WyFormer with extras: $EXTRAS"
    VIRTUAL_ENV="$VENV" "$UV" pip install \
        -c "$VENV/container-constraints.txt" \
        -c "$VENV/lock-constraints.txt" \
        -e ".[$EXTRAS]"
' _ "$WYFORMER_REPO" "$UV" "$WYFORMER_VENV" "$PLATFORM_DIR" "$EXTRAS"

echo
echo "==> Verifying that torch still comes from the container"
"$PLATFORM_DIR/run.sh" python -c "
import torch, wyckoff_transformer
print('torch              ', torch.__version__)
print('torch imported from', torch.__file__)
print('cuda build         ', torch.version.cuda)
print('cuda available     ', torch.cuda.is_available())
print('wyckoff_transformer', wyckoff_transformer.__file__)
assert torch.__file__.startswith('/usr/local/lib'), \
    'torch was installed into the venv instead of reused from the container'
print()
print('OK: the venv is using the container torch.')
"
