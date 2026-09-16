#!/usr/bin/env bash
# NOTE: this script is for building container images for distribution. It is not a
# host's environment setup. A host sets itself up with its own script under
# scripts/platforms/<platform>/, which may call this one and must not rely on more
# than that. Keep host-specific paths, workarounds and checks in those scripts and
# out of this file.
#
# Build the WyFormer virtual environment for the locally checked-out repo, to be
# run *inside* the base PyTorch container:
#
#   singularity run --nv pytorch_2.14.0-cuda12.6-cudnn9-devel.sif \
#       bash scripts/build_singularity_venv.sh
#
# Design
# ------
# * The container ships PyTorch 2.14.0+cu126 in /usr/local/lib/python3.12/dist-packages.
#   The venv is created with --system-site-packages so that CUDA-enabled Torch (and
#   its bundled CUDA libraries) is inherited rather than a second, cu13 wheel stack
#   being pulled in.
# * Dependencies are resolved with `uv pip compile` (project deps only) and then
#   installed from a requirements file with torch / nvidia-* / triton stripped out.
#   `uv sync` is avoided on purpose: it writes a universal uv.lock that also has to
#   resolve the genbench-oracle group's `material-hasher` git dependency, which is
#   not wanted here and the container has no working git.
# * BASE_PYTHON is the image's torch-carrying interpreter. When it is itself a venv
#   (some images keep torch in a venv of their own), --system-site-packages reaches only
#   the interpreter *beneath* that venv, so its site-packages are added by a .pth.
# * UV_CACHE_DIR defaults to the checkout. Point it somewhere shared when several
#   checkouts on one machine each build a venv.
# * VENV_DIR (default .venv, relative to REPO_DIR) is where the venv is built, so a host
#   can build beside a venv jobs are still using and swap it in afterwards.
#   VENV_RELOCATABLE=1 makes it relocatable (relative entry-point shebangs), which is
#   what lets it be renamed into place. VENV_EXTRAS is a space-separated list of
#   optional-dependency groups resolved together with the base set, e.g. "relax".
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/project/11001786/WyFormer/WyckoffTransformer}
UV=${UV:-$HOME/.local/bin/uv}
BASE_PYTHON=${BASE_PYTHON:-/usr/bin/python3.12}
VENV_DIR=${VENV_DIR:-.venv}
VENV_EXTRAS=${VENV_EXTRAS:-}
VENV_RELOCATABLE=${VENV_RELOCATABLE:-0}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$REPO_DIR/.uv-cache}
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy

cd "$REPO_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"
EXTRA_ARGS=()
for extra in $VENV_EXTRAS; do
    EXTRA_ARGS+=(--extra "$extra")
done
VENV_ARGS=(--clear --system-site-packages --python "$BASE_PYTHON")
[ "$VENV_RELOCATABLE" = 1 ] && VENV_ARGS+=(--relocatable)

# The host's agent brief, as CLAUDE.local.md. This script serves more than one
# platform and the container's hostname identifies none, so the caller names it.
if [[ -n "${WYFORMER_PLATFORM:-}" ]]; then
    scripts/platforms/link_agent_brief.sh "$WYFORMER_PLATFORM"
else
    echo "note: WYFORMER_PLATFORM unset; CLAUDE.local.md not linked (see docs/platforms/README.md)" >&2
fi

REQ_FULL="$REPO_DIR/.venv-requirements.txt"
REQ_NOTORCH="$REPO_DIR/.venv-requirements.no-torch.txt"

"$UV" --version
"$BASE_PYTHON" --version
"$BASE_PYTHON" -c 'import torch; print("base torch:", torch.__version__, torch.version.cuda)'

# 1. Resolve the runtime dependency set (no dev group, no dependency groups -> the
#    material-hasher git source is never referenced; optional extras only as asked).
"$UV" pip compile --python "$BASE_PYTHON" \
    --emit-index-url --no-annotate --no-header ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    -o "$REQ_FULL" pyproject.toml

# 2. Drop Torch and its CUDA companions; the container provides them.
grep -viE '^(torch|nvidia-[a-z0-9-]+|pytorch-triton|triton|triton-[a-z]+)([[:space:]=<>!~;]|$)' \
    "$REQ_FULL" > "$REQ_NOTORCH"

# 3. Fresh venv that can see the container's dist-packages, then install the
#    compiled closure verbatim (--no-deps: the file is already fully pinned, and
#    without it uv would re-add torch as schedulefree's dependency).
"$UV" venv "${VENV_ARGS[@]}" "$VENV_DIR"
if "$BASE_PYTHON" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)'; then
    "$BASE_PYTHON" -c 'import site; print(*site.getsitepackages(), sep="\n")' \
        > "$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/container-base.pth"
fi
"$UV" pip install --python "$VENV_PYTHON" --no-deps -r "$REQ_NOTORCH"

# 4. The project itself, editable, without touching the dependency set again.
#    (build isolation pulls hatchling + the numpy<2 build pin into a throwaway env.)
"$UV" pip install --python "$VENV_PYTHON" --no-deps -e .

"$VENV_PYTHON" - <<'PY'
from importlib.metadata import version
import torch, wyckoff_transformer
from wyckoff_transformer.trainer import train_from_config  # noqa: F401
import wandb, datasets, schedulefree, pyxtal, smact, matminer, omegaconf  # noqa: F401
import pymatgen.core  # noqa: F401
import importlib.util, inspect
# Triton comes with the image's torch or not at all -- an image for GPUs below
# Triton's minimum compute capability has none. When present it must import.
triton = importlib.import_module("triton") if importlib.util.find_spec("triton") else None
from pyxtal.crystal import random_crystal
# The pair-tolerance fix, merged upstream in PyXtal 1.1.5 (the version pyproject.toml
# pins); older PyXtal hands check_wp a single tolerance. A resolution that ended up on
# an older release is a wrong-science environment, not a cosmetic difference, so fail
# the build over it.
assert "pair_tol" in inspect.getsource(random_crystal.check_wp), \
    "pyxtal lacks the pair-tolerance fix -- pyproject.toml pins pyxtal ==1.1.5"
print("torch          :", torch.__version__, torch.version.cuda, "cuda_ok=", torch.cuda.is_available())
print("torch from     :", torch.__file__)
print("triton from    :", triton.__file__ if triton else "not in the image")
print("wyckoff_transf :", wyckoff_transformer.__file__, version("wyckoff-transformer"))
print("pymatgen       :", version("pymatgen"))
print("pyxtal         :", version("pyxtal"), "(pair-tolerance fix present)")
print("import chain OK")
PY

echo "venv ready: $REPO_DIR/$VENV_DIR"
