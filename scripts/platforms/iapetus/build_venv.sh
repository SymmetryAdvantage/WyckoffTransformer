#!/usr/bin/env bash
# Build the WyFormer virtual environment for the locally checked-out repo, to be
# run *inside* the iapetus PyTorch image, with the repository bind-mounted at
# /workspace (docs/platforms/iapetus/environment.md):
#
#   docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
#       -v "$PWD:/workspace" -w /workspace pytorch:2.14.0-cuda11.8-py312-universal \
#       bash -lc 'bash scripts/platforms/iapetus/build_venv.sh'
#
# The ASPIRE 2A counterpart is scripts/platforms/aspire2a/build_singularity_venv.sh;
# keep the two in step.
#
# Design
# ------
# * The image ships a custom PyTorch 2.14.0 build for CUDA 11.8 and the host's old
#   GPUs. The venv is created with --system-site-packages so that this Torch (and
#   its bundled CUDA libraries) is inherited rather than a stock wheel stack being
#   pulled in over it.
# * Dependencies are resolved with `uv pip compile` (project deps only) and then
#   installed from a requirements file with torch / nvidia-* / triton stripped out.
#   `uv sync` is avoided on purpose: it writes a universal uv.lock that also has to
#   resolve the genbench-oracle group's `material-hasher` git dependency, which is
#   not wanted here and the container has no working git.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace}
UV=${UV:-$(command -v uv)}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$REPO_DIR/.uv-cache}
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy

cd "$REPO_DIR"
REQ_FULL="$REPO_DIR/.venv-requirements.txt"
REQ_NOTORCH="$REPO_DIR/.venv-requirements.no-torch.txt"

"$UV" --version
python --version
python -c 'import torch; print("base torch:", torch.__version__, torch.version.cuda)'

# 1. Resolve the runtime dependency set (no dev group, no optional extras, no
#    dependency groups -> the material-hasher git source is never referenced).
"$UV" pip compile --python /usr/bin/python3.12 \
    --emit-index-url --no-annotate --no-header \
    -o "$REQ_FULL" pyproject.toml

# 2. Drop Torch and its CUDA companions; the container provides them.
grep -viE '^(torch|nvidia-[a-z0-9-]+|pytorch-triton|triton|triton-[a-z]+)([[:space:]=<>!~;]|$)' \
    "$REQ_FULL" > "$REQ_NOTORCH"

# 3. Fresh venv that can see the container's dist-packages, then install the
#    compiled closure verbatim (--no-deps: the file is already fully pinned, and
#    without it uv would re-add torch as schedulefree's dependency).
"$UV" venv --clear --system-site-packages --python /usr/bin/python3.12 .venv
"$UV" pip install --python .venv/bin/python --no-deps -r "$REQ_NOTORCH"

# 4. The project itself, editable, without touching the dependency set again.
#    (build isolation pulls hatchling + the numpy<2 build pin into a throwaway env.)
"$UV" pip install --python .venv/bin/python --no-deps -e .

./.venv/bin/python - <<'PY'
from importlib.metadata import version
import torch, wyckoff_transformer
from wyckoff_transformer.trainer import train_from_config  # noqa: F401
import wandb, datasets, schedulefree, pyxtal, smact, matminer, omegaconf  # noqa: F401
import pymatgen.core  # noqa: F401
import inspect, sys
from pathlib import Path
from pyxtal.crystal import random_crystal
# The pinned fork from [tool.uv.sources]; stock PyXtal hands check_wp a single
# tolerance. A resolution that silently fell back to PyPI is a wrong-science
# environment, not a cosmetic difference, so fail the build over it.
assert "pair_tol" in inspect.getsource(random_crystal.check_wp), \
    "pyxtal is not the patched fork -- see [tool.uv.sources] in pyproject.toml"
print("torch          :", torch.__version__, torch.version.cuda, "cuda_ok=", torch.cuda.is_available())
print("torch from     :", torch.__file__)
print("wyckoff_transf :", wyckoff_transformer.__file__, version("wyckoff-transformer"))
print("pymatgen       :", version("pymatgen"))
print("pyxtal         :", version("pyxtal"), "(patched fork)")
# Triton's GPU compiler needs compute capability 7.0; the K20c is 3.5
# (docs/platforms/iapetus/environment.md). The requirements filter above already
# strips it, so a copy in the venv means something reinstalled it.
assert not any(Path(sys.prefix).glob("lib/python*/site-packages/triton")), \
    "triton is installed in the venv; it cannot run on iapetus -- uninstall it"
print("import chain OK")
PY

echo "venv ready: $REPO_DIR/.venv"
