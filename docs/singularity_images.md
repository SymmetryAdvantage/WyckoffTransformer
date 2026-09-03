# Building and publishing WyFormer Singularity images

This guide builds GPU-enabled WyFormer SIF images for the stable release and
development branches, then publishes them to GitHub Container Registry (GHCR)
using the ORAS transport.

The images are based on the official CUDA PyTorch runtime image:

```text
pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime
```

That image provides CUDA 13.2 and PyTorch 2.14.0. The build deliberately uses
that provided PyTorch installation rather than downloading a second PyTorch
wheel into the WyFormer virtual environment.

## Published tags

GHCR repository names are lowercase. The following tag policy keeps stable and
development images unambiguous:

| Source ref | GHCR tag | Purpose |
| --- | --- | --- |
| `v1.0.7` | `v1.0.7` | Immutable stable release image |
| `v1.0.7` | `latest` | Most recent stable release image |
| `main` | `main` | Current default-branch image |
| `training-loss-fixes` | `training-loss-fixes` | Current feature-branch image |

`latest` must only be moved when publishing a new stable Git tag. Branch tags
are intentionally mutable and should be rebuilt when their branch tip changes.

The full image references are:

```text
oras://ghcr.io/symmetryadvantage/wyckofftransformer:v1.0.7
oras://ghcr.io/symmetryadvantage/wyckofftransformer:latest
oras://ghcr.io/symmetryadvantage/wyckofftransformer:main
oras://ghcr.io/symmetryadvantage/wyckofftransformer:training-loss-fixes
```

## Prerequisites

- SingularityCE 4.3 or newer, with a working fakeroot mapping.
- `git`, `curl`, and a POSIX shell on the build host.
- Sufficient local storage. Each final image is currently about 4.5 GiB, and
  the base-image and build caches require additional temporary space.
- A GHCR token with `write:packages` and access to the
  `SymmetryAdvantage` organization.

Authenticate Singularity to GHCR before publishing. Keep the token out of
shell history and source control:

```bash
read -rsp "GHCR token: " GHCR_TOKEN
printf '\n'
singularity registry login -u kazeevn -p "$GHCR_TOKEN" oras://ghcr.io
unset GHCR_TOKEN
```

Confirm the stored credential without exposing its token:

```bash
singularity registry list
```

Expected output includes `oras://ghcr.io`.

## Create an isolated build workspace

Do not build from a dirty development checkout. Clone the source into a
temporary directory and archive only the runtime source files into each build
stage; this excludes datasets, checkpoints, local Git state, and uncommitted
changes.

```bash
workdir=$(mktemp -d)
git clone https://github.com/SymmetryAdvantage/WyckoffTransformer.git \
  "$workdir/source-repository"
cd "$workdir/source-repository"
git fetch --tags --force
```

Install a pinned standalone `uv` binary into the build workspace. This avoids
mutating the base image's externally managed Python installation.

```bash
mkdir -p "$workdir/tools"
curl -LsSf https://astral.sh/uv/0.12.6/install.sh |
  env UV_UNMANAGED_INSTALL="$workdir/tools" sh
"$workdir/tools/uv" --version
```

## Definition file

Create `$workdir/WyFormer.def` with the following contents:

```def
Bootstrap: docker
From: pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

%labels
    org.opencontainers.image.title WyFormer
    org.opencontainers.image.description Wyckoff Transformer with CUDA-enabled PyTorch
    org.opencontainers.image.source https://github.com/SymmetryAdvantage/WyckoffTransformer
    org.opencontainers.image.revision {{ SOURCE_REVISION }}
    org.opencontainers.image.version {{ IMAGE_VERSION }}
    org.opencontainers.image.base.name pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

%files
    ./source /opt/wyformer
    ./uv /usr/local/bin/uv

%post
    /usr/local/bin/uv --version

    cd /opt/wyformer
    sed -i -E 's/"torch[[:space:]]*(==|>=)[^"]*"/"torch ==2.14.0"/' pyproject.toml
    sed -i \
        -e '/^cdvae = \["cdvae-property-models"\]$/d' \
        -e '/^cdvae-property-models = { path = "cdvae_property_models" }$/d' \
        pyproject.toml
    /usr/local/bin/uv venv --system-site-packages .venv
    UV_CACHE_DIR=/opt/wyformer/.uv-cache \
        /usr/local/bin/uv sync --no-dev --no-install-package torch

    rm -rf /opt/wyformer/.uv-cache

%environment
    export VIRTUAL_ENV=/opt/wyformer/.venv
    export PATH="/opt/wyformer/.venv/bin:${PATH}"

%test
    test -x /opt/wyformer/.venv/bin/python
    /opt/wyformer/.venv/bin/python - <<'PY'
import pathlib

import torch
from wyckoff_transformer.cli.generate import main

torch_path = pathlib.Path(torch.__file__).resolve()
assert torch.__version__.startswith("2.14.0"), torch.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
assert not torch_path.is_relative_to("/opt/wyformer/.venv"), torch_path
PY

%runscript
    if [ "$#" -eq 0 ]; then
        exec /opt/wyformer/.venv/bin/python
    fi
    exec "$@"
```

### Why the metadata is patched

The source revisions were locked around PyTorch 2.11.x, and development
branches require it explicitly. The CUDA base provides PyTorch 2.14.0+cu132
instead. The `sed` command changes the copied `pyproject.toml` inside the
image to require 2.14.0, while `--no-install-package torch` prevents `uv`
from installing another Torch distribution into `.venv`.
`--system-site-packages` makes the base image's CUDA-enabled Torch visible to
the virtual environment.

The optional `cdvae` extra and its local source entry are also removed from the
copied metadata. That optional package hard-pins Torch 2.11.0, which makes
uv's universal dependency resolution incompatible with the CUDA 13.2 base.
This does not affect the default WyFormer installation. The image does not
include the optional CDVAE package or its model checkpoints.

## Build each revision

Run the following helper from the cloned repository directory. It creates a
fresh source archive for each ref, labels the SIF with its immutable commit
SHA, and runs the definition file's Python/Torch check.

```bash
build_root="$workdir/build"
mkdir -p "$build_root/cache" "$build_root/tmp"

build_image() {
  local source_ref=$1
  local image_tag=$2
  local stage="$build_root/$image_tag"
  local revision

  mkdir -p "$stage/source"
  git archive --format=tar "$source_ref" -- \
    pyproject.toml uv.lock README.md LICENSE hatch_build.py src scripts yamls |
    tar -x -C "$stage/source"
  cp "$workdir/tools/uv" "$stage/uv"
  cp "$workdir/WyFormer.def" "$stage/WyFormer.def"

  revision=$(git rev-parse "$source_ref")
  (
    cd "$stage"
    SINGULARITY_CACHEDIR="$build_root/cache" \
    SINGULARITY_TMPDIR="$build_root/tmp" \
      singularity build --fakeroot --force \
        --build-arg "SOURCE_REVISION=$revision" \
        --build-arg "IMAGE_VERSION=$image_tag" \
        "WyFormer-$image_tag.sif" WyFormer.def

    singularity test "WyFormer-$image_tag.sif"
    singularity exec --nv "WyFormer-$image_tag.sif" python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available through Singularity --nv"
print(f"torch={torch.__version__} cuda={torch.version.cuda} "
      f"gpu={torch.cuda.get_device_name(0)}")
PY
  )
}

build_image v1.0.7 v1.0.7
build_image origin/main main
build_image origin/training-loss-fixes training-loss-fixes
```

The `--nv` check requires a GPU build host with a sufficiently recent NVIDIA
driver. The non-GPU `singularity test` still confirms the application import,
the virtual environment, the inherited Torch location, and the expected CUDA
build.

## Publish to GHCR

After all local builds succeed, publish the stable image under both its
immutable release tag and the stable `latest` alias:

```bash
registry=oras://ghcr.io/symmetryadvantage/wyckofftransformer

singularity push \
  "$build_root/v1.0.7/WyFormer-v1.0.7.sif" \
  "$registry:v1.0.7"
singularity push \
  "$build_root/v1.0.7/WyFormer-v1.0.7.sif" \
  "$registry:latest"
```

Then publish the development images:

```bash
singularity push \
  "$build_root/main/WyFormer-main.sif" \
  "$registry:main"
singularity push \
  "$build_root/training-loss-fixes/WyFormer-training-loss-fixes.sif" \
  "$registry:training-loss-fixes"
```

Direct ORAS pushes create an organization-scoped GHCR package. If it does not
immediately appear on the repository's Packages tab, open the organization
package list at
`https://github.com/orgs/SymmetryAdvantage/packages`, select
`wyckofftransformer`, and associate it with the
`SymmetryAdvantage/WyckoffTransformer` repository.

## Use an image on an HPC cluster

Pull the required image once:

```bash
singularity pull \
  oras://ghcr.io/symmetryadvantage/wyckofftransformer:latest
```

Run a Python command with the host NVIDIA driver exposed:

```bash
singularity exec --nv wyckofftransformer_latest.sif \
  python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

For command-line entry points, use `singularity exec --nv` with
`wyformer-generate`, `wyformer-cryspr`, or another command available from the
image's virtual environment.
