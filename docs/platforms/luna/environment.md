# The luna environment: building, updating, and why it looks like this

WyFormer runs inside an Apptainer container built from an official PyTorch CUDA
image. A uv venv sits on top of that container and **reuses the container's
torch** rather than installing its own.

Read this before changing the image tag, the extras, or how the venv is built.
For day-to-day running see [usage.md](usage.md).

---

## Rebuilding from scratch

```bash
scripts/platforms/luna/build_image.sh   # ~20 min, mostly download
scripts/platforms/luna/build_venv.sh    # ~5 min
```

`build_venv.sh` deletes and recreates the venv, so it is safe to rerun. Nothing
needs to be installed on the host: the PyTorch image ships its own `uv`, and the
build uses that, falling back to a host `uv` (`$UV`, default `~/.local/bin/uv`)
only if a future image tag stops including one.

To rebuild into a scratch location without disturbing the working venv:

```bash
WYFORMER_VENV=$PWD/.venv-test scripts/platforms/luna/build_venv.sh
```

---

## Adding dependencies

**Always pass both constraint files.** A bare `uv pip install` re-resolves torch
and will pull the CUDA 13 build over the container's, breaking the environment
in a way that only shows up as `torch.cuda.is_available() == False` or a driver
error at run time:

```bash
scripts/platforms/luna/run.sh uv pip install \
    -c .venv-luna/container-constraints.txt \
    -c .venv-luna/lock-constraints.txt \
    <package>
```

If a package genuinely needs a newer torch than the container's, the right move
is to rebuild on a newer image, not to let uv install one into the venv.

---

## Updating the container image

Change `WYFORMER_IMAGE_TAG` in `scripts/platforms/luna/config.sh`, point
`WYFORMER_CONTAINER` at a new path, then rerun both build scripts. **Check
`nvidia-smi` first** — the CUDA constraint below is the thing most likely to
bite. After rebuilding, confirm torch still comes from the container with the
health check in [troubleshooting.md](troubleshooting.md).

---

## Why the environment is built this way

These are the non-obvious constraints. Please do not "simplify" them away
without re-checking the underlying facts.

### The image is pinned to CUDA 12.6, not 13.x

PyTorch 2.14.0 is published for CUDA 12.6, 13.0 and 13.2. Luna's driver is
550.163.01, i.e. a **CUDA 12.4** driver. CUDA 12.x minor-version compatibility
lets a 12.6 build run on it, but CUDA 13.x requires a driver >= 580 and fails at
runtime. **Check `nvidia-smi` before bumping the tag.**

The `devel` variant is used rather than `runtime` so nvcc, the CUDA headers and
a compiler are present for dependencies that lack a matching wheel.

### The container is a sandbox directory, not a .sif

Apptainer 1.5.1 bundles mksquashfs 4.7.5, whose orderer crashes while packing an
image this large:

```
FATAL ERROR: Bug in orderer
```

The system mksquashfs is 4.6.1 and would probably succeed, but apptainer takes
its mksquashfs path from `/etc/apptainer/apptainer.conf`, which needs root to
change. Building a sandbox skips squashfs altogether. `apptainer exec` treats a
sandbox directory and a `.sif` identically, so nothing downstream is affected.

To retry a `.sif` (the 4.7.5 bug is concurrency-related, so single-threaded may
get through):

```bash
apptainer build --mksquashfs-args="-processors 1" \
    ~/containers/pytorch-2.14.0-cuda12.6.sif ~/containers/pytorch-2.14.0-cuda12.6
```

Then set `WYFORMER_CONTAINER` to the `.sif`.

### uv does not honour system-site-packages

This is the subtle one, and getting it wrong is silent rather than loud. The
venv is created with `--system-site-packages` so the container's torch is
importable, but that is **not sufficient**:

> `uv pip install` inspects only the target venv's own site-packages. It ignores
> `include-system-site-packages`.

Left alone, uv therefore resolves torch from PyPI and installs a second copy
into the venv. On luna that meant **torch 2.14.0 with CUDA 13 wheels**
(`nvidia-cublas 13.1.1.3`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`), which the
CUDA 12.4 driver cannot run. The venv's site-packages precedes the container's
on `sys.path`, so that broken torch would have won at import time.

`reuse_container_packages.py` prevents this in two ways:

1. It **copies** the `*.dist-info` metadata of the container's CUDA stack into
   the venv's site-packages. That metadata is what uv reads to decide a package
   is already installed.
2. It writes `container-constraints.txt`, pinning those packages to the exact
   versions the container ships, which is passed to `uv pip install -c`. The
   local version segment (`torch==2.14.0+cu126`) means any drift fails loudly
   instead of silently substituting the wrong CUDA build.

Two details are load-bearing and were arrived at the hard way:

**The metadata is copied, not symlinked.** uv enumerates site-packages without
following symlinks, so a symlinked `foo.dist-info` is invisible to it — `uv pip
list` did not show the linked torch at all. uv then tried to install over the
symlink and died with `Read-only file system (os error 30)`. The directories are
metadata only, a few hundred KB.

**Only the CUDA stack is reused**, not everything torch depends on: `torch`,
`triton`, `nvidia-*` and `cuda-*`, 11 distributions in total. The pure-Python
deps torch happens to pull in (`fsspec`, `jinja2`, `networkx`, `sympy`,
`filelock`, ...) are deliberately left to uv. Pinning those as well made the
resolution unsatisfiable — the container's `fsspec 2026.7.0` is newer than the
`<=2026.6.0` that `datasets` permits. uv installs its own copies into the venv,
which shadow the container's, and that is fine: they are architecture-neutral,
so there is nothing to mismatch.

### Versions come from uv.lock, but the CUDA stack does not

`uv pip install -e ".[...]"` resolves from scratch and takes the newest of
everything, which quietly degrades the environment:

- pandas 3.0.5 is selected;
- matminer 0.10.1 requires `pandas<3`, so the resolver backtracks matminer to
  **0.8.0**;
- matminer 0.8.0 does `from scipy.special import sph_harm`, removed in scipy
  1.17;
- the test suite dies at collection with
  `ImportError: cannot import name 'sph_harm' from 'scipy.special'`.

Nothing here is container-specific — it is the ordinary reason the project ships
a `uv.lock`. So `build_venv.sh` derives constraints from the lock with

```bash
uv export --frozen --no-hashes --no-header --no-annotate --no-emit-project \
    --extra dev --extra relax --extra research
```

and filters out the CUDA stack before passing the result to
`uv pip install -c`. The filter is anchored (`^torch==`, not `^torch`) so that
`torch-ema` and `torchmetrics`, which are ordinary MACE dependencies, are kept.

`uv sync` is **not** used, even though it is the project's normal workflow and
what CI runs. `uv sync` manages the venv strictly: it would discard the reused
container metadata and install the lock's CPU **torch 2.11.0**, which is exactly
what this setup exists to avoid. The lock supplies versions; the container
supplies torch.

A consequence worth knowing: because the lock pins the CPU torch, the two
constraint files must not overlap. `container-constraints.txt` owns `torch`,
`triton` and `nvidia-*`/`cuda-*`; `lock-constraints.txt` owns everything else.
Both are regenerated by `build_venv.sh` and live in the venv.

### The venv only works inside the container

`.venv-luna/pyvenv.cfg` has `home = /usr/bin`, pointing at the container's
python. Running `.venv-luna/bin/python` from the host will not work. Always go
through `run.sh` or `apptainer exec`.
