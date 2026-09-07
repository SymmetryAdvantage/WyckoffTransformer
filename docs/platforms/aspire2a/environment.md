# The ASPIRE 2A environment

## Why there is a container

The compute nodes are RHEL 8.10 with no `python`, no `git`, no `uv` and no
`singularity` on the default `PATH`, and a glibc too old for current wheels.
So WyFormer runs inside a stock PyTorch image, and the repo's `.venv` is built
**against that image**:

```
~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif       12 GB, on the home quota
  docker://pytorch/pytorch:2.14.0-cuda12.6-cudnn9-devel   (Ubuntu 24.04)
  built 2026-09-04 with singularity 3.10.0
  ships python 3.12.3 + torch 2.14.0+cu126 + triton + CUDA
      in /usr/local/lib/python3.12/dist-packages
```

`.venv` is created with `--system-site-packages`, so torch, triton and the
bundled `nvidia-*` libraries are **inherited from the image** rather than being
a second, conflicting wheel stack. That is the whole design, and its consequence
is that `.venv` is meaningless outside the container: `.venv/bin/python` is a
symlink to an interpreter that does not exist on the host.

The node driver is 570.124.06 (CUDA 12.8), which runs the cu126 build fine.

### Getting singularity

`module load singularity` gives 3.10.0; `singularity/4.3.1` is also available.
Batch scripts should not rely on the module system being initialised, so they do
what `scripts/train_in_pb.sh` does:

```bash
if ! command -v singularity >/dev/null 2>&1; then
    export PATH="/app/apps/singularity/sup/squashfuse/0.6.1/bin:/app/apps/singularity/3.10.0/bin:$PATH"
fi
command -v singularity >/dev/null 2>&1 || { source /etc/profile.d/modules.sh && module load singularity; }
```

---

## Building the venv

```bash
module load singularity
cd /scratch/users/nus/kna/WyckoffTransformer
singularity run --nv ~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif \
    bash scripts/build_singularity_venv.sh
```

Four steps, all inside the container:

1. `uv pip compile pyproject.toml` — **project dependencies only**, no extras, no
   dependency groups -> `.venv-requirements.txt`.
2. `grep -v` out `torch`, `nvidia-*`, `triton`, `pytorch-triton` ->
   `.venv-requirements.no-torch.txt`. The container provides them.
3. `uv venv --clear --system-site-packages` then
   `uv pip install --no-deps -r .venv-requirements.no-torch.txt`. `--no-deps` is
   load-bearing: the file is already a full pinned closure, and without it uv
   re-adds torch as `schedulefree`'s dependency.
4. `uv pip install --no-deps -e .` — the project itself.

Cold build ~20 min (bandwidth-bound, so **do it on a login node**); warm ~3 min.

**`uv sync` is deliberately not used.** Its universal lock also has to resolve
the `genbench-oracle` group's `material-hasher` git dependency and the `cdvae`
extra's `torch==2.11` pin. The container has no git and no libcurl, so that
resolution fails; and neither pin is wanted here. There is no `uv.lock` in this
checkout, which is why the ASPIRE 2A dependency set is a **fresh resolution**
rather than a pinned one — see the version-skew section below.

`uv` itself is a standalone binary at `~/.local/bin/uv` (0.12.6). The build sets
`UV_CACHE_DIR=$REPO/.uv-cache` (3 GB — keeping it off the 50 GB home quota),
`UV_PYTHON_DOWNLOADS=never` and `UV_LINK_MODE=copy`.

### Adding or repairing the project install

Re-running only step 4 is safe and cheap, and is what you want after adding a
console script to `[project.scripts]` or changing package data:

```bash
bash scripts/run_in_singularity.sh \
    ~/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
```

(As of 2026-09-08 the venv is behind on this: only `wyformer-generate`,
`wyformer-cryspr` and `wyformer-protocol` have entry points. The other six run
as `python -m wyckoff_transformer.cli.<module>`.)

### The `relax` extra: ORB and MACE

The base build has **no** `orb-models` or `mace-torch`, so
`wyformer-protocol --stage relax` and anything touching the hull MLIPs fails
with `ModuleNotFoundError: No module named 'orb_models'`. Add it with:

```bash
bash scripts/run_in_singularity.sh bash -c '
  export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
  ~/.local/bin/uv pip compile --python /usr/bin/python3.12 --emit-index-url \
      --no-annotate --no-header --extra relax -o .venv-requirements.relax.txt pyproject.toml
  grep -viE "^(torch|nvidia-[a-z0-9-]+|pytorch-triton|triton|triton-[a-z]+)([[:space:]=<>!~;]|\$)" \
      .venv-requirements.relax.txt > .venv-requirements.relax.no-torch.txt
  ~/.local/bin/uv pip install --python .venv/bin/python --no-deps \
      -r .venv-requirements.relax.no-torch.txt'
```

~30 min on a login node. `scripts/protocol_relax.pbs` does this automatically
when `orb_models` is missing. It is currently installed (orb-models 0.7.0,
mace-torch 0.3.16) and it also pulled in pytest 9.1.1, torchmetrics, warp-lang
and e3nn.

**Installing into `.venv` while jobs are running is a live risk.** The venv is
shared by every chained job on every node; four were running during the last
audit. The relax extra's only deltas against the training set were patch-level
(numpy 2.5.2 -> 2.5.3, rich 15 -> 13.9.4), which is why it was survivable. Do not
assume the next one will be.

---

## Never run `uv` from the host

```
# WRONG -- from a host shell
uv run pytest
uv sync
```

The host has no `python` and no `git` on `PATH`, and `.venv/bin/python` points
at the container's interpreter. uv reports

> Ignoring existing virtual environment linked to non-existent Python interpreter

**deletes `.venv`**, and rebuilds an empty one. Everything goes through
`scripts/run_in_singularity.sh`.

To recover from that wipe, `.venv-requirements.no-torch.txt` is still there, so
steps 3-4 alone are enough — inside the container:

```bash
~/.local/bin/uv venv --clear --system-site-packages --python /usr/bin/python3.12 .venv
~/.local/bin/uv pip install --python .venv/bin/python --no-deps -r .venv-requirements.no-torch.txt
~/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
```

The host *does* have a working git via `module load git/2.39.2`, which is what
you want for anything touching the repo itself.

---

## Dependency version skew

Because the build resolves fresh instead of using a lockfile, it picks up
whatever is newest on PyPI at build time. As of 2026-09-08 that has broken
`matminer` 0.8.0 (unmaintained since 2023) against both `scipy` 1.18.1 and
`pymatgen` 2026.5.4:

| Import | Fails with |
| --- | --- |
| `matminer.featurizers.site.bonding` | `cannot import name 'sph_harm' from 'scipy.special'` |
| `matminer.utils.data` | `cannot import name '_pt_data' from 'pymatgen.core.periodic_table'` |

Neither `scipy` nor `matminer` is pinned in `pyproject.toml`. Training,
generation and relaxation are unaffected; the CDVAE evaluation metrics and the
`formula_energy` Magpie baselines are not. See
[troubleshooting.md](troubleshooting.md#matminer-is-broken-against-the-resolved-scipy-and-pymatgen).

---

## Caches and data

| Path | Size | Note |
| --- | --- | --- |
| `<repo>/cache/` | 23 GB | dataset caches; `lemat_bulk_ehull` 11 G, `lemat_bulk_fmax1` 5.7 G, `alex_mp_20` 3.5 G, `mp_20` 911 M |
| `<repo>/.uv-cache` | 3.0 GB | uv, on scratch by design |
| `<repo>/.venv` | 1.5 GB | container-only |
| `~/.cache/huggingface` | 5.8 GB | on the **home** quota |
| `~/.cache/cached_path` | ORB checkpoints | on the **home** quota — `cached_path` ignores `XDG_CACHE_HOME` |

`~/.bash_profile` sets `XDG_CACHE_HOME=$HOME/scratch/.cache`,
`SINGULARITY_CACHEDIR` and `SINGULARITY_BIND=/scratch`, which keeps most caches
on Lustre. `cached_path` is the notable exception. Home is 34.6 GB of a 50 GB
quota, so check `myquota` before pulling another multi-GB checkpoint.

`.venv`, `.uv-cache` and all four `.venv-requirements*.txt` files are
gitignored. The requirements files are generated — the base pair by
`build_singularity_venv.sh`, the `relax` pair by `protocol_relax.pbs` — and are
never authoritative: each is whatever PyPI resolved at the moment that script
ran. Keep them on disk anyway, since they let you reinstall after a `.venv` wipe
without repeating the network-bound `uv pip compile`.
