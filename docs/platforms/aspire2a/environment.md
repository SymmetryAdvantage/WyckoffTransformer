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
Batch scripts should not rely on the module system being initialised, so they
source it first when `module` is undefined, as
`scripts/platforms/aspire2a/run_in_singularity.sh` does:

```bash
if ! command -v singularity >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load singularity
fi
```

Do not put `/app/apps/singularity/...` on `PATH` by hand instead; the module is the
interface, and the paths behind it change with the installed version.

---

## Building the venv

```bash
module load singularity
cd /home/project/11001786/WyFormer/WyckoffTransformer
singularity run --nv --bind /home/project ~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif \
    env WYFORMER_PLATFORM=aspire2a bash scripts/build_singularity_venv.sh
```

**Not while any job runs:** the main checkout's `.venv` is a symlink to the one venv
every checkout and chain shares, and step 3 clears it
([below](#where-the-one-venv-actually-is)). Unlock it first with `store_lock.sh unlock`.

`--bind /home/project` is required because Singularity does not bind project mounts
automatically on ASPIRE 2A. `WYFORMER_PLATFORM=aspire2a` also links `CLAUDE.local.md` to
[agent_brief.md](agent_brief.md); see [../README.md](../README.md). The venv
build is the expensive way to get it -- `bash scripts/platforms/aspire2a/env_init.sh`
(or `bash scripts/platforms/link_agent_brief.sh aspire2a`) from a login node does only
that, and needs nothing but bash.

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
the `genbench-oracle` group's `material-hasher` git dependency. The container
has no git and no libcurl, so that resolution fails, and the pin is not wanted
here anyway. There is no `uv.lock` in this checkout, which is why the ASPIRE 2A
dependency set is a **fresh resolution** rather than a pinned one — see the
version-skew section below.

The **PyXtal pin is a tarball, not a git branch, because of this host.** The
fork in `[tool.uv.sources]` is referenced as
`https://github.com/kazeevn/PyXtal/archive/<sha>.tar.gz`: a `{ git = ... }`
source makes uv shell out to `git`, which this container does not have, and the
compile in step 1 would fail outright on a *base* dependency. A tarball needs
only HTTPS. Step 4's verification asserts the patched `check_wp` is what ended
up in the venv, so a silent fallback to PyPI PyXtal fails the build.

To move that pin into the existing venv without a rebuild — the venv is shared
by every running job, so see the live-risk warning below — compile the direct
requirements and install the one line:

```bash
bash scripts/platforms/aspire2a/run_in_singularity.sh bash -c '
  export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
  ~/.local/bin/uv pip compile --no-deps --no-annotate --no-header pyproject.toml \
      | grep "^pyxtal " > .venv-requirements.pyxtal.txt
  ~/.local/bin/uv pip install --python .venv/bin/python --no-deps \
      -r .venv-requirements.pyxtal.txt'
```

`uv` itself is a standalone binary at `~/.local/bin/uv` (0.12.6). The build sets
`UV_CACHE_DIR=$REPO/.uv-cache` (3 GB — keeping it off the 50 GB home quota),
`UV_PYTHON_DOWNLOADS=never` and `UV_LINK_MODE=copy`.

### Adding or repairing the project install

Re-running only step 4 is safe and cheap, and is what you want after adding a
console script to `[project.scripts]` or changing package data:

```bash
bash scripts/platforms/aspire2a/run_in_singularity.sh \
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
bash scripts/platforms/aspire2a/run_in_singularity.sh bash -c '
  export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
  ~/.local/bin/uv pip compile --python /usr/bin/python3.12 --emit-index-url \
      --no-annotate --no-header --extra relax -o .venv-requirements.relax.txt pyproject.toml
  grep -viE "^(torch|nvidia-[a-z0-9-]+|pytorch-triton|triton|triton-[a-z]+)([[:space:]=<>!~;]|\$)" \
      .venv-requirements.relax.txt > .venv-requirements.relax.no-torch.txt
  ~/.local/bin/uv pip install --python .venv/bin/python --no-deps \
      -r .venv-requirements.relax.no-torch.txt'
```

~30 min on a login node. It is currently installed (orb-models 0.7.0,
mace-torch 0.3.16) and it also pulled in pytest 9.1.1, torchmetrics, warp-lang
and e3nn. `scripts/platforms/aspire2a/protocol_relax.pbs` used to run this itself
when `orb_models` was missing; since the venv is read-only it stops with a pointer
here instead.

**The venv is read-only; installing into it is a deliberate step.** Unlock it, install,
lock it again:

```bash
bash scripts/platforms/aspire2a/store_lock.sh unlock /scratch/users/nus/kna/WyckoffTransformer/.venv
# ... the install ...
bash scripts/platforms/aspire2a/store_lock.sh lock /scratch/users/nus/kna/WyckoffTransformer/.venv
```

**Installing into `.venv` while jobs are running is a live risk.** The venv is
shared by every chained job on every node and every worktree; four were running during
the last audit. The relax extra's only deltas against the training set were patch-level
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
`scripts/platforms/aspire2a/run_in_singularity.sh`.

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
whatever is newest on PyPI at build time. As of 2026-09-08 that had broken
`matminer` 0.8.0 (unmaintained since 2023) against both `scipy` 1.18.1 and
`pymatgen` 2026.5.4:

| Import | Fails with |
| --- | --- |
| `matminer.featurizers.site.bonding` | `cannot import name 'sph_harm' from 'scipy.special'` |
| `matminer.utils.data` | `cannot import name '_pt_data' from 'pymatgen.core.periodic_table'` |

`pyproject.toml` now floors `matminer` at 0.10.1, which fixes both imports and,
through its own `pandas<3`, holds pandas at 2.3.3. Fresh resolves get the
working set; a `.venv` built before 2026-09-09 does not. See
[troubleshooting.md](troubleshooting.md#matminer-is-broken-against-the-resolved-scipy-and-pymatgen).

---

## Git worktrees: reusing the shared virtual environment

On other platforms (such as zeus and iapetus), each git worktree creates its own virtual
environment. On ASPIRE 2A every checkout shares **one** venv instead: ASPIRE 2A's
filesystems are slow on metadata (thousands of small files), so building one per
worktree is too slow. The workflow -- how to create a worktree, why not with
`claude --worktree`, what the launcher enforces -- is in
[usage.md](usage.md#working-in-a-git-worktree).

### Where the one venv actually is

Checked 2026-09-16:

| Path | What |
| --- | --- |
| `/scratch/users/nus/kna/WyckoffTransformer/.venv` | the venv itself (1.5 GB), built in the old scratch checkout |
| `/home/project/11001786/WyFormer/WyckoffTransformer/.venv` | a symlink to it |
| `<worktree>/.venv` | a symlink to the main checkout's, made by `env_init.sh` |

The old scratch checkout, `/scratch/users/nus/kna/WyckoffTransformer`, is still in use:
the chains submitted before the main checkout moved to `/home/project` run from it and
from this venv. Do not remove either while any of them is queued.

The venv is read-only ([usage.md](usage.md#the-shared-cache-and-venv-are-read-only)).
Rebuilding it in place (`uv venv --clear` through the symlink) would pull it from under
every running job; build a new one elsewhere and repoint the symlink between chains.

### How a worktree gets its own code from the shared venv
- The editable install's `.pth` names the checkout it was installed from --
  `/scratch/users/nus/kna/WyckoffTransformer/src`, not the main checkout's.
- `scripts/platforms/aspire2a/run_in_singularity.sh` therefore puts the invoking checkout's
  `src` first on `PYTHONPATH` (`PYTHONPATH=$REPO_DIR/src:$PYTHONPATH`), which Python searches
  before any `.pth` entry. Anything run outside that launcher imports the old checkout's code.
- `run_in_singularity.sh` also binds `/home/project` and `/data/projects` into Singularity
  (not mounted by default on ASPIRE 2A), so the main checkout, the data store and the cache
  are visible inside the container.

### Creating and initialising a worktree
```bash
bash scripts/platforms/aspire2a/create_worktree.sh <name> [start-point]
```
creates `/home/users/nus/kna/scratch/WyFormer/worktrees/<name>` on branch `<name>` and runs
`env_init.sh` in it, which links `CLAUDE.local.md` to `docs/platforms/aspire2a/agent_brief.md`,
links `.venv` to the shared venv, and validates `paths.env`. For a worktree made some other
way, run `bash scripts/platforms/aspire2a/env_init.sh` from inside it.

---

## Caches and data

Storage paths are defined in `~/.config/wyformer/paths.env` (see `docs/data_store.md`):

```bash
WYFORMER_DATA=/home/project/11001786/WyFormer/data
WYFORMER_CACHE=/home/project/11001786/WyFormer/cache
WYFORMER_RUNS=/scratch/users/nus/kna/WyFormer/runs
WANDB_DIR=/scratch/users/nus/kna/WyFormer
```

| Path | Size / Type | Note |
| --- | --- | --- |
| `/home/project/11001786/WyFormer/data` | GPFS project | Untracked raw datasets (`WYFORMER_DATA`), e.g. `lemat-bulk` |
| `/home/project/11001786/WyFormer/cache` | GPFS project | Tokenised dataset caches (`WYFORMER_CACHE`), e.g. `lemat_bulk_fmax1_stress`; **read-only** |
| `/scratch/users/nus/kna/WyFormer/runs` | Lustre scratch | Training checkpoints and run artifacts (`WYFORMER_RUNS`) |
| `/scratch/users/nus/kna/WyFormer` | Lustre scratch | Local W&B output directory (`WANDB_DIR`) |
| `/scratch/users/nus/kna/WyFormer/logs` | Lustre scratch | PBS output of every launcher |
| `/home/users/nus/kna/scratch/WyFormer/worktrees/` | Lustre scratch | Working git worktrees |
| `/scratch/users/nus/kna/WyckoffTransformer/.venv` | 1.5 GB, Lustre | Shared container-only virtual environment, **read-only**; the main checkout's `.venv` links to it |
| `<repo>/.uv-cache` | 3.0 GB | uv cache, kept on scratch by design |
| `~/.cache/huggingface` | 5.8 GB | on the **home** quota |
| `~/.cache/cached_path` | ORB checkpoints | on the **home** quota — `cached_path` ignores `XDG_CACHE_HOME` |

`~/.bash_profile` sets `XDG_CACHE_HOME=$HOME/scratch/.cache`, `SINGULARITY_CACHEDIR` and
`SINGULARITY_BIND=/scratch`, which keeps most temporary caches on Lustre. `cached_path` is the notable
exception. Home is 34.6 GB of a 50 GB quota, so check `myquota` before pulling another multi-GB checkpoint.

`.venv`, `.uv-cache` and all four `.venv-requirements*.txt` files are gitignored. Keep the generated
requirements files on disk: they allow reinstalling after a `.venv` wipe without repeating the
network-bound `uv pip compile`.
