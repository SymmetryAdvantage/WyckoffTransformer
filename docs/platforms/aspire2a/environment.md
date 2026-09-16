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

From a login node (it downloads), in any checkout:

```bash
bash scripts/platforms/aspire2a/build_venv.sh              # build, verify, swap in, lock
bash scripts/platforms/aspire2a/build_venv.sh --no-swap    # build and verify only
```

It always builds the main checkout's `.venv`, at
`/home/project/11001786/WyFormer/WyckoffTransformer/.venv` -- project storage, out of
reach of scratch's purge policy -- and it is safe while jobs run: the new venv is built
beside the live one as `.venv.build-<timestamp>`, and only after its import check passes
is the old `.venv` renamed to `.venv.previous-<timestamp>` and the new one renamed into
place, then locked read-only. The venv is relocatable (`uv venv --relocatable`), which is
what makes the rename safe. Jobs that start afterwards use the new venv -- including the
next link of a chain that started on the old one; a process already running keeps what
it has imported but imports anything new from the new venv. Delete a `.venv.previous-*`
once nothing runs on it (`store_lock.sh unlock` it first).

Inside the container it runs the generic `scripts/build_singularity_venv.sh` with
`VENV_DIR=.venv.build-<timestamp> VENV_RELOCATABLE=1 VENV_EXTRAS=relax`, which does four
steps:

1. `uv pip compile pyproject.toml --extra relax` -- project dependencies plus the `relax`
   extra, no dependency groups -> `.venv-requirements.txt`.
2. `grep -v` out `torch`, `nvidia-*`, `triton`, `pytorch-triton` ->
   `.venv-requirements.no-torch.txt`. The container provides them.
3. `uv venv --clear --system-site-packages --relocatable` then
   `uv pip install --no-deps -r .venv-requirements.no-torch.txt`. `--no-deps` is
   load-bearing: the file is already a full pinned closure, and without it uv
   re-adds torch as `schedulefree`'s dependency.
4. `uv pip install --no-deps -e .` -- the project itself, editable from the main checkout.

then imports the chain training needs and asserts that PyXtal has the pair-tolerance fix
(merged upstream in 1.1.5, which `pyproject.toml` pins), so a resolution that fell back
to an older PyXtal fails the build instead of producing wrong structures.

`--bind /home/project,/data/projects` is required because Singularity does not bind
project mounts automatically on ASPIRE 2A. `WYFORMER_PLATFORM=aspire2a` also links
`CLAUDE.local.md` to [agent_brief.md](agent_brief.md); `env_init.sh` does only that.

**`uv sync` is deliberately not used.** Its universal lock also has to resolve
the `genbench-oracle` group's `material-hasher` git dependency. The container
has no git and no libcurl, so that resolution fails, and the pin is not wanted
here anyway. There is no `uv.lock` in this checkout, which is why the ASPIRE 2A
dependency set is a **fresh resolution** rather than a pinned one — see the
version-skew section below.

`uv` itself is a standalone binary at `~/.local/bin/uv` (0.12.6). The build sets
`UV_CACHE_DIR=/scratch/users/nus/kna/WyFormer/uv-cache` (regenerable, so on scratch and
off the 50 GB home quota), `UV_PYTHON_DOWNLOADS=never` and `UV_LINK_MODE=copy`.

### Adding or repairing the project install

A full rebuild with `build_venv.sh` is the default way to change the venv: it never
touches the live one. After adding a console script to `[project.scripts]` or changing
package data, re-running only step 4 in place is cheap, but it installs into the live
venv, so it needs the unlock/lock below and carries the live risk:

```bash
bash scripts/platforms/aspire2a/store_lock.sh unlock /home/project/11001786/WyFormer/WyckoffTransformer/.venv
bash scripts/platforms/aspire2a/run_in_singularity.sh \
    ~/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
bash scripts/platforms/aspire2a/store_lock.sh lock /home/project/11001786/WyFormer/WyckoffTransformer/.venv
```

### The `relax` extra: ORB and MACE

`build_venv.sh` installs the `relax` extra (`orb-models`, `mace-torch`), which
`wyformer-protocol --stage relax` and anything touching the hull MLIPs needs; without
it they fail with `ModuleNotFoundError: No module named 'orb_models'`.
`scripts/platforms/aspire2a/protocol_relax.pbs` checks for it and stops with a pointer
here rather than installing into the shared venv.

**The venv is read-only; installing into it in place is a deliberate step** -- unlock,
install, lock, as above -- and **a live risk**: the venv is shared by every chained job on
every node and every worktree. When the relax extra was once installed into a live venv,
its only deltas against the training set were patch-level (numpy 2.5.2 -> 2.5.3,
rich 15 -> 13.9.4), which is why it was survivable. Prefer a rebuild.

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

| Path | What |
| --- | --- |
| `/home/project/11001786/WyFormer/WyckoffTransformer/.venv` | the venv itself, a real directory on project storage, built by `build_venv.sh` |
| `<worktree>/.venv` | a symlink to it, made by `env_init.sh` |
| `/scratch/users/nus/kna/WyckoffTransformer/.venv` | the previous venv (read-only), built 2026-09-05 in the old scratch checkout; stale against `pyproject.toml` (stock PyXtal 1.1.4, matminer 0.8.0) |

Until 2026-09-16 the main checkout's `.venv` was a symlink to that scratch venv, which
scratch's purge policy could have deleted from under every job. The old scratch checkout,
`/scratch/users/nus/kna/WyckoffTransformer`, is still in use: the chains submitted before
the main checkout moved to `/home/project` run from it and from its venv. Remove both
once none of them is queued.

### How a worktree gets its own code from the shared venv
- The editable install's `.pth` names the checkout it was installed from -- the main
  checkout's `src`.
- `scripts/platforms/aspire2a/run_in_singularity.sh` therefore puts the invoking checkout's
  `src` first on `PYTHONPATH` (`PYTHONPATH=$REPO_DIR/src:$PYTHONPATH`), which Python searches
  before any `.pth` entry. Anything run outside that launcher imports the main checkout's code.
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
| `/home/project/11001786/WyFormer/WyckoffTransformer/.venv` | GPFS project | Shared container-only virtual environment, **read-only** |
| `/scratch/users/nus/kna/WyFormer/uv-cache` | Lustre scratch | uv cache, regenerable, so on scratch by design |
| `~/.cache/huggingface` | 5.8 GB | on the **home** quota |
| `~/.cache/cached_path` | ORB checkpoints | on the **home** quota — `cached_path` ignores `XDG_CACHE_HOME` |

`~/.bash_profile` sets `XDG_CACHE_HOME=$HOME/scratch/.cache`, `SINGULARITY_CACHEDIR` and
`SINGULARITY_BIND=/scratch`, which keeps most temporary caches on Lustre. `cached_path` is the notable
exception. Home is 34.6 GB of a 50 GB quota, so check `myquota` before pulling another multi-GB checkpoint.

`.venv`, `.uv-cache` and all four `.venv-requirements*.txt` files are gitignored. Keep the generated
requirements files on disk: they allow reinstalling after a `.venv` wipe without repeating the
network-bound `uv pip compile`.
