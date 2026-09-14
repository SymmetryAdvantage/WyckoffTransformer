# The iapetus environment

WyFormer runs inside
`pytorch:2.14.0-cuda11.8-py312-universal`, the custom image from
`/mnt/hdd/kna/pytorch-research/`. It contains Python 3.12, custom PyTorch
2.14.0.post2, CUDA 11.8, and kernels for both the K20c (`sm_35`) and GTX 750 Ti
(`sm_50`). It is matched to the host's NVIDIA 470.256.02 driver.

The project venv is created **inside that container**, at the bind-mounted
repository path. It inherits the container site-packages so torch remains the
image's custom build. Never create or activate this venv on the host: the host
default Python is 3.14 and the venv's interpreter and torch libraries belong to
the image.

## Creating or repairing the venv

From the checkout the venv is for -- the main one or a worktree:

```bash
scripts/platforms/iapetus/build_venv.sh
```

It runs `scripts/build_singularity_venv.sh`, the generic builder meant for
distributable images, inside the image through `run.sh`. Anything iapetus alone
needs belongs in `build_venv.sh`, not in that script. `build_venv.sh` passes
`BASE_PYTHON=/opt/venv312/bin/python`, then installs the CPU-only Warp
wheel (below) and fails if Triton is importable. `WYFORMER_PLATFORM=iapetus`
links `CLAUDE.local.md` to [agent_brief.md](agent_brief.md); see
[../README.md](../README.md). Measured on 2026-09-15: **34 s** and **1.1 GiB**
per checkout, with nothing to download once `~/.cache/uv` is warm.

The build script compiles a fully pinned requirements file, removes torch and
its CUDA stack, then installs those requirements and the editable project with
`--no-deps`. `uv sync` is intentionally not used: it would install the lock's
ordinary torch selection over the custom container build.

The image keeps torch in a venv of its own, `/opt/venv312`, and
`--system-site-packages` reaches only the interpreter beneath it
(`/opt/uv-python/cpython-3.12...`), which has no torch. The build therefore
writes `container-base.pth` into the venv's site-packages, naming
`/opt/venv312/lib/python3.12/site-packages`. Paths in a `.pth` are appended
after the venv's own, so a package installed in `.venv` (Warp) wins over the
image's copy.

The uv cache is the host's `~/.cache/uv`, which `run.sh` mounts. It is on a
different mount from the checkout inside the container, so uv copies rather
than hardlinks; that is where the 1.1 GiB goes.

### Triton exclusion

Do not install Triton in the iapetus-specific venv. Its K20c GPUs are compute
capability 3.5, while Triton's GPU compiler requires capability 7.0 or newer.
If Triton was added while testing, remove it from the venv:

```bash
scripts/platforms/iapetus/run.sh /usr/local/bin/uv pip uninstall --python .venv/bin/python triton
```

### Moving a single pinned dependency into the venv

A full rebuild with `build_venv.sh` takes well under a minute with a warm uv
cache and restores the Warp wheel, so it is the default way to pick up changed
pins. To move one package without touching the rest -- say while another
process is using the venv -- compile just the direct requirements and install
that one line, so `pyproject.toml` stays the only place the pin is written
(`pyxtal` as the example):

```bash
scripts/platforms/iapetus/run.sh bash -lc '
    uv pip compile --no-deps --no-annotate --no-header pyproject.toml \
        | grep "^pyxtal " > /tmp/pyxtal.txt
    uv pip install --python .venv/bin/python --no-deps -r /tmp/pyxtal.txt'
```

`--no-deps` on both commands is what keeps this safe: without it the resolver
reconsiders the whole closure and puts a PyPI torch over the container's custom
build. Note that a bare `uv pip install pyxtal` does **not** work here —
`tool.uv.sources` applies only to a requirement that comes from the project's
own dependency list, so the source has to reach uv through `pyproject.toml`.

### CPU-only Warp with GPU ORB inference

The base image currently carries the stock CUDA-enabled `warp-lang` wheel,
which requires a newer driver than iapetus has. `build_venv.sh` installs the
CPU-only `warp-lang` wheel built for this host,
`/mnt/hdd/kna/pytorch-research/archive/dist-warp/warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl`,
into `.venv`, where it takes precedence over the inherited package (it warns
and skips the step if the wheel has moved; `WYFORMER_WARP_WHEELS` names another
directory). This is intentional: Warp constructs ORB's neighbor list on the
CPU, while the ORB model and its forward evaluation still run on the selected
GPU.

The stock `orb-models` calculator passes its CUDA device to Warp and therefore
does not work with this CPU-only wheel. Use the CPU-neighbour/GPU-forward
calculator path in `scripts/run_cryspr_reconstruction_study.py`:
`build_patched_orb_calculator(device="cuda")`. It constructs the graph with
`device="cpu"` and transfers the resulting batch to the CUDA model. Do not
replace the CPU-only Warp wheel with the inherited CUDA wheel: Warp 1.17
requires a CUDA 12 driver, whereas iapetus is limited to driver 470 / CUDA
11.4.

The container entrypoint detects the owner of `/workspace` and drops to that
UID/GID, so files written to the mounted checkout remain host-owned.

## Data store, cache and runs

`~/.config/wyformer/paths.env` on iapetus (see [../../data_store.md](../../data_store.md)):

```
WYFORMER_DATA=/mnt/hdd/kna/wyformer/data
WYFORMER_CACHE=/mnt/hdd/kna/wyformer/cache
WYFORMER_RUNS=/mnt/hdd/kna/wyformer/runs
WANDB_DIR=/mnt/hdd/kna/wyformer/runs
```

All four are on the 916 GiB HDD, `/mnt/hdd`, not the 233 GiB root filesystem.
The container cannot read the host's config file, so `run.sh` resolves the four
keys with `scripts/wyformer_paths.sh` on the host, passes them in the
environment tier, and mounts each directory at its host path. Inside the
container `python -m wyckoff_transformer.paths` therefore reports the config
file as *absent* and every key as `[environment]` -- that is correct.

An older checkout at `/mnt/hdd/kna/WyckoffTransformer` (branch
`training-loss-fixes`, with uncommitted changes and an in-checkout `cache/`,
`data/` and `runs/`) predates the store. It is not the project checkout; leave
it alone unless its owner says otherwise.

## Git worktrees: one venv each

Every worktree builds its own venv, with the same command run from inside it:

```bash
scripts/platforms/iapetus/build_venv.sh
```

Inside the container every checkout is `/workspace`, so a venv's editable
install and script shebangs always point at the checkout `run.sh` was started
from. Do not share one venv between checkouts anyway: a rebuild for one
checkout's `pyproject.toml` would change what the others import.

`run.sh` also handles the worktree's `.git`, which is a file naming
`/home/kna/WyckoffTransformer/.git/worktrees/<name>` by host path: it mounts the
main checkout's `.git` directory read-only at that path, so `git` and W&B's
commit capture work inside the container. Verified on 2026-09-15 with a worktree
under `.claude/worktrees/`: `git status` and the branch name resolve, the store
paths resolve as in the main checkout, and `import wyckoff_transformer` loads the
worktree's `src`.

## Do not use host-local torch setup

Do not copy `uv.toml.cpu` to `uv.toml`, install a PyPI CUDA torch wheel, or use
`uv.toml.local`. The first two bypass the custom Kepler/Maxwell-compatible
torch; the last points at zeus's private wheel index.
