# The zeus environment: building, updating, and why it looks like this

WyFormer runs on zeus in an ordinary host venv — no container. The thing that
makes this environment specific to zeus is that **torch is built here**, from
source, and installed from a flat file index at `/mnt/hdd/torch_wheels/`.

Read this before running `uv sync` by hand, changing `uv.toml`, or adding a
dependency. For day-to-day running see [usage.md](usage.md).

---

## Creating or repairing the environment

One command:

```bash
scripts/platforms/zeus/env_init.sh              # dev + relax + nep
scripts/platforms/zeus/env_init.sh --dry-run    # show the plan, change nothing
WYFORMER_EXTRAS="dev relax nep research" scripts/platforms/zeus/env_init.sh
```

It does two things: copies `scripts/platforms/zeus/uv.toml` to the repository
root, where uv looks for it, and then runs `uv sync` with the extras this host
wants. Any extra arguments are forwarded to `uv sync`, so `--dry-run`,
`--inexact` and `--upgrade` all work.

It does **not** run `uv venv` first. `uv sync` creates `.venv` if it is
missing, honouring `requires-python` from `pyproject.toml`, whereas `uv venv`
would destroy and recreate an environment that already exists. The script is
therefore safe to re-run.

To build into a scratch location without disturbing the working venv — useful
while a long job is running out of `.venv`:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/venv-test scripts/platforms/zeus/env_init.sh
```

Note that `uv pip list` does **not** honour `UV_PROJECT_ENVIRONMENT`; to
inspect a scratch venv, pass `--python /tmp/venv-test/bin/python`.

### Refreshing a rebuilt wheel with the same version

Local wheel filenames can stay the same after a rebuild. A normal sync then
leaves the old binary installed. Force a reinstall from the wheel itself:

```bash
uv pip install --python .venv/bin/python --reinstall --no-deps \
  '/mnt/hdd/torch_wheels/torch-2.14.1+cu134-cp314-cp314-linux_x86_64.whl'
```

The Python 3.14 wheel rebuilt on 2026-10-23 links to `libcudss.so.1`, supplied
by the current host cuDSS. The older build needs the removed `libcudss.so.0`.
Do not symlink across those ABI versions. Only install wheels compatible with
the venv's Python and platform; older `cp312` builds in the directory are for
Python 3.12.

---

## Git worktrees: one venv each

Give every worktree its own venv, built by the same command from inside it:

```bash
scripts/platforms/zeus/env_init.sh
```

It is cheap here. uv hardlinks packages out of `~/.cache/uv`, which is on the same
NVMe as the checkouts: measured on 2026-09-14, a worktree venv took **19 s** and
**29 MB** of new disk, downloading 3 packages.

Do not borrow the main checkout's venv instead. Its editable install points at
`/home/kna/WyckoffTransformer/src`, so a worktree running that interpreter imports
the main checkout's code, and `uv run` from the worktree repoints that install at
the worktree -- changing what the main checkout imports. A copied `.venv` has the
same problem: its scripts and `.pth` name the checkout it was built in.

`uv.lock` and `uv.toml` are untracked (see below), so a fresh worktree has neither.
Claude Code copies them from the main checkout through `.worktreeinclude`; a worktree
made with a bare `git worktree add` needs `cp /home/kna/WyckoffTransformer/uv.lock .`
before `env_init.sh`, or the sync resolves afresh and can install versions the main
checkout does not run. `env_init.sh` supplies `uv.toml` itself.

What a worktree venv gives you, verified: `import wyckoff_transformer` and every
repository-relative path (`yamls/`, `generated/`) resolve inside the worktree; the
main checkout's venv is untouched; torch is the locally built wheel. The `nep`
extra now includes calorine in the worktree venv (see below).

---

## Where the data, cache and runs live

zeus keeps them under XDG's data directory, on the NVMe root, as recorded in
`~/.config/wyformer/paths.env` (format and rules: `docs/data_store.md`):

```
WYFORMER_DATA=/home/kna/.local/share/wyformer/data
WYFORMER_CACHE=/home/kna/.local/share/wyformer/cache
WYFORMER_RUNS=/home/kna/.local/share/wyformer/runs
WANDB_DIR=/home/kna/.local/share/wyformer
```

The paths are written out literally. `XDG_DATA_HOME` is **unset** here -- it is a
specification default, not an exported variable -- and the file is never
expanded anyway, so `$XDG_DATA_HOME/wyformer` would be an error, not
`~/.local/share/wyformer`.

The checkout and `~/.local/share` are the same filesystem, so moving a directory
between them is a rename, not a copy. Nothing on this machine may assume the old
in-checkout paths: there are no compatibility symlinks, and other projects that
read `/home/kna/WyckoffTransformer/data/...` directly (DiffCSPNew) have to point at
the store instead.

`~/.config/wyformer/env.sh` is a leftover of an earlier plan to export all four
locations from `~/.bashrc`. Nothing reads it.

The untracked datasets were moved into the store on 2026-09-14: `formula_energy`,
`lemat-bulk`, `lemat_bulk_fmax1`, `lemat_bulk_fmax1_stress`,
`unique_fingerprints.parquet`, and the three `mp_2026_gga_gap` CSVs (its README is
tracked and stayed). Everything tracked by git stayed in the checkout.

## The state of the current venv

Rebuilt 2026-10-16 on Python 3.14. The host's system `python3` moved from
3.12.3 to 3.14, which left the old venv's `/usr/bin/python3` link pointing at
an interpreter with no matching `site-packages`. `env_init.sh` now syncs with
`--python 3.14`, while `pyproject.toml` still admits 3.12 for the iapetus and
aspire2a containers. Two traps on the way:

- `orb-models` 0.7.0 pins `dm-tree ==0.1.8`. That release has no cp314 wheel,
  and its bundled abseil does not compile with GCC 15. `pyproject.toml`
  overrides the pin to 0.1.9, which builds from source.
- That build, and any other old pybind11 sdist, needs
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` under CMake 4. uv caches the resulting
  wheel, so only the first build after a cache purge needs it:
  `CMAKE_POLICY_VERSION_MINIMUM=3.5 scripts/platforms/zeus/env_init.sh`.

| | |
| --- | --- |
| Python | 3.14.3, uv-managed (`~/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu`) |
| `include-system-site-packages` | `false` — nothing is inherited from the host Python |
| torch | 2.14.1+cu134, CUDA 13.4, locally built wheel |
| triton | 3.8.0, from PyPI, required by the torch wheel's own METADATA |
| NVIDIA CUDA runtime distributions | none — CUDA and MKL come from the host; `nvidia-ml-py` is the Python NVML interface |
| Extras installed | `dev`, `relax`, `nep` (the `env_init.sh` default) |
| Project | installed editable (`_editable_impl_wyckoff_transformer.pth`) |

The Torch 2.14.1 wheel's NVSHMEM RUNPATH names a removed build directory.
The host library is installed by `libnvshmem3-cuda-13`, but its subdirectory
is absent from the system loader path. Link it into Torch's `$ORIGIN` search
directory, without installing a conflicting NVIDIA runtime wheel:

```bash
ln -s /usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3 \
    .venv/lib/python3.14/site-packages/torch/lib/libnvshmem_host.so.3
```

`env_init.sh` creates this link after a real sync (including scratch venvs),
and leaves `--dry-run` untouched. This fixes `import torch` and GPU use with
no shell activation or library-path exports. Reinstalling Torch removes the
link; recreate it after an explicit wheel reinstall.

---

## The local wheel index

`uv.toml` in the repository root is what wires it up. It is **untracked**
(`.gitignore`), because every setting in it is true of this machine only; the
tracked copy is `scripts/platforms/zeus/uv.toml`, and `env_init.sh` puts it in
place. Its core is the index:

```toml
[[index]]
name = "zeus"
url = "file:///mnt/hdd/torch_wheels/"
format = "flat"
```

Two packages come from that index. Of the 287 packages the lock resolves, the
rest come from PyPI, git or local path sources:

| Package | Version |
| --- | --- |
| `torch` | 2.14.1+cu134 |
| `spglib` | 2.7.0 |

The directory also holds wheels for unrelated work on this box (`vllm`,
`flash_attn`, `nixl`, `torchvision`, several older `torch` builds, and
`torch-scatter`/`torch-sparse`, left over from when the now-removed `cdvae`
extra needed them). It is a shared scratch index, not a WyFormer artefact.
Nothing prunes it.

### `uv.lock` is untracked too, and deliberately

`uv.lock` records the index as the *source* of torch:

```toml
name = "torch"
version = "2.14.1+cu134"
source = { registry = "/mnt/hdd/torch_wheels/" }
```

A lockfile containing that line is wrong for every machine except this one, so
`uv.lock` is gitignored. `.gitignore` carries the reasoning, and notes that the
line was removed twice before and came back both times. Do not commit the lock,
and do not delete the ignore entry.

The practical consequence: on zeus the lock is **local state**. It resolves
against whichever `uv.toml` is active. If you switch `uv.toml` (e.g. to
`scripts/platforms/cpu/uv.toml`), the existing lock is meaningless and must be
regenerated.

---

## The build environment

`~/.profile` sources `~/.torch_profile.sh` in every login shell. That file is
the reason the local wheels exist and work:

```bash
source /opt/intel/oneapi/2025.3/oneapi-vars.sh   # unless SETVARS_COMPLETED=1
export TORCH_CUDA_ARCH_LIST="8.9"                # RTX 6000 Ada only
export CFLAGS="-march=native"
export CXXFLAGS="-march=native"
export BLAS="MKL"
export USE_CUDNN=1 USE_CUSPARSELT=1 USE_CUDSS=1 USE_CUFILE=1 USE_TENSORRT=ON
export USE_MKL=1 USE_MKLDNN=ON USE_OPENMP=1 USE_NUMA=1 USE_NCCL=1
export _GLIBCXX_USE_CXX11_ABI=1
```

`/etc/profile.d/cuda.sh` adds `CUDA_HOME=/usr/local/cuda` (→ 13.4), and
`/etc/ld.so.conf.d/mkl.conf` puts oneAPI's MKL on the system library path.

Three consequences worth knowing:

**The wheels are not portable.** `TORCH_CUDA_ARCH_LIST=8.9` means sm89 kernels
only, and `-march=native` targets this Xeon. Copying anything out of
`/mnt/hdd/torch_wheels/` to another machine is not safe. For a portable GPU
build, use the Singularity images instead — see
[../../singularity_images.md](../../singularity_images.md).

**`-march=native` is exported to every shell**, not just to deliberate torch
builds. Any dependency that compiles from source in this environment — a
`uv pip install` of a package with no wheel — silently inherits it and produces
a binary that only runs on this CPU. That is fine on zeus and a trap if the
venv is ever copied.

**Nothing needs to be sourced at run time.** The MKL dependency is satisfied by
a RUNPATH baked into the wheel:

```text
$ORIGIN:...:/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/local/cuda/lib64:...
```

Verified: `env -i .venv/bin/python -c "import torch; torch.cuda.is_available()"`
works with no oneAPI variables set at all. The flip side is that torch is
pinned to those absolute host paths — removing or relocating oneAPI's MKL, or
`/usr/local/cuda`, breaks `import torch`.

The oneAPI initialiser also **prints a banner on every login shell**, which
lands in the middle of scripted output. Set `SETVARS_COMPLETED=1` to suppress
the re-init, or read the value you want with `env -i`.

**MAGMA is compiled in.** `torch._C._has_magma` is `True` at runtime. The
`magma_*`/`magmablas_*` symbols are statically linked into
`libtorch_cuda_linalg.so` (630 of them, verified with `nm -D`); there is no
separate `libmagma.so` to keep on the library path.

**OpenMP comes from `libgomp`, not `libiomp5`.** The build links
`libmkl_gnu_thread.so.3` (MKL's GNU-threading variant), and every
OpenMP-using library in the wheel (`libtorch_cpu.so`, `libshm.so`,
`libtorch_cuda_linalg.so`, `libtorch_global_deps.so`) needs `libgomp.so.1`
only. Nothing needs `libiomp5.so` / `libomp.so.5`, so there is no
Intel-vs-LLVM OpenMP conflict to work around on this box.

---

## Syncing the environment

`uv sync` installs only the extras it is *told* about, and prunes everything
else. Measured with `--dry-run`, a bare `uv sync` here would uninstall 42
packages, including:

```text
mace-torch  orb-models  triton  torchmetrics  torch-ema  matscipy  h5py  pytest  ruff
```

That is `uv sync` doing its job, not a bug. The extras therefore have to be
named, which is what `env_init.sh` exists for:

```bash
uv sync --extra dev --extra relax --extra nep
```

### Why the extras are not in `uv.toml`

They cannot be. `uv.toml` accepts an `extra = [...]` key, but it is honoured by
the `uv pip` interface and **ignored by `uv sync`**. Verified: with
`extra = ["dev", "relax", "nep"]` in `uv.toml`, `uv sync --dry-run` still
plans to remove MACE, ORB, pytest, ruff and triton. `default-groups` in
`uv.toml` is ignored by `uv sync` too, and
`--extra` has no environment variable. The command line is the only channel, so
the extras live in `env_init.sh` and `scripts/platforms/zeus/uv.toml` carries a
comment saying why it does not set them.

### Two flags worth knowing

`--inexact` makes a sync additive: it installs what the lock requires without
removing anything else. Useful when you want the lock's packages but not its
opinion about what should be absent.

`uv sync` never *upgrades* a package the lock already pins. To move a pin:

```bash
uv lock --upgrade-package mace-torch
```

That is how the mace-torch pin was moved: the venv had a hand-installed 0.3.16
while the lock still said 0.3.15, and `uv lock --upgrade-package mace-torch`
brought the lock forward rather than letting a sync downgrade the venv.

A dozen peripheral transitive packages are still ahead of the lock — `boto3`,
`botocore`, `cffi`, `configargparse`, `cryptography`, `google-*`, `lmdb`,
`prettytable`, `opentelemetry-api`. A sync moves them back to their locked
versions, which is the point of having a lock; **nothing WyFormer depends on is
in that set** (no torch, pymatgen, pyxtal, spglib, numpy, pandas, scipy, wandb,
MACE, ORB, ASE or matminer). Use `uv lock --upgrade` if you deliberately want
the lock to catch up instead.

---

## Adding dependencies

```bash
uv pip install <package>    # from inside the checkout; uv finds its .venv, no activation
```

torch lives inside the venv, so uv can see it and will not try to reinstall it;
no constraint files are needed. Two things can still go wrong:

**A package that requires a different torch** will drag a PyPI torch in and
shadow the local build. `uv pip list | grep -c '^nvidia'` should stay at 0; if
`nvidia-*` wheels appear, a manylinux torch was installed and the locally built
one is gone.

**A package with no wheel** compiles against `-march=native` and the oneAPI
toolchain. That works, and pins the result to this box.

If the dependency is permanent, add it to `pyproject.toml` and re-run
`env_init.sh` rather than leaving it as a hand-install that the next sync
removes.

### calorine and NEP89

The `nep` extra provides `calorine`, which runs the NEP89 pre-relaxation
potential (`--prerelax-mlip nep89`, `--stage prescreen`; see
[the variants doc](../../de_novo_ranking_protocol_nep89_variants.md)). As of
2026-09-27 the zeus lock resolves calorine 4.0, whose metadata has no NumPy
upper bound; `scripts/platforms/zeus/env_init.sh` installs it with the `nep`
extra while retaining NumPy 2.5.3. Earlier calorine releases had an upper pin
that made an unconstrained sync downgrade NumPy.

The sdist build has one host-specific consequence:

- It compiles under this box's `-march=native`, like anything else without a
  wheel here. Fine on zeus, not portable off it.

The model file itself is not a dependency: it is 15 MB of plain text pinned by
GPUMD commit and SHA-256 in `cryspr/nep89.py`, downloaded on first use into
`~/.cache/wyckoff_transformer/nep_models/`.

### `[tool.uv.extra-build-dependencies]` and the shadowing warning

Every uv invocation in this repository prints:

```text
warning: Found both a `uv.toml` file and a `[tool.uv]` section in an adjacent
`pyproject.toml`. The following fields from `[tool.uv]` will be ignored in
favor of the `uv.toml` file:
- extra-build-dependencies
```

A `uv.toml` replaces the `[tool.uv]` table wholesale rather than merging with
it, so `pyproject.toml`'s `extra-build-dependencies` — which tells uv that
`torch-scatter` and `torch-sparse` need torch present at build time — is inert
on zeus. `scripts/platforms/zeus/uv.toml` therefore repeats it:

```toml
[extra-build-dependencies]
torch-scatter = ["torch"]
torch-sparse = ["torch"]
```

The warning still prints — uv emits it whenever both files exist, whatever
their contents — but the effective value now matches `pyproject.toml`, so
nothing is silently lost. It does not bite today anyway, because both packages
come prebuilt from the local index; it would bite if `uv.toml` ever pointed at
an index without them.

The copy-then-sync shape of `env_init.sh` is the same one CI uses, which does
`cp scripts/platforms/cpu/uv.toml uv.toml` before `uv sync --extra dev --extra
research --extra relax` (`.github/workflows/pytest.yml`). Per-host `uv.toml`
plus command-line extras is the established pattern in this repository, not a
zeus invention.

---

## `torch.compile`, triton, and CPU-only installs

`torch.compile` requires Triton. The `compile` extra is deprecated and has
been removed from `pyproject.toml`; do not request it in `uv sync` or
`WYFORMER_EXTRAS`. Use the initializer's `dev relax nep` defaults.

The local Torch 2.14.1+cu134 wheel requires `triton==3.8.0` on Linux x86_64,
so dependency resolution installs it automatically. The host's `uv.toml`
also pins that version through `constraint-dependencies`. Update that pin
when rebuilding Torch, checking the wheel's `Requires-Dist` metadata.

CPU-only Torch builds do not need Triton. Keeping it out of the project's
base dependencies avoids installing a GPU compiler in CPU-only environments.
An older source-built Torch wheel may not declare Triton; report a missing
runtime dependency before changing that host's installation recipe.

`cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-cu13` and
`cuequivariance-ops-torch-cu13` 0.12.0 were installed on 2026-10-05 with
`uv pip install --no-deps` for EquFlashV2. The ops package declares a
`nvidia-cublas` dependency, but this host supplies cuBLAS through system CUDA
13.4. Do not install the CUDA extra with unrestricted dependency resolution:
it can replace host libraries or local Torch. The EquFlashV2 GPU energy,
forces and stress probe passed. See [mlip_bias.md](mlip_bias.md) for the
separate model overlays and provenance.
