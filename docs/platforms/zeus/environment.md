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
scripts/platforms/zeus/env_init.sh              # dev + relax + compile
scripts/platforms/zeus/env_init.sh --dry-run    # show the plan, change nothing
WYFORMER_EXTRAS="dev relax compile research" scripts/platforms/zeus/env_init.sh
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

---

## The state of the current venv

| | |
| --- | --- |
| Python | 3.12.3, the system interpreter (`home = /usr/bin` in `pyvenv.cfg`) |
| `include-system-site-packages` | `false` — nothing is inherited from the host Python |
| torch | 2.14.0+cu133, CUDA 13.3, locally built wheel |
| triton | 3.8.0, from PyPI, via the `compile` extra |
| `nvidia-*` distributions | none — CUDA and MKL come from the host |
| Extras installed | `dev`, `relax`, `compile` |
| Project | installed editable (`_editable_impl_wyckoff_transformer.pth`) |

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
| `torch` | 2.14.0+cu133 |
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
version = "2.14.0+cu133"
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

`/etc/profile.d/cuda.sh` adds `CUDA_HOME=/usr/local/cuda` (→ 13.3), and
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
uv sync --extra dev --extra relax --extra compile
```

### Why the extras are not in `uv.toml`

They cannot be. `uv.toml` accepts an `extra = [...]` key, but it is honoured by
the `uv pip` interface and **ignored by `uv sync`**. Verified: with
`extra = ["dev", "relax", "compile"]` in `uv.toml`, `uv sync --dry-run` still
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
source .venv/bin/activate
uv pip install <package>
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

`torch.compile` works on zeus. It needs `triton`, which is installed by the
`compile` extra and pinned to `3.6.0` by `constraint-dependencies` in
`scripts/platforms/zeus/uv.toml`.

This used to fail with `TritonMissing`, which matters because
`compile_model: true` appears across `yamls/models/lemat_62/*`,
`lemat_14/*`, `matbench_discovery_mp_2022/*` and `mp_20/**`, and `Trainer`
applies `torch.compile(self.model, fullgraph=False)` when it is set
(`src/wyckoff_transformer/trainer.py:458`). `compile_perceptrons` in the
cascade model uses the same machinery
(`src/wyckoff_transformer/cascade/model.py:327`).

### Why triton is an extra rather than a base dependency

torch declares triton itself — but only in some of its wheels, and whether a
zeus-built wheel does depends on the release process at build time, not
something fixed once and for all:

| torch wheel | triton |
| --- | --- |
| PyPI (GPU) `torch` | pinned for Linux (e.g. `triton==3.6.0; platform_system == "Linux"` on 2.11.0) |
| `download.pytorch.org/whl/cpu` | no triton |
| Built from source on zeus, torch 2.11.0 (previous) | no triton — the release process hadn't started adding the pin to source builds yet |
| Built from source on zeus, torch 2.14.0+cu133 (current) | `triton==3.8.0; platform_system == "Linux" and platform_machine == "x86_64"` — now baked in, same as the official wheels |

So the CPU-only story that kept triton out of the base dependencies is real and
worth preserving: triton is a 640 MB Linux-only GPU compiler, and a CPU-only or
non-Linux install has no use for it. Listing it in `[project.dependencies]`
would inflict it on every such install, and there is no environment marker for
"this machine has a GPU" to condition on.

The two facts are reconciled by making it **opt-in**:

- `pyproject.toml` gains `compile = ["triton >=3.6"]`;
- environments that use a stock PyPI GPU torch, or the current zeus build,
  never need the extra, because torch already pulls triton in;
- environments on a CPU torch, macOS or Windows never get triton, exactly as
  before — including CI, which syncs `dev research relax`;
- zeus asks for the extra explicitly in `env_init.sh` anyway, so a future
  torch rebuild that drops the pin again doesn't silently lose
  `torch.compile`.

The version pin is split for the same reason. The extra keeps a loose floor
(`>=3.6`) because the right triton depends on which torch you have; the host
`uv.toml` pins the exact series with

```toml
constraint-dependencies = ["triton ==3.8.0"]
```

because this host's torch is 2.14.0+cu133, and that wheel's own METADATA
requires exactly triton 3.8.0. Without that constraint nothing stops a
resolver taking whichever triton pairs with the next torch bump on its own
schedule. **Bump both together when torch is rebuilt, and check the new
wheel's `Requires-Dist` for triton rather than assuming the old
"source builds carry no pin" behaviour still holds.**

`cuequivariance_torch` is a separate matter and is still not installed; the MLIP
code gates cuEQ behind an import check and degrades silently. Its
`cuequivariance-cu13` extra also depends on `triton >=3`, which is why triton
was already present in `uv.lock` — locked, but not installed, until now.
