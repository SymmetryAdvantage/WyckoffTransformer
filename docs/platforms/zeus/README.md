# WyFormer on zeus

`zeus.nus.edu.sg` — Ubuntu 24.04.4, kernel 6.8.0-138, Intel Xeon w7-3455
(**24 physical cores**, 48 threads), 1 TB RAM (1,007 GiB), 2x NVIDIA RTX 6000 Ada
Generation (46 GB each), NVIDIA driver **610.57.04**, CUDA toolkit **13.3**.

WyFormer runs **directly on the host** here, in an ordinary uv venv — no
container. The one unusual part is the other end: **torch is a locally built
wheel**, served from a flat file index at `/mnt/hdd/torch_wheels/`, compiled for
this box's GPUs (sm89), this box's CPU (`-march=native`) and Intel MKL.

There is **no scheduler**. zeus is an interactive machine shared with other
users and with unrelated LLM work; you pick your own GPU and your own worker
count.

## Which page do you want?

| I want to... | Read |
| --- | --- |
| run training, generation or tests | [usage.md](usage.md) |
| rebuild the venv, add a dependency, understand the local wheel index | [environment.md](environment.md) |
| work out why something is broken | [troubleshooting.md](troubleshooting.md) |

---

## Quick start

Everything is already built. Activate the venv and run:

```bash
cd /home/kna/WyckoffTransformer
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```

To create or repair the environment, one command does it:

```bash
scripts/platforms/zeus/env_init.sh          # add --dry-run to see the plan first
```

Two things to know before your first run:

1. **Pick a GPU explicitly.** Both cards are shared and usually busy. See
   [usage.md](usage.md#choosing-a-gpu).
2. **Sync through `env_init.sh`, not a bare `uv sync`.** A bare sync prunes the
   venv down to the base dependencies, taking MACE, ORB, pytest, ruff and
   triton with it. See
   [environment.md](environment.md#syncing-the-environment).

---

## Layout

| Path | What |
| --- | --- |
| `/home/kna/WyckoffTransformer` | The working checkout |
| `<repo>/.venv` | The venv — a plain host venv, python 3.12.3 |
| `<repo>/uv.toml` | Declares the local wheel index. **Untracked**; copied from the file below |
| `scripts/platforms/zeus/uv.toml` | Tracked source of truth for the above |
| `<repo>/uv.lock` | Pins torch to `/mnt/hdd/torch_wheels/`. **Untracked** by design |
| `/mnt/hdd/torch_wheels/` | Flat index of locally built wheels (torch, spglib, pyg, ...) |
| `~/.torch_profile.sh` | The build environment, sourced from `~/.profile` |
| `scripts/platforms/zeus/env_init.sh` | Venv bootstrap: copies the uv.toml, then syncs with the right extras |

Storage:

| Mount | Device | Size | Free | Holds |
| --- | --- | --- | --- | --- |
| `/` | `nvme0n1p3` | 3.4 T | 889 G | home, the checkout, `.venv`, `cache/`, `runs/`, `~/.cache/huggingface` (279 G) |
| `/mnt/hdd` | `sda1` | 15 T | 4.5 T | `torch_wheels/`, bulk data |

---

## Verified state

Confirmed working on 2026-09-07, on branch `training-loss-fixes`:

| Check | Result |
| --- | --- |
| `pytest` | 594 passed, 40 skipped, 10 deselected, 328 subtests, 100 s |
| torch provenance | `2.11.0` / CUDA `13.2`, from `.venv/.../torch`, built locally (`cp312-cp312-linux_x86_64`) |
| `torch.cuda.is_available()` | True, 2 devices, 44.4 GiB each, sm89 |
| CUDA wheels in the venv | **none** — no `nvidia-*` distributions at all; CUDA and MKL come from the host |
| torch import in a bare env (`env -i`) | works; MKL resolves via the wheel's RUNPATH |
| `wyformer-generate --hf-model SymmetryAdvantage/WyFormer-Alex-MP20` | 1000 structures, 5.25 s generation / 9.7 s wall on a contended GPU |
| `torch.compile` | works, via `triton==3.6.0` from the `compile` extra |
| `train.py --pilot` with `compile_model: true` | full run: training plus both evaluation passes, no `TritonMissing` |
| CPU path unaffected | `uv.toml.cpu` + CI's extras resolves `torch 2.11.0+cpu` with **no triton**; adding `--extra compile` brings it in |
| W&B auth | `~/.netrc`, default entity `kazeev`; `symmetry-advantage` team visible |
| Installed extras | `dev`, `relax`, `compile`. Absent: `cdvae`, `research`, `mp`, group `genbench-oracle` |
| `uv lock --check` | lockfile up to date (292 packages: 4 from the local index, the rest PyPI, git or path) |
| `env_init.sh` into a scratch venv | 172 packages, clean install, no `nvidia-*` wheels |
| Cached datasets | `lemat_bulk_fmax1` 13 G, `lemat_bulk_ehull` 5.2 G, `alex_mp_20` 3.5 G, `mp_20` 911 M, and others under `cache/` |

Both GPUs were in use by other jobs during these checks, which is the normal
state of the machine rather than an exception.
