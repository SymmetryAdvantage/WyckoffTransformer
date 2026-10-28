# WyFormer on zeus

`zeus.nus.edu.sg` — Ubuntu 24.04.4, kernel 6.8.0-138, Intel Xeon w7-3455
(**24 physical cores**, 48 threads), 1 TB RAM (1,007 GiB), 2x NVIDIA RTX 6000 Ada
Generation (46 GB each), NVIDIA driver **615.71.09**, CUDA toolkit **13.4**.

On 2026-10-05 the venv was upgraded to local Torch `2.14.1+cu134`, retaining
Triton 3.8.0, for the [EquFlashV2/PET study](mlip_bias.md). Both GPU 1
single-point probes pass. [environment.md](environment.md) documents the
NVSHMEM runtime link needed by this wheel and the controlled cuEquivariance
installation. The older verification table below records the previous setup.

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

Everything is already built for the main checkout. From the root of the checkout
you are working in — `/home/kna/WyckoffTransformer`, or a git worktree after
building its own venv with `env_init.sh` below — run its interpreter directly:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train.py \
    yamls/models/mp_20/der/uncond_adamw_wsd.yaml mp_20 cuda --pilot
```

Agents: don't `source .venv/bin/activate` and don't `cd` to the main checkout
from a worktree. A worktree-isolated Claude Code session refuses the first, and
the second runs the main checkout's code. See [usage.md](usage.md).

To create or repair the environment, one command does it:

```bash
scripts/platforms/zeus/env_init.sh          # add --dry-run to see the plan first
```

Two things to know before your first run:

1. **Pick a GPU explicitly.** Both cards are shared and usually busy. See
   [usage.md](usage.md#choosing-a-gpu).
2. **Sync through `env_init.sh`, not a bare `uv sync`.** A bare sync prunes the
   venv down to the base dependencies, taking MACE, ORB, pytest and ruff
   with it. Triton is required by the local Torch wheel. See
   [environment.md](environment.md#syncing-the-environment).

---

## Layout

| Path | What |
| --- | --- |
| `/home/kna/WyckoffTransformer` | The main checkout |
| `<repo>/.claude/worktrees/<name>` | Git worktrees made by Claude Code, each with its own `.venv` |
| `<repo>/.venv` | The venv — a plain host venv, python 3.14. One per checkout, never shared |
| `/home/kna/.local/share/wyformer/` | The data store: untracked `data/`, `cache/`, `runs/` and `wandb/`, shared by all checkouts. Set in `~/.config/wyformer/paths.env`; see [data_store.md](../../data_store.md) |
| `<repo>/uv.toml` | Declares the local wheel index. **Untracked**; copied from the file below |
| `scripts/platforms/zeus/uv.toml` | Tracked source of truth for the above |
| `<repo>/uv.lock` | Pins torch to `/mnt/hdd/torch_wheels/`. **Untracked** by design |
| `/mnt/hdd/torch_wheels/` | Flat index of locally built wheels (torch, spglib, pyg, ...) |
| `~/.torch_profile.sh` | The build environment, sourced from `~/.profile` |
| `scripts/platforms/zeus/env_init.sh` | Venv bootstrap: copies the uv.toml, then syncs with the right extras |
| `scripts/platforms/zeus/roe_surprisal.sh`, `roe_torpedo.sh` | Rules-of-engagement evaluations; see [below](#rules-of-engagement-evaluations) |

Storage:

| Mount | Device | Size | Free | Holds |
| --- | --- | --- | --- | --- |
| `/` | `nvme0n1p3` | 3.4 T | 712 G | home, the checkouts and their `.venv`s, the data store (45 G), `~/.cache/huggingface` (279 G) |
| `/mnt/hdd` | `sda1` | 15 T | 4.5 T | `torch_wheels/`, bulk data |

---

## Rules-of-engagement evaluations

Two drivers run the [rules of engagement](../../rules_of_engagement.md) on both GPUs
without a scheduler. They are the zeus counterparts of the aspire2a PBS launchers
`roe_surprisal_in_pbs.sh` and `roe_torpedo_in_pbs.sh`, set up for the models trained on
`lemat_bulk_fixed_*`. The gene lookup, structure novelty, the PBE hull of the predicted
`e_hull` and the torpedo residuals all use LeMat-Bulk-fixed. The ORB hull is still
LeMat-Bulk's published one.

```bash
R=$(.venv/bin/python -c "from wyckoff_transformer.paths import runs_root; print(runs_root())")
# One fully relaxed 10,000-gene pool, every mode replayed from it
# (docs/generative_novelty_screen.md). --backbone cfg_fixed or uncond_fixed.
nohup bash scripts/platforms/zeus/roe_surprisal.sh --backbone cfg_fixed \
    > "$R/roe_surprisal/cfg_fixed.log" 2>&1 &
# Chemical-system targets, one 20,000-gene pool, joint- and reference-hull arms.
nohup bash scripts/platforms/zeus/roe_torpedo.sh > "$R/roe/torpedo.log" 2>&1 &
```

- **`--pilot`** runs a 200-gene pool (surprisal) or 3 targets with 10 genes per arm
  (torpedo) in 15–20 minutes and uploads nothing. Run one before a full pool.
- **Every step is skipped if its output exists**, so rerunning the same command resumes.
- **Outputs** go under `$WYFORMER_RUNS/roe_surprisal/<backbone>` and
  `$WYFORMER_RUNS/roe/torpedo_<backbone>`, with a copy of the driver as `driver.sh`.
  Results are uploaded to W&B under `symmetry-advantage`.
- **Devices and workers:** `DEVICES=0,1`, `WORKERS_PER_DEVICE=5` and `PYXTAL_CORES=20` are
  environment overrides. At the defaults a CFG pool relaxes at about 0.8 s per trial,
  about 6 h for its 26,000 trials.
- **Model directories must hold a plain `config.yaml`.** The `config.yaml` among a W&B
  run's files is W&B's wrapped `{value: ...}` format and fails to load. Write
  `OmegaConf.create(dict(run.config))` instead.

---

## Verified state

Confirmed working on 2026-09-09, on branch `training-loss-fixes`, after
switching to the `torch-2.14.0+cu133` wheel and purging the `cdvae` extra that
had been pinning torch to 2.11.0 project-wide:

| Check | Result |
| --- | --- |
| `pytest` | 638 passed, 40 skipped, 10 deselected, 1483 subtests, 120 s |
| torch provenance | `2.14.0+cu133` / CUDA `13.3`, from `.venv/.../torch`, built locally (`cp312-cp312-linux_x86_64`) |
| `torch.cuda.is_available()` | True, 2 devices, sm89 |
| `torch._C._has_magma` | True — MAGMA is compiled in, statically linked (no `libmagma.so` needed) |
| OpenMP | `libgomp.so.1` throughout; no `libiomp5.so` / `libomp.so.5` dependency anywhere in the wheel |
| CUDA wheels in the venv | **none** — no `nvidia-*` distributions at all; CUDA and MKL come from the host |
| `wyformer-generate --hf-model SymmetryAdvantage/WyFormer-Alex-MP20` | 1000 structures, ~5.3 s on a contended GPU |
| `torch.compile` | works, via `triton==3.8.0` required by Torch itself; the former `compile` extra is deprecated |
| Installed extras | `dev`, `relax`, former `compile` (historical; now deprecated). Absent: `research`, `mp`, group `genbench-oracle` |
| `uv lock --upgrade` | resolves cleanly to torch 2.14.0+cu133 / triton 3.8.0 with the `cdvae` pin gone; drops `torch-scatter`/`torch-sparse`, which nothing else needs |
| Cached datasets | `lemat_bulk_fmax1` 13 G, `lemat_bulk_ehull` 5.2 G, `alex_mp_20` 3.5 G, `mp_20` 911 M, and others under `cache/` |

Both GPUs were in use by other jobs during these checks, which is the normal
state of the machine rather than an exception.
