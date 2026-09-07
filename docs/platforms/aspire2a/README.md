# WyFormer on ASPIRE 2A

NSCC Singapore's ASPIRE 2A. This directory covers the **AI partition** — the
`asp2a-gpu*` nodes reached through the `ai` routing queue on PBS server
`pbs102`. Login nodes are `asp2a-login-nus01/02`.

Two things shape everything else here:

1. **WyFormer runs inside a Singularity container.** The host has no `python`
   and no `uv` on `PATH` (and no `git` until `module load git/2.39.2`), and the compute nodes are RHEL 8.10 with a
   glibc far older than the wheels want. The repo's `.venv` is built
   `--system-site-packages` against a stock PyTorch image and is only valid
   *inside* that image. Never run `python`, `uv` or `pytest` against it from a
   host shell — see [environment.md](environment.md#never-run-uv-from-the-host).
2. **Nothing runs interactively for long.** PBS caps a GPU job at 24 h (2 h in
   the dev queue). Every real WyFormer run is therefore a **self-chaining PBS
   job** that checkpoints, hands the node back before the wall, and re-submits
   itself. See [usage.md](usage.md).

## Which page do you want?

| I want to... | Read |
| --- | --- |
| submit training, generation or a relaxation pool | [usage.md](usage.md) |
| rebuild the venv, add a dependency, understand the container | [environment.md](environment.md) |
| work out why something is broken | [troubleshooting.md](troubleshooting.md) |

---

## Quick start

Everything is already built. From a login node:

```bash
cd /scratch/users/nus/kna/WyckoffTransformer
bash scripts/train_in_pb.sh yamls/models/NextToken/v6/base_sg.yaml mp_20 --pilot
```

That submits a 2 h pilot to the dev queue and prints the job id. For a real run,
drop `--pilot`: it goes to `aiq1` (24 h, 1 GPU) and chains itself to completion.

To run something by hand on a GPU, take a dev node first:

```bash
qsub -I -q ai -P 11001786 -l select=1:ngpus=1:ncpus=16:mem=110gb -l walltime=02:00:00
cd /scratch/users/nus/kna/WyckoffTransformer
module load singularity
bash scripts/run_in_singularity.sh python -c "import torch; print(torch.cuda.is_available())"
```

Three things to know before your first run:

1. **You cannot `qsub -q aiq1`.** Every AI queue is `from_route_only`. You
   submit to `-q ai` and the router picks the queue from `(ngpus, walltime)`.
   See [usage.md](usage.md#queues).
2. **Submit from a login node.** A job cannot `qsub` to the other PBS server,
   and it cannot `ssh` to a login node either.
3. **Do not `uv sync` / `uv run` from the host shell.** It deletes `.venv`, and
   right now that `.venv` is shared with every running job.

---

## Hardware

The AI partition is 18 `asp2a-gpu*` nodes on AMD EPYC 7713 (2x64 cores, 8 NUMA
nodes). `asp2a-gpu002` and `asp2a-gpu003` were both NVIDIA **A100-SXM4-40GB**,
driver **570.124.06** (CUDA 12.8); PBS does not advertise the GPU model, so the
other nodes are assumed but unverified.

| Nodes | GPUs | CPUs (`ncpus`) | RAM | `node_pool` |
| --- | --- | --- | --- | --- |
| `asp2a-gpu001`–`006` | 8 | 128 | 990 GB | `4gpu` (005, 006: `8gpu`) |
| `asp2a-gpu007`–`018` | 4 | 64 | 462 GB | `4gpu` (013: `testing`) |

Jobs are cgroup-confined: with `ncpus=16` the job sees `nproc` 16 and one GPU,
regardless of what the node holds. The AI queues default to
`node_pool=4gpu`, so a 1-GPU job lands anywhere except `gpu005/006/013`.

---

## Layout

| Path | What |
| --- | --- |
| `/scratch/users/nus/kna/WyckoffTransformer` | The working checkout (Lustre) |
| `<repo>/.venv` | Python 3.12.3 venv, **container-only**, `--system-site-packages` |
| `<repo>/.uv-cache` | uv cache (3.0 GB), kept off the home quota |
| `~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif` | The base image, 12 GB |
| `~/.cache/cached_path/` | ORB checkpoints — on the **home** quota, see below |
| `/raid` | Node-local NVMe, 14 TB, per-job dir `/raid/pbs.<jobid>` |

Storage and quotas (`myquota`, `myprojects` from `/app/apps/local/bin`):

| Mount | Type | Quota | Used | Holds |
| --- | --- | --- | --- | --- |
| `/home/users/nus/kna` | GPFS | **50 GB** | 34.6 GB | the `.sif` (12 GB), `~/.cache` (7 GB), `~/.netrc` |
| `/scratch/users/nus/kna` | Lustre | 100 TB | 82.5 GB | the checkout, `cache/` (23 GB), `runs/`, `.venv` |
| `/raid` | node-local XFS | — | — | throwaway job scratch, **gone when the job ends** |

The **home quota is the tight one**: 50 GB, a quarter of it already the
container image, and `cached_path` (ORB checkpoints) ignores `XDG_CACHE_HOME`
and writes there. Watch it before pulling another MLIP.

Compute budget: project `11001786`, **1.91 M SU left of 46.3 M** as of
2026-09-07. A 24 h 1-GPU job is not free; check `myprojects` before queueing a
long chain.

---

## Scripts

Contrary to the rule in `AGENTS.md`, the ASPIRE 2A-specific scripts are **not**
under `scripts/platforms/aspire2a/` — they are still in `scripts/`:

| Script | What |
| --- | --- |
| `run_in_singularity.sh` | run any command against `.venv` inside the image |
| `build_singularity_venv.sh` | build `.venv` (run *inside* the image) |
| `train_in_pb.sh` | the general self-chaining training launcher |
| `train_ehull_5x.pbs`, `train_ehull_ssops.pbs`, `train_formula_energy.pbs` | its single-purpose predecessors |
| `protocol_relax.pbs` | self-chaining relax + score for one generated pool |

One script *is* in the right place, because nothing running depends on its path:

| Script | What |
| --- | --- |
| `scripts/platforms/aspire2a/prefetch_cached_path.sh` | parallel-range fetch of a checkpoint into the `cached_path` cache, ETag-verified |

They cannot be moved while chains are in flight: a running link re-`qsub`s
**itself** by absolute path, so renaming the file breaks the chain mid-run. Move
them once no `wyf_*` job is queued or running, and update the `$REPO/scripts/...`
paths inside each one at the same time.

---

## Verified state

Confirmed on 2026-09-08 on `asp2a-gpu002`, branch `training-loss-fixes`:

| Check | Result |
| --- | --- |
| container | `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-devel` (Ubuntu 24.04), built 2026-09-04 |
| torch | `2.14.0+cu126`, from the container's `dist-packages`, `cuda.is_available()` True |
| GPU seen by the job | 1x A100-SXM4-40GB, 39.5 GiB, sm80 |
| venv python | 3.12.3; `wyckoff-transformer` 1.0.6 editable |
| key versions | pymatgen 2026.5.4, numpy 2.5.3, triton 3.8.0, pyxtal 1.1.4, spglib 2.7.0 |
| relax extra | orb-models 0.7.0, mace-torch 0.3.16 — present |
| `pytest` | **633 passed, 39 skipped, 5 errors, 1483 subtests, 204 s** — plus one collection error |
| the 6 failures | all `matminer` 0.8.0 vs new scipy/pymatgen — see [troubleshooting.md](troubleshooting.md#matminer-is-broken-against-the-resolved-scipy-and-pymatgen) |
| W&B online from a compute node | works, `~/.netrc`, entity `kazeev` |
| HuggingFace download | ~16 MB/s from a compute node |
| ORB checkpoint S3 (`us-west-1`) | 32 kB/s on one connection, ~400 kB/s on 16 — [troubleshooting.md](troubleshooting.md#the-orb-checkpoint-downloads-at-a-crawl) |
| orb-v3 checkpoint (102 MB) | pre-fetched and ETag-verified into `~/.cache/cached_path` |
| `ssh` compute -> login node | refused (`Connection closed`) |

Four production chains (`wyf_ehull5x`, `wyf_ssops`, `wyf_gene_*`, `wyf_relax`)
were running against the shared `.venv` during these checks. That is the normal
state, and it is why nothing here installs into `.venv` casually.
