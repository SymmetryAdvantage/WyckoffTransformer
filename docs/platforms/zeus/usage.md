# Running WyFormer on zeus

zeus is an interactive, shared machine with **no scheduler**: no `sbatch`, no
`squeue`, no queue to wait in. You get the resources you take, which means the
etiquette of picking a GPU and capping your worker count is on you.

Everything runs from the activated venv:

```bash
cd /home/kna/WyckoffTransformer
source .venv/bin/activate
```

`uv run python scripts/...` also works and is what the write-ups in `docs/`
use. Be aware that `uv run` reconciles the venv against `uv.lock` first, so it
can move packages under you — see
[environment.md](environment.md#syncing-the-environment).

If the venv is missing or has been pruned, rebuild it with one command:

```bash
scripts/platforms/zeus/env_init.sh
```

For how the environment is built, see [environment.md](environment.md). When
something breaks, see [troubleshooting.md](troubleshooting.md).

---

## Choosing a GPU

Two cards, both healthy, both usually busy:

| `nvidia-smi` index | PCI bus | `/dev/nvidia` minor | Model | Memory |
| --- | --- | --- | --- | --- |
| 0 | `0000:16:00.0` | 0 | RTX 6000 Ada Generation | 46068 MiB |
| 1 | `0000:34:00.0` | 1 | RTX 6000 Ada Generation | 46068 MiB |

All three numberings agree, so `CUDA_VISIBLE_DEVICES=1` is `nvidia-smi`'s
GPU 1 is `/dev/nvidia1`. Both cards are identical, so CUDA's
default `FASTEST_FIRST` ordering has nothing to reorder. Re-verify by UUID if a
card is ever replaced:

```bash
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i).uuid)"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
```

Check the load before you claim one:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

Neither card has recorded any ECC errors. The machine is shared with other
users and with unrelated LLM work, so a card at 90% utilisation with 16 GB in
use is the normal state, not a fault. Always select explicitly rather than
letting torch grab `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py ... cuda
```

Inside the process the selected device is `cuda:0`, so pass `cuda` as the
device argument to the WyFormer scripts.

To see who else is on a card:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -o user=,args= -p <pid>
```

To run something deliberately on CPU only, set `CUDA_VISIBLE_DEVICES=""` — but
note that an *empty* value means "no GPUs visible", so never forward it
accidentally to a job that wants a GPU.

---

## Worker counts: 24 physical cores, not 48

`nproc` reports **48**, and that number is wrong for planning: the Xeon w7-3455
has **24 physical cores** with 2 threads each, on a single NUMA node. There is
1 TB of RAM, so memory is rarely the constraint; core oversubscription is.

Two different defaults are in play:

**pandarallel-based tokenisation gets it right.** `WyckoffProcessor` calls
`pandarallel.initialize()` with no arguments when `n_jobs` is `None`, which
preserves pandarallel's `NB_PHYSICAL_CORES` default of 24
(`src/wyckoff_transformer/wyckoff_processor.py:436`). That is the right number
for this box, so tokenisation needs no tuning.

If you want fewer, you have to do it in Python: `tokenise_dataset` takes
`n_jobs`, but `scripts/tokenise_a_dataset.py --n-jobs` only *looks* like it
works — argparse accepts the flag and the script then raises
`NotImplementedError: n_jobs is not implemented yet` before doing anything
(`scripts/tokenise_a_dataset.py:26`). Omit it.

**The CrySPR relaxation scripts get it wrong.** `scripts/cryspr_orb.py`,
`cryspr_orb_correction.py` and `cryspr_chgnet.py` use `os.cpu_count()`, i.e.
**48**, unless `NP` is set:

```python
try:
    nb_workers = int(os.environ["NP"])
except:  # noqa: E722
    print("Warning: NP variable unspecified, set as all CPU cores available.")
    nb_workers = os.cpu_count()
```

Those scripts also warn if `OMP_NUM_THREADS` is unset, because a pool of
workers each running a multi-threaded BLAS is far slower than a pool of
single-threaded ones. Set both:

```bash
NP=16 OMP_NUM_THREADS=1 python scripts/cryspr_orb.py ...
```

**16 is the established figure** for CPU MLIP work on this box — it fits inside
the 24 physical cores alongside the usual background load. The oracle
reconstruction study in
[`docs/archive/pyxtal_dof_reduction_study.md`](../../archive/pyxtal_dof_reduction_study.md)
used `--workers 16` for 72 worker-hours on that basis.

`wyformer-cryspr` (`src/wyckoff_transformer/cli/relax.py`) takes `--workers`
and defaults to **1**, so it never oversubscribes by accident. It pins
`OMP_NUM_THREADS=MKL_NUM_THREADS=1` and `torch.set_num_threads(1)` in each
worker, which is what makes an N-worker pool actually use N cores.

---

## MLIP relaxation: CPU by default

The CrySPR scripts default to `ORB_DEVICE=cpu`:

```python
DEFAULT_ORB_DEVICE = os.environ.get("ORB_DEVICE", "cpu")
```

That default suits zeus: the GPUs are usually contended while 24 cores are
usually not, and a wide CPU pool beats queueing behind someone else's training
run. `wyformer-cryspr --device` accepts `cpu`, `cuda` or `auto` if you want to
override it for a short job.

`cuequivariance_torch` is **not installed**, so the cuEQ acceleration path for
MACE and TACE is off. Both builders check for it by import and fall back
silently, so nothing breaks; GPU MACE is simply not accelerated. Installing the
`cuequivariance-cu13` extra would also pull in `cuequivariance-ops-torch-cu13`
against a locally built torch, which is untested here — see
[environment.md](environment.md#torchcompile-triton-and-cpu-only-installs).

---

## Common tasks

Pilot training run, end to end:

```bash
source .venv/bin/activate
python scripts/cache_a_dataset.py mp_20
python scripts/tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
CUDA_VISIBLE_DEVICES=1 python scripts/train.py yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```

Much of this is already cached — `cache/` holds `mp_20`, `alex_mp_20`,
`lemat_bulk_ehull`, `lemat_bulk_fmax1`, `mp_2022`, `mpts_52`, `carbon_24`,
`perov_5` and others, 24 GB in total. Check before recomputing.

Generation from the published checkpoint (weights cached under
`~/.cache/huggingface`):

```bash
CUDA_VISIBLE_DEVICES=1 wyformer-generate out.json.gz \
    --hf-model SymmetryAdvantage/WyFormer-Alex-MP20 --device cuda
```

1000 structures took 5.25 s on a contended card.

Configs that set `compile_model: true` work — triton is installed. A pilot run
of `base_sg.yaml` with compiling forced on completed training and evaluation on
one card without a `TritonMissing` error.

Tests:

```bash
pytest                 # 594 passed, 40 skipped in 100 s
pytest -m slow         # opt-in slow diagnostics
pytest --run-relax     # needs network and a MACE model
```

---

## Weights & Biases

Credentials are in `~/.netrc`, so nothing needs to be passed and no token
belongs in the environment. The account's default entity is `kazeev`; the
visible teams are `kazeev`, `hse_lambda`, `symmetry-advantage` and `ai4x`.

Log real runs to the shared team, which is the project convention:

```bash
WANDB_ENTITY=symmetry-advantage CUDA_VISIBLE_DEVICES=1 python scripts/train.py ...
```

`WANDB_MODE=disabled` is for throwaway smoke tests only — a run started with it
is not recorded anywhere.

---

## Long-running jobs

There is no scheduler holding your job, so anything long must be detached from
the session that started it:

```bash
nohup env CUDA_VISIBLE_DEVICES=1 WANDB_ENTITY=symmetry-advantage \
    python scripts/train.py yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda \
    > train.log 2>&1 &
```

or run it inside `tmux`/`screen`. `*.log` is gitignored.

Disk is the thing to watch on multi-day runs: `/` has 889 GB free but also
holds a 279 GB HuggingFace cache belonging partly to unrelated work, plus
`cache/` (24 GB) and `runs/` (611 MB). `/mnt/hdd` has 4.5 TB free and is the
right place for bulk output.

---

## Containers

zeus is where the published Singularity images are built. That is a portable
GPU environment, unrelated to the local venv, and it is documented separately
in [`docs/singularity_images.md`](../../singularity_images.md).

Available here: `singularity-ce 4.3.7-noble`, plus a usable `docker` (the
account is in the `docker` group) and `podman`. The account also has `sudo`.
