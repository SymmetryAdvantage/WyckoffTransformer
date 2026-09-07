# Running WyFormer on luna

Everything runs through `scripts/platforms/luna/run.sh`, which enters the
container, activates the venv and runs your command from the repository root.
With no arguments it drops you into an interactive shell.

For how the environment is built, see [environment.md](environment.md). When
something breaks, see [troubleshooting.md](troubleshooting.md).

---

## Choosing a GPU

**GPU 2 is faulty.** It reports uncorrectable ECC errors and has a pending row
remap, and any CUDA allocation on it dies with:

```
torch.AcceleratorError: CUDA error: uncorrectable ECC error encountered
```

Clearing it needs a GPU reset as root (`nvidia-smi -i 2 -r`), or a reboot if the
device is in use. Until then, skip it. Check the state with:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv
```

The other seven GPUs are shared with other users and are often near capacity, so
always select a device explicitly rather than letting torch grab `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=4 scripts/platforms/luna/run.sh ...
```

Inside the container the selected device is `cuda:0`, so pass `cuda` as the
device argument to the WyFormer scripts.

`run.sh` forwards `CUDA_VISIBLE_DEVICES` only when it is actually set. This is
deliberate: `apptainer exec --env CUDA_VISIBLE_DEVICES=""` does not mean "no
preference", it means *no GPUs are visible*, and `torch.cuda.is_available()`
silently returns False. Keep that guard if you edit the script.

Device-node numbers do **not** match these indices; see
[troubleshooting.md](troubleshooting.md#devnvidian-is-not-gpu-n) before
diagnosing anything by `/dev/nvidiaN`.

`run.sh` also pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `CUDA_VISIBLE_DEVICES=N`
means `nvidia-smi`'s GPU N. CUDA's default ordering is a heuristic and is not
guaranteed to agree — see
[troubleshooting.md](troubleshooting.md#device-ordering-cuda_visible_devices-vs-nvidia-smi).

---

## Common tasks

Pilot training run, end to end:

```bash
R="scripts/platforms/luna/run.sh"
$R python scripts/cache_a_dataset.py mp_20
$R python scripts/tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
CUDA_VISIBLE_DEVICES=4 $R python scripts/train.py yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```

Generation from a pre-trained HuggingFace model (the published checkpoint is
`SymmetryAdvantage/WyFormer-Alex-MP20`; weights are cached under `~/.cache/huggingface`):

```bash
CUDA_VISIBLE_DEVICES=4 scripts/platforms/luna/run.sh \
    wyformer-generate out.json.gz --hf-model SymmetryAdvantage/WyFormer-Alex-MP20 --device cuda
```

`--initial-n-samples` must exceed `--firm-n-samples` (defaults 1100 and 1000),
since invalid structures are filtered out before subsampling. Asking for more
firm samples than you drew fails with `Not enough valid structures to subsample`.

Tests:

```bash
scripts/platforms/luna/run.sh pytest
```

---

## Weights & Biases

W&B is used throughout, and **runs should be logged to it**. Credentials are
already stored in `~/.netrc`, which is inside the bind-mounted home directory, so
they work inside the container with no extra setup and no token in the
environment. Nothing needs to be passed:

```bash
CUDA_VISIBLE_DEVICES=4 scripts/platforms/luna/run.sh python scripts/train.py \
    yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda
```

The account's default entity is `kazeev`. To log to the shared team instead
(the project's internal convention):

```bash
WANDB_ENTITY=symmetry-advantage CUDA_VISIBLE_DEVICES=4 scripts/platforms/luna/run.sh ...
```

`WANDB_MODE=disabled` exists for throwaway smoke tests only — do not use it for
real experiments, or the run is not recorded anywhere.

`run.sh` forwards `CUDA_VISIBLE_DEVICES`, `WANDB_MODE`, `WANDB_ENTITY`,
`WANDB_API_KEY` and `HF_TOKEN` from the host, each only when it is set. Because
the credentials come from `~/.netrc`, `WANDB_API_KEY` is normally unnecessary.

---

## Long-running jobs

luna has 1.5 TB of RAM and normally hundreds of GB free, but several WyFormer
steps use `pandarallel`, which fans out to **128 workers** by default (one per
physical core) and briefly allocates a lot. That spike is enough to trip
supervisors that watch system memory, and a training run tied to an interactive
session can be killed part-way through even though the run itself is healthy.

Launch anything long detached, so it does not share the fate of the session:

```bash
nohup env CUDA_VISIBLE_DEVICES=4 \
    scripts/platforms/luna/run.sh python scripts/train.py \
    yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda \
    > train.log 2>&1 &
```

or run it inside `tmux`/`screen`. If a step dies with no Python traceback,
suspect the OOM killer or a supervisor rather than the environment: check
`head -3 /proc/meminfo` and rerun when the machine is quieter.

`WyckoffProcessor.tokenise_dataset` accepts `n_jobs` if you need to cap the
worker count from your own code; the CLI scripts do not expose it.
