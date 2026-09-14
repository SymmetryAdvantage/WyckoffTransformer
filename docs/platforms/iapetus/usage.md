# Running WyFormer on iapetus

iapetus has no scheduler. Run every WyFormer command through the custom PyTorch
container; `.venv` is valid only inside it. Create the venv first as described
in [environment.md](environment.md).

[`scripts/platforms/iapetus/run.sh`](../../../scripts/platforms/iapetus/run.sh)
does this; the examples below abbreviate it as `run`:

```bash
alias run=scripts/platforms/iapetus/run.sh
run                     # interactive shell in the venv
run python -m pytest
```

It bind-mounts the checkout at `/workspace`, resolves the command against
`.venv/bin` (so `python` is the venv's interpreter, while torch still resolves
from the image's `/opt/venv312` through `--system-site-packages`), and mounts
`$HOME/.cache` and `$HOME/.netrc`. Those last two matter: the image builds its
own `/home/kna` rather than inheriting the host's, so without them every
invocation re-downloads the ORB checkpoint and the LeMat-Bulk hull parquet, and
anything that talks to W&B fails with `No API key configured`.

Resolving the command is not the same as setting `PATH`: the image's entrypoint
prepends `/opt/venv312/bin` to whatever `PATH` it is given, so a bare `python`
is the *image's* interpreter, which cannot see the project venv. Because torch
does live in the image, that failure surfaces far from its cause -- as
`ModuleNotFoundError: No module named 'sklearn'`. The `wyformer-*` entry points
are immune either way; their shebang names the venv interpreter outright.

The image's entrypoint runs as the owner of the mounted checkout, so files the
container writes stay host-owned.

## Common tasks

```bash
run python -m pytest      # pytest is not installed as a venv script
run python scripts/cache_a_dataset.py mp_20
run python scripts/tokenise_a_dataset.py mp_20 \
    yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
run python scripts/train.py \
    yamls/models/NextToken/v6/base_sg.yaml mp_20 cpu --pilot
```

## GPU jobs

The container's custom torch is the supported GPU build. Check occupancy, then
select a device explicitly:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv
CUDA_VISIBLE_DEVICES=0 run python scripts/train.py \
    yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```

Within the process, the selected GPU is `cuda:0`; pass `cuda`, not the original
host index, to the WyFormer command. The K20c cards have less than 5 GiB each
and the GTX 750 Ti has 2 GiB, so start with a small batch size.

For long CPU or GPU work, use `tmux` or another session manager so it survives
terminal disconnection.

## ORB on GPU

On iapetus, ORB uses a split device path: CPU-only Warp constructs neighbour
lists and the ORB model evaluates the resulting graphs on the selected GPU.
The CUDA Warp wheel cannot run on the host's driver, so this is the supported
way to use ORB GPU inference.

The CRySPR reconstruction study already implements this path. Select the GPU
as normal and run the script; its `build_patched_orb_calculator` keeps graph
construction on CPU before moving the batch to the CUDA model:

```bash
CUDA_VISIBLE_DEVICES=0 run python \
    scripts/run_cryspr_reconstruction_study.py --devices cuda:0
```

Do not use an unmodified `orb_models` ASE calculator with `device="cuda"` on
this host. It passes the CUDA device to Warp and fails because the installed
Warp build intentionally exposes CPU only. See [troubleshooting.md](troubleshooting.md)
for a minimal ORB health check.

## The de novo ranking protocol

Its stages want different hardware, and `wyformer-protocol-wandb` runs all four
in order; give each the resources it needs in one command:

```bash
run wyformer-protocol-wandb <run-id> \
    --output-dir generated/<run-id>/protocol \
    --skip-generate \
    --pyxtal-cores 6 \
    --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 --workers-per-device 1
```

- **`--pyxtal-cores 6`, the physical core count.** Measured over the same 200
  genes at 6, 10 and 12 cores, twice each: 73, 73 and 68 s of pool time, with a
  28% spread *within* one setting -- no resolvable difference. What does move
  is the summed per-task time, 286 s to 587 s for the identical 477 draws, so
  every worker past the six physical cores is absorbed by contention rather
  than turned into throughput. Six leaves the shared box usable at no measured
  cost. The draws are a long tail on a floor of nothing (median 0.21 s, p99
  14 s), so wall time is set by which slow genes a run happens to draw, and
  `--pyxtal-timeout` is the knob that actually bounds it.
- **Use all three cards, with per-card shares.** Naming a device twice gives it
  two worker slots, which is how the two K20c cards get two workers each and
  the GTX 750 Ti one; `--workers-per-device` alone cannot express that.
- **The GTX 750 Ti is worth including** even at 2 GiB and half the K20c's
  throughput on paper. ORB peaks at **296 MiB** of reserved memory on a 28-atom
  cell, and on this host Warp builds the neighbour list on the *CPU* (see
  [ORB on GPU](#orb-on-gpu)), so much of a trial is not GPU work at all: an
  idle 750 Ti relaxed the same 28-atom cell in 25-39 s against 22-40 s on a
  two-worker K20c, and took 27% of the trials as the fifth worker.
- **Check `nvidia-smi` first** and do not share a card with a training job. A
  second protocol process on the same card roughly halves throughput: five
  workers over three cards ran at 17 trials/min where four workers sharing
  cuda:0 with another job managed 9.
- A 1000-gene cohort is ~2400 trials, so about 2.5 hours. Run it under `tmux`,
  and just re-run the same command if it is interrupted -- `--resume` is the
  default and both the draws and the relaxations are logged per trial.
