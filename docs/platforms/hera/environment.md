# Hera environment

Follow the [Zeus environment guide](../zeus/environment.md) for the host venv,
local wheel index, dependency management and one venv per checkout. The
following local observations and setup results were recorded on 2026-10-16.

| Item | Hera |
| --- | --- |
| OS / kernel | Ubuntu 26.04.1 LTS / `7.0.0-38-generic` (Zeus's page records Ubuntu 24.04.4) |
| RAM | 499 GiB (Zeus's page records 1,007 GiB) |
| NVIDIA driver | `615.71.10` (since the reboot on 2026-10-22) |
| Checkout | `/home/kna/WyckoffTransformer` |
| Local wheel index | `/mnt/hdd/torch_wheels/`; includes Torch `2.14.1+cu134` for CPython 3.12 and 3.14 |
| Project environment | `uv` 0.13.5 at `~/.local/bin/uv` (since 2026-10-23); Python 3.14.4 `.venv` installed with `dev relax nep`; Torch import and CUDA tensor computation pass |
| Data store config | `~/.config/wyformer/paths.env` copied from Zeus over SSH on 2026-10-16 |

## Setup and worktrees

Use Python 3.14, following Zeus and the user's explicit instruction. `uv` is
at `~/.local/bin/uv` (0.13.5, reinstalled on 2026-10-23 after `snapd` was purged). Reuse
[`scripts/platforms/zeus/env_init.sh`](../../../scripts/platforms/zeus/env_init.sh)
and its tracked `uv.toml`; after each invocation, select Hera's agent brief
because the shared initializer links Zeus's:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 WYFORMER_EXTRAS="dev relax nep" scripts/platforms/zeus/env_init.sh
scripts/platforms/link_agent_brief.sh hera
```

The initializer explicitly selects Python 3.14; `UV_PYTHON=3.12` no longer
overrides it. The CMake policy setting is part of Zeus's documented source-build
recipe. The host needs the `cmake` executable before this recipe can succeed.

## Runtime and pilot

On 2026-10-16 at commit `5d0e62313277b8840e7e20fd45c83c50445fac1a`,
Torch 2.14.1+cu134, Triton 3.8.0 and Python 3.14.4 loaded successfully.
CUDA 13.4 was available and a tensor calculation passed on an RTX 6000 Ada.
The user installed the missing cuDNN 9, NCCL, cuSPARSELt and cuDSS host libraries.
The runtime library names needed by this wheel include `libcudnn.so.9`,
`libnccl.so.2`, `libcusparseLt.so.0`, `libcudss.so.0` and `libnvshmem_host.so.3`.

The three-epoch MP20 DER pilot completed on GPU 0 with online W&B logging:
[5mu98ebn](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/5mu98ebn).
The process exited 0 after training, uncalibrated generation, temperature
calibration and calibrated generation evaluation. Final validation loss was
61.0268. W&B confirmed syncing 29 artifact files, including model/configuration
and generated outputs. This short pilot validates the environment and execution
path; three epochs do not establish model quality.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online OMP_NUM_THREADS=8 .venv/bin/python -u scripts/train.py \
    yamls/models/mp_20/der/uncond_adamw_wsd.yaml mp_20 cuda --pilot --torch-num-thread 8
```

## Setup prerequisites and earlier failures

The first setup attempts on 2026-10-16 exposed missing host prerequisites.
The user installed CMake, Python development headers (`python3-dev`, supplying
`/usr/include/python3.14/Python.h`), NVSHMEM, cuDNN 9, NCCL, cuSPARSELt and
cuDSS. Source builds of `dm-tree==0.1.9` and `calorine==4.0` then succeeded.
Use the documented initializer to install `dev relax nep`; the deprecated
`compile` extra no longer exists and Torch requires Triton itself.

The initial Python 3.12 Torch wheel required `libmpi_cxx.so.40` from Zeus's
Open MPI installation. Installing MPICH did not supply that library. Hera
subsequently adopted Zeus's Python 3.14 recipe and the refreshed CPython 3.14
Torch wheel, which loads successfully. The Python 3.12 environment was removed
at the user's request. Keep future setup on Python 3.14.

The setup followed the user's instruction to report missing dependencies
without workarounds; each missing host prerequisite was installed by the user.

## Data store

The user requested Zeus's storage layout. Its config was copied with `scp`:
data, cache and runs are under `/home/kna/.local/share/wyformer/`, and
`WANDB_DIR` is `/home/kna/.local/share/wyformer`. These are local Hera paths.
The full Zeus cache (91 GiB, including dataset tables, tensors and tokenisers)
was copied successfully on 2026-10-16 using
`scripts/store_sync.sh zeus pull --go cache` (exit 0). Raw datasets and run
directories were not part of this cache transfer.
Inspect the resolved locations from the checkout root:

```bash
.venv/bin/python -m wyckoff_transformer.paths
```

## Formula CSP campaign (2026-10-16)

The campaign at commit `b25687e9a1512891ef7192dd5e6a5f48aaafaa7c`, with
the new configurations and launcher archived in its W&B selection artifact, is
[csp-formula-hera-20261016](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016).
Its manifest is `yamls/campaigns/csp_formula_hera_20261016.yaml`. It queues three
independent MP-20 models, then three independent LeMat-Bulk-fixed models:
formula only, formula plus row formation energy, and the latter with CFG
(`condition_dropout: 0.1`). Both GPUs participate in each run through DDP.
All six finished by 2026-10-20. Their best model artifacts and resume checkpoints
were verified in W&B. CSP reconstruction accuracy has not yet been measured.

The recipes were selected from 328 W&B generator runs. None of those runs used
formula conditioning, so this campaign adapts compatible recipes rather than
reproducing a prior CSP result. The manifest records the source runs, their
commits and the changes. All models use embeddings 16/16/8, three encoder
layers, a 128-wide feed-forward layer, AdamW at peak learning rate 0.003 and
warmup-stable-decay. MP-20 retains dropout 0.2; LeMat uses 0. The global training
batches are 27,136 for MP-20 and 50,000 for LeMat. The MP-20 horizon is one million
epochs (one optimiser step each); LeMat runs for 20,000 epochs with multiple
steps per epoch.

All six use `composition_conditioning: true`,
`condition_on_cell_size: false`, `predict_start: true`, and corrected paired
Wyckoff augmentation. The energy models use `formation_energy_per_atom` with
an identity transform. MP-20 and LeMat retain their separate manifest energy
definitions. The formula-only LeMat model uses
`lemat_bulk_fixed_fmax1_stress_ehull01`; the energy models use its unsliced
parent. Train/validation/test splits stay separate. Each model starts from
random weights.

MP-20 formation energies include MaterialsProject2020Compatibility corrections;
LeMat uses its own uncorrected PBE reference. Those definitions are recorded in
each model's `field_provenance.json`. CFG drops the entire formula-plus-energy
conditioning vector on 10% of training examples and adds a null indicator.

`scripts/platforms/hera/train_csp_campaign.py` owns the sequential queue and
resumes failed jobs from local or W&B checkpoints. It refuses to restart an
existing run without accessible resume state. Its W&B summary records each
job's state, and its files include the training logs. Start it from this
checkout with `.venv/bin/python`, supplying `--evidence` with the W&B run
snapshot archived as `wandb_runs.json` in the campaign's selection artifact.
The launcher takes a file lock under the resolved runs store to prevent two
copies from owning the queue.

On 2026-10-18, all three MP-20 runs had completed their one-million-epoch
horizons at the commit above. Their best validation NLLs, as recorded in W&B,
were:

| MP-20 variant | W&B run | Best validation NLL | Final validation NLL | Best epoch | Artifact version |
| --- | --- | ---: | ---: | ---: | --- |
| Formula only | [mp20_unconditional](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-mp20_unconditional) | 13.541571 | 21.421669 | 8000 | v6 |
| Formula + formation energy | [mp20_formation_energy](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-mp20_formation_energy) | 13.377284 | 24.592396 | 12000 | v8 |
| Formula + formation energy, CFG | [mp20_formation_energy_cfg](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-mp20_formation_energy_cfg) | 13.114642 | 22.639149 | 767000 | v7 |

The final MP-20 validation losses are substantially worse than the selected
losses. The formula-only and formation-energy models reached their best
checkpoints early; CFG's best checkpoint came much later. Use the selected
best weights for inference. `last_checkpoint.pt` contains the final resume state
and does not identify the best validation weights.

The first LeMat attempt failed before training: its configured 50,000-row
validation batch exceeded the ehull01 split's 31,099 rows. The test split has
31,327 rows. On 2026-10-18 those two batch sizes were corrected, the failed
run and directory were preserved, and the campaign restarted that job as
[lemat_fixed_unconditional-r1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-lemat_fixed_unconditional-r1).
The manifest records the replacement run explicitly; the launcher now checks
every configured batch against its cached split length before starting the
queue. The repair and updated configurations are archived in the campaign's
next selection-artifact version.

On 2026-10-20 the campaign and all six successful training runs finished.
Each LeMat run completed 20,000 epochs at the commit above, with the archived
configurations and the evaluation-batch repair described here. The final LeMat
best validation NLLs were:

| LeMat variant | W&B run | Best validation NLL | Best epoch | Artifact version |
| --- | --- | ---: | ---: | --- |
| Formula only, ehull01 | [lemat_fixed_unconditional-r1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-lemat_fixed_unconditional-r1) | 8.729337 | 19999 | v26 |
| Formula + formation energy | [lemat_fixed_formation_energy](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-lemat_fixed_formation_energy) | 7.300169 | 19999 | v24 |
| Formula + formation energy, CFG | [lemat_fixed_formation_energy_cfg](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-lemat_fixed_formation_energy_cfg) | 7.378689 | 19999 | v22 |

Epochs in both tables use the zero-based W&B labels. Each selected checkpoint
is `symmetry-advantage/WyckoffTransformer/best_model_<run-id>:<version>`;
the run links above give the full run ids. All three LeMat models achieved
their best validation loss at the final epoch. The formula-only model and
the energy models use different dataset populations, so their NLL difference
does not isolate the effect of energy conditioning.

Training completion does not establish CSP reconstruction accuracy; that
evaluation has not been run. The finished campaign's summary was reconciled
against the child runs because some job labels retained their pre-restart
states. The original failed LeMat attempt remains recorded separately.

The caches copied from Zeus already contain the row formation energies.
`scripts/prepare_formation_energy_token_cache.py <dataset>` builds the sibling
`der_tokenizer_v1_formation_energy` tensors without changing the parent cache
or token ids. It checks row alignment against space groups, hull distances and
gene-minimum energies, records `provenance(...)`, and refuses overwrites.
Both MP-20 and the unsliced LeMat cache were prepared on 2026-10-16; every parent
tensor was checked for exact preservation. The MP-20 tokeniser/tensor
directories arrived without owner write permission; it was enabled to add the
sibling and restored afterward.

## NVIDIA driver upgrade (2026-10-22)

At 12:54 on 2026-10-22 the NVIDIA userspace was upgraded to 615.71.10 while the
loaded kernel module stayed at 615.71.09. `nvidia-smi` and NVML then failed with
`Failed to initialize NVML: Driver/library version mismatch`, although Torch CUDA
still ran a matmul on both cards. The user rebooted hera; afterwards the module and
userspace both reported 615.71.10 and `nvidia-smi` worked. After an unattended driver
upgrade, reboot before starting a long GPU job rather than relying on the mismatched
state.

## cuDSS 1.0 upgrade (2026-10-23)

At 12:17 UTC on 2026-10-23, `apt install cudss-cuda-13` upgraded cuDSS from 0.8.0.10 to
1.0.0.35 and removed `libcudss0-cuda-13`, so `libcudss.so.0` was gone. The Torch
2.14.1+cu134 wheel's `libtorch_cuda.so` needs that soname. From then on, every new Python
process failed at `import torch` with `ImportError: libcudss.so.0: cannot open shared
object file`. Processes already running kept the library they had mapped. The
solid-electrolyte campaign lost one stage this way (phosphide `score`).

In the worktree venv `/home/kna/WyckoffTransformer-se/.venv`, the old library now ships
with Torch. `apt download libcudss0-cuda-13=0.8.0.10-1` fetched the package from the
host's local repo (`/var/cudss-local-repo-ubuntu2404-0.8.0`) without installing it.
`dpkg-deb -x` then extracted `libcudss.so.0.8.0`, which was copied with its
`libcudss.so.0` link into `site-packages/torch/lib/`. That directory is on the library's
`$ORIGIN` RUNPATH, so no environment variable or restart is needed.

The main checkout's venv, `/home/kna/WyckoffTransformer/.venv`, still fails to import
Torch. Fix it the same way, reinstall `libcudss0-cuda-13` next to 1.0 (if the packages
coexist), or rebuild Torch against cuDSS 1.0.

At 12:57 the user rebuilt the wheel against cuDSS 1.0:
`/mnt/hdd/torch_wheels/torch-2.14.1+cu134-cp314-cp314-linux_x86_64.whl`.
Its sha256 is `d003b007…9ff2`. It comes from the same source as the installed wheel, tag
v2.14.1 at commit `5c48869`. Its `libtorch_cuda.so` needs `libcudss.so.1`, which
resolves to the system cuDSS 1.0, and no library is missing.

It was checked unpacked on `PYTHONPATH`, ahead of the worktree venv:
- Matmuls ran on both GPUs.
- A CUDA `torch.sparse.spsolve`, which goes through cuDSS, gave the same 4e-16 residual
  as the old build.
- `torch.compile` worked.
- ORB `orb_conserv_inf` was run on six relaxed campaign structures (24–288 atoms). Its
  energies and forces differed from the old build by at most 4e-4 eV and 9e-4 eV/Å. Two
  runs of the old build differ from each other by up to 1e-3 eV and 8e-4 eV/Å, so the
  new build is equivalent within float32 GPU noise.

The same session purged `snapd`, which also removed `/snap/bin/uv`. The user reinstalled
it as `~/.local/bin/uv` (0.13.5).

The rebuilt wheel then went into the main checkout's venv. `uv.lock` records the local
wheel by path, with no hash, and its version is unchanged, so `uv sync` would keep the
old build. Replace it explicitly:

```bash
uv pip install --python .venv/bin/python --no-deps --reinstall-package torch \
    /mnt/hdd/torch_wheels/torch-2.14.1+cu134-cp314-cp314-linux_x86_64.whl
```

The `libnvshmem_host.so.3` link survived the reinstall. Afterwards:
- The smoke test and ORB numbers above matched again.
- `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest src/wyckoff_transformer` gave
  1592 passed and 53 skipped.

The worktree venv keeps the old build and its copied `libcudss.so.0` until the
solid-electrolyte relax run ends. Replacing Torch under a driver that keeps starting new
stages could let one of them import a half-installed package.

## Solid-electrolyte campaign (2026-10-22)

[`se-families-hera-20261022`](../../solid_electrolyte_campaign.md) runs from the
worktree `/home/kna/WyckoffTransformer-se` (branch `se-families-campaign`, its own
venv). It needed two inputs hera did not have:

- `data/lemat-bulk/lemat_pbe.csv.gz` (1.06 GB), the geometry the protocol's `score`
  stage matches novelty against, copied from zeus. `scripts/store_sync.sh` syncs
  directories only: given this file as a subpath, rsync failed with `Not a directory`
  and left an empty directory of that name behind. Copy a single file with plain
  `rsync -a zeus:<store>/data/lemat-bulk/lemat_pbe.csv.gz <store>/data/lemat-bulk/`.
- `data/obelix/all.csv`, OBELiX at commit `4eaac88`, downloaded from GitHub.

`mp_2026_gga_gap` was re-tokenised with `der_tokenizer_v1` in 54 s; the
gene-level band-gap regressor then trained at about 82 steps/s on one RTX 6000 Ada.

### SSH from hera to iapetus

hera reached zeus and iapetus only through an agent forwarded by the user's login, so
an ORB lane on iapetus died whenever that session ended (2026-10-23). hera now has its
own key, `~/.ssh/id_ed25519` (`kna@hera`, SHA256:s818UQKx…), authorised on iapetus and
zeus on 2026-10-23. `ssh-copy-id` copies the keys of the machine it runs on, so it has to
be run *on hera*; run from a laptop it authorises the laptop. Long jobs that ssh to
another host should run with `SSH_AUTH_SOCK=` so they never depend on a session's agent.

### Hugging Face token

The `HF_TOKEN` in hera's `.env` (`hera-wyformer`) is a fine-grained token with only
`repo.content.read` on the user's namespace. It downloads models but cannot create or push
to `SymmetryAdvantage` repos (`403 Forbidden` from `api/repos/create`, 2026-10-23).
Push from a machine with a write token, or give hera one.
