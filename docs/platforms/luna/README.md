# WyFormer on luna

`luna.rei-official.com` — Ubuntu 24.04.3, 8x NVIDIA L40S (48 GB each),
NVIDIA driver **550.163.01**, Apptainer 1.5.1.

WyFormer runs inside an Apptainer container built from an official PyTorch CUDA
image. A uv venv sits on top of that container and **reuses the container's
torch** instead of installing its own.

## Which page do you want?

| I want to... | Read |
| --- | --- |
| run training, generation or tests | [usage.md](usage.md) |
| rebuild the venv, add a dependency, bump the image | [environment.md](environment.md) |
| work out why something is broken | [troubleshooting.md](troubleshooting.md) |

---

## Quick start

Everything is already built. Example code for running a training job:

```bash
CUDA_VISIBLE_DEVICES=4 scripts/platforms/luna/run.sh python scripts/train.py \
    yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```

`run.sh` enters the container, activates the venv and runs the command from the
repository root. With no arguments it drops you into an interactive shell.

Pick a GPU explicitly — the machine is shared and **GPU 2 is faulty**. See
[usage.md](usage.md#choosing-a-gpu).

To rebuild anything, see [environment.md](environment.md).

---

## Layout

| Path | What |
| --- | --- |
| `~/containers/pytorch-2.14.0-cuda12.6` | The container (a **sandbox directory**, not a `.sif`) |
| `<repo>/.venv-luna` | The venv; usable **only inside the container** |
| `<venv>/container-constraints.txt` | CUDA versions pinned to the container's |
| `<venv>/lock-constraints.txt` | Everything else, pinned from the project `uv.lock` |
| `scripts/platforms/luna/` | The scripts described in these pages |

The scripts:

| Script | Purpose |
| --- | --- |
| `config.sh` | Shared paths and settings; sourced by the others |
| `build_image.sh` | Pulls and builds the container |
| `build_venv.sh` | Builds the venv inside the container |
| `run.sh` | Runs a command inside the container venv |
| `reuse_container_packages.py` | Makes uv treat the container's CUDA stack as installed |

---

## Verified state

Confirmed working on 2026-09-07 with the setup described above:

| Check | Result |
| --- | --- |
| `pytest` | 110 passed, 33 skipped |
| `pytest --run-relax` (what CI runs) | 141 passed, 2 skipped, 3m33s |
| torch provenance | `2.14.0+cu126` from `/usr/local/lib/python3.12/dist-packages` |
| CUDA stack installed into the venv | none — all 11 packages reused from the container |
| Locked versions honoured | matminer 0.10.1, pandas 2.3.3, scipy 1.17.1 |
| `cache_a_dataset.py mp_20` | `cache/mp_20/data.pkl.gz` |
| `tokenise_a_dataset.py mp_20` | tensors + tokeniser written |
| `train.py ... --pilot` | full 3-epoch run on one L40S, ~40 min including evaluation; writes `best_model_params.pt`, `generated_wp_no_calibration.json.gz`, `generated_wp_temperature_calibration.json.gz` |
| `wyformer-generate --hf-model` | 1000 structures in 8.3 s on one L40S; 77 space groups, 0.7% P1 |
| W&B auth from inside the container | authenticates via `~/.netrc` as `kazeev`; `symmetry-advantage` team visible |
