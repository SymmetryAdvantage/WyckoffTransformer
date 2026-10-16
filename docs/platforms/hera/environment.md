# Hera environment

Follow the [Zeus environment guide](../zeus/environment.md) for the host venv,
local wheel index, dependency management and one venv per checkout. The
following local observations and setup results were recorded on 2026-10-16.

| Item | Hera |
| --- | --- |
| OS / kernel | Ubuntu 26.04.1 LTS / `7.0.0-38-generic` (Zeus's page records Ubuntu 24.04.4) |
| RAM | 499 GiB (Zeus's page records 1,007 GiB) |
| NVIDIA driver | `615.71.09` |
| Checkout | `/home/kna/WyckoffTransformer` |
| Local wheel index | `/mnt/hdd/torch_wheels/`; includes Torch `2.14.1+cu134` for CPython 3.12 and 3.14 |
| Project environment | `uv` at `/snap/bin/uv`; Python 3.14.4 `.venv` installed with `dev relax nep`; Torch import and CUDA tensor computation pass |
| Data store config | `~/.config/wyformer/paths.env` copied from Zeus over SSH on 2026-10-16 |

## Setup and worktrees

Use Python 3.14, following Zeus and the user's explicit instruction. `uv` is
available at `/snap/bin/uv` as of the follow-up inspection on 2026-10-16. Reuse
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
