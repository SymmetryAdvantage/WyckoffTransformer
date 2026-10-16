# Hera environment

Follow the [Zeus environment guide](../zeus/environment.md) for the host venv,
local wheel index, dependency management and one venv per checkout. The
following local observations were recorded on 2026-10-16, before environment
setup; they are not a record of a working WyFormer installation.

| Item | Hera |
| --- | --- |
| OS / kernel | Ubuntu 26.04.1 LTS / `7.0.0-38-generic` (Zeus's page records Ubuntu 24.04.4) |
| RAM | 499 GiB (Zeus's page records 1,007 GiB) |
| NVIDIA driver | `615.71.09` |
| Checkout | `/home/kna/WyckoffTransformer` |
| Local wheel index | `/mnt/hdd/torch_wheels/`; includes Torch `2.14.1+cu134` for CPython 3.12 and 3.14 |
| Project environment | No `.venv` or root `uv.toml` yet; `uv` is absent from the session's `PATH` |
| Data store config | `~/.config/wyformer/paths.env` has not been created |

## Setup and worktrees

Use Python 3.12 to match the Zeus recipe and its runtime-library fix. Install
`uv` before following that recipe. Reuse
[`scripts/platforms/zeus/env_init.sh`](../../../scripts/platforms/zeus/env_init.sh)
and its tracked `uv.toml`; after each invocation, select Hera's agent brief
because the shared initializer links Zeus's:

```bash
UV_PYTHON=3.12 scripts/platforms/zeus/env_init.sh
scripts/platforms/link_agent_brief.sh hera
```

The Zeus runtime prerequisites still need checking here. At inspection,
`~/.torch_profile.sh` and the NVSHMEM library at Zeus's documented path,
`/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3`, were absent.
`/opt/intel/oneapi/` and `/usr/local/cuda` exist, but their presence alone does
not establish that the local Torch wheel can load. Resolve runtime dependencies
using the [Zeus environment guide](../zeus/environment.md) before running jobs.

## Data store

Choose Hera's external storage locations and create `paths.env` following
[data_store.md](../../data_store.md#configuring-a-machine). Storage locations
are not yet configured here; Zeus's paths describe Zeus's store. Once the venv
and config exist, inspect the resolved locations from the checkout root:

```bash
.venv/bin/python -m wyckoff_transformer.paths
```
