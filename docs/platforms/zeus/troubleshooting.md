# Troubleshooting WyFormer on zeus

Symptoms seen on this machine, and what they mean.

| Symptom | Cause |
| --- | --- |
| `TritonMissing: Cannot find a working triton installation` | triton is gone — the `compile` extra was not synced, or a bare `uv sync` pruned it. Re-run `scripts/platforms/zeus/env_init.sh`. Configs with `compile_model: true` need it. |
| `triton not found; flop counting will not work for triton kernels` on every `import torch` | Same cause, and the cheapest signal that triton is missing: it is a warning from `torch/utils/flop_counter.py`, not a failure. |
| MACE, ORB, triton, `pytest` or `ruff` suddenly missing | Something ran a bare `uv sync`, which prunes every extra. Restore with `scripts/platforms/zeus/env_init.sh`. |
| A resolver picking triton 3.7.x, or a version mismatch with torch | The `constraint-dependencies` pin is gone. The root `uv.toml` is untracked — re-copy it via `env_init.sh`. |
| `uv pip list` shows `nvidia-cublas`, `nvidia-cudnn-cu13`, ... | A PyPI manylinux torch was installed over the locally built one. The count of `nvidia-*` distributions should be 0. Reinstall torch from the local index. |
| Nothing resolves; uv cannot find torch 2.14.0+cu133 | `uv.toml` is missing, so the local flat index is not declared. It is untracked — recreate it from `uv.toml.local`. |
| `warning: Found both a uv.toml file and a [tool.uv] section ... extra-build-dependencies` | Expected on every uv invocation here, and harmless: the zeus `uv.toml` repeats the shadowed value. See [environment.md](environment.md#tooluvextra-build-dependencies-and-the-shadowing-warning). |
| `import torch` fails with a missing `libmkl_*.so.3` | The wheel finds MKL through a RUNPATH into `/opt/intel/oneapi/mkl/latest/lib/intel64`. oneAPI has moved or been removed. |
| Jobs are much slower than the core count suggests | 48 is threads, not cores; there are 24 physical cores. A CrySPR script with `NP` unset uses 48 workers. See [usage.md](usage.md#worker-counts-24-physical-cores-not-48). |
| `:: initializing oneAPI environment ...` in the middle of scripted output | `~/.profile` sources `~/.torch_profile.sh` in every login shell. Set `SETVARS_COMPLETED=1` to skip the re-init. |
| `torch.cuda.is_available()` is False while `nvidia-smi` is fine | `CUDA_VISIBLE_DEVICES` was set to the empty string somewhere. Empty means *no GPUs*, not *any GPU*. |
| A CUDA OOM on a card that `nvidia-smi` showed as free a minute ago | The machine is shared and unscheduled. Re-check occupancy and pick the other card. |
| A wheel built here fails on another machine | `-march=native` and `TORCH_CUDA_ARCH_LIST=8.9` are exported globally. Nothing built on zeus is portable — use the [Singularity images](../../singularity_images.md). |

---

## Health check

```bash
cd /home/kna/WyckoffTransformer
source .venv/bin/activate
python -c "
import torch
print('torch    ', torch.__version__, '/ cuda', torch.version.cuda)
print('file     ', torch.__file__)
print('available', torch.cuda.is_available(), torch.cuda.device_count())
print('magma    ', torch._C._has_magma)
import triton
print('triton   ', triton.__version__)
"
uv pip list | grep -c '^nvidia' ; echo '^ must be 0'
```

Expected:

```text
torch     2.14.0+cu133 / cuda 13.3
file      /home/kna/WyckoffTransformer/.venv/lib/python3.12/site-packages/torch/__init__.py
available True 2
magma     True
triton    3.8.0
0
^ must be 0
```

The wheel tag is the quickest way to tell the local build from a PyPI one:

```bash
grep Tag .venv/lib/python3.12/site-packages/torch-*.dist-info/WHEEL
# Tag: cp312-cp312-linux_x86_64   <- local build
# Tag: cp312-cp312-manylinux_...  <- came from PyPI, wrong
```

torch living inside the venv is correct here: there is no container, and
`include-system-site-packages` is `false`, so torch **must** be in the venv.
A path pointing outside it means something else is being imported.

---

## Before syncing, run it dry

`uv sync` reconciles the venv with `uv.lock` for exactly the extras you name,
and both the lock and `uv.toml` are untracked local state on this box. Go
through the script, and look before you leap:

```bash
scripts/platforms/zeus/env_init.sh --dry-run
```

Lines beginning `-` are uninstalls. A bare `uv sync` removes 42 packages here.
The script's own invocation is close to a no-op: it moves about a dozen
peripheral transitive packages (`boto3`, `cffi`, `cryptography`, `google-*`,
`lmdb`, `prettytable`, `opentelemetry-api`) back to their locked versions and
touches nothing WyFormer depends on. Pass `--inexact` for a purely additive
sync that removes nothing.

---

## Checking who has the GPUs

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -o user=,args= -p <pid>
```

Device numbering is consistent here — `nvidia-smi` index, PCI bus order and
`/dev/nvidia` minor all agree — so `/dev/nvidia1` really is GPU 1. The mapping is in [usage.md](usage.md#choosing-a-gpu); regenerate it
rather than trusting it if cards are ever reseated.
