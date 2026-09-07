# Troubleshooting WyFormer on luna

Symptoms actually seen while building this environment, and what they mean.

| Symptom | Cause |
| --- | --- |
| `torch.cuda.is_available()` is False, but `nvidia-smi` looks fine | `CUDA_VISIBLE_DEVICES` was forwarded as an empty string. Empty means *no GPUs*, not *any GPU*. |
| `CUDA error: uncorrectable ECC error encountered` | You are on GPU 2. It is faulty; use another. |
| `mksquashfs ... FATAL ERROR: Bug in orderer` during `apptainer pull` | Bundled mksquashfs 4.7.5 cannot pack an image this large. Build a sandbox instead — `build_image.sh` already does. |
| `Read-only file system (os error 30)` under `.venv-luna/.../*.dist-info` | A `.dist-info` was symlinked into the container instead of copied, and uv tried to write to it. `reuse_container_packages.py` copies for this reason. |
| `ImportError: cannot import name 'sph_harm' from 'scipy.special'` | matminer was resolved to 0.8.0 because the `uv.lock` constraints were not applied. Rebuild with `build_venv.sh`. |
| `uv pip list` shows `nvidia-cublas 13.x` / `nvidia-cudnn-cu13` | A CUDA 13 torch was installed into the venv. It cannot run on this driver. Rebuild with `build_venv.sh`. |
| A job pinned to GPU N appears to hold `/dev/nvidia<other>` | Not a bug. Device-node minors do not match `nvidia-smi` indices — see [`/dev/nvidiaN` is not GPU N](#devnvidian-is-not-gpu-n). |

To check quickly that the environment is intact:

```bash
scripts/platforms/luna/run.sh python -c "
import torch
print(torch.__version__, torch.__file__, torch.cuda.is_available())"
```

`torch.__file__` must be under `/usr/local/lib/python3.12/dist-packages`, i.e.
the container's copy. If it points into `.venv-luna`, the venv has acquired its
own torch and should be rebuilt.

---

## `/dev/nvidiaN` is not GPU N

Device-node minor numbers are the driver's own enumeration and, on luna, differ
from `nvidia-smi` indices for **every** GPU. `nvidia-smi` sorts by PCI bus; the
minors do not:

| `nvidia-smi` index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCI bus | 01 | 21 | 41 | 61 | 81 | a1 | c1 | e1 |
| `/dev/nvidia` minor | 2 | 3 | **1** | 0 | **6** | 7 | 5 | 4 |

So a job pinned to GPU 4 holds `/dev/nvidia6`, and the faulty GPU 2 is
`/dev/nvidia1`. Do not diagnose by device node. To check which GPU a process is
really on, resolve through the bus id:

```bash
# what the job was asked to use
tr '\0' '\n' < /proc/<pid>/environ | grep CUDA_VISIBLE_DEVICES

# which device node it actually holds -> minor number
ls -l /proc/<pid>/fd | grep -o 'nvidia[0-9]*' | sort -u

# minor -> PCI bus id
for i in /proc/driver/nvidia/gpus/*/information; do
    echo "minor=$(awk '/Device Minor/{print $NF}' "$i")  bus=$(basename "$(dirname "$i")")"
done | sort -V

# PCI bus id -> nvidia-smi index
nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader
```

Note that `ps`, `pgrep` and even `nvidia-smi` can stall for minutes when the
machine is loaded, whereas the `/proc` reads above stay responsive. Prefer them
when the box is busy.

The table above will change if cards are reseated or the driver reloads;
regenerate it rather than trusting it blindly.

---

## Device ordering: `CUDA_VISIBLE_DEVICES` vs `nvidia-smi`

`nvidia-smi` always enumerates by PCI bus id. CUDA does **not**: its default
`CUDA_DEVICE_ORDER` is `FASTEST_FIRST`, a heuristic, so in general
`CUDA_VISIBLE_DEVICES=N` need not mean `nvidia-smi`'s GPU N.

On luna the two orderings do coincide today — all eight cards are identical
L40S, so the heuristic has nothing to reorder. Verified by matching GPU UUIDs
from both tools:

```bash
scripts/platforms/luna/run.sh python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i).uuid)"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
```

`run.sh` nonetheless pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` (overridable from the
host), so the mapping stays true to `nvidia-smi` even if a card is ever replaced
with a different model. Do not remove that without re-verifying the UUIDs.
