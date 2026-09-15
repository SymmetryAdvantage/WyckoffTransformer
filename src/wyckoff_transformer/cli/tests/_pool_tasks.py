"""Task functions for the worker pool tests.

In a module of their own, imported by its package path, so that a spawned
worker can unpickle them however pytest imported the test module.
"""
import os
import time
from pathlib import Path

from wyckoff_transformer.cli import worker_pool

CUDA_FAILURE = (
    "AcceleratorError: CUDA error: unspecified launch failure\n"
    "Search for `cudaErrorLaunchFailure' in https://docs.nvidia.com/cuda/"
)

_DEVICE = None


def init(counter, slots, broken_at_start=()):
    """Claim a slot, as the protocol's initialiser does, minus the potential."""
    global _DEVICE
    with counter.get_lock():
        index = counter.value
        counter.value = index + 1
    _DEVICE = slots[index % len(slots)]
    worker_pool.set_worker_device(_DEVICE)
    if _DEVICE in broken_at_start:
        worker_pool.mark_worker_faulty("RuntimeError: CUDA error: no CUDA-capable device")


def _first_time(flag_dir, index) -> bool:
    flag = Path(flag_dir) / f"{index}.flag"
    if flag.exists():
        return False
    flag.touch()
    return True


def task(index, mode, flag_dir=None):
    """One trial; *mode* says how its worker misbehaves."""
    row = {"index": index, "trial": 0, "status": "ok", "device": _DEVICE, "error": None}
    if mode == "cuda_on_bad" and _DEVICE == "bad":
        row.update(status="failed", error=CUDA_FAILURE)
    elif mode == "oom_always":
        row.update(status="failed", error="OutOfMemoryError: CUDA out of memory.")
    elif mode == "raise":
        raise ValueError("the cell collapsed")
    elif mode == "crash_once" and _first_time(flag_dir, index):
        os._exit(1)
    elif mode == "crash_zero" and index == 0:
        time.sleep(0.05)  # long enough for the other workers to be busy too
        os._exit(1)
    elif mode == "hang_once" and _first_time(flag_dir, index):
        time.sleep(120)
    elif mode == "hang_always":
        time.sleep(120)
    return row, index
