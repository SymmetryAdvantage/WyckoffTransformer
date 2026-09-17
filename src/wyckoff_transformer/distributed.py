"""Training on several GPUs of one node with DistributedDataParallel.

One process per GPU, launched by torchrun, which sets RANK, LOCAL_RANK and WORLD_SIZE.
Without those variables -- a plain `python scripts/train.py` -- everything here reduces to
the single-process context and nothing changes. See docs/distributed_training.md.

What the ranks must agree on, and how they do:

- **The model.** DDP broadcasts rank 0's parameters when it wraps the model and averages
  the gradients in every backward pass.
- **The shape of each step.** WyFormer draws `known_seq_len` and the prediction head per
  step, and a head no rank uses receives no gradient. Every rank draws them from the same
  `random.Random(shared_seed)`, so all ranks take the same step and DDP averages
  gradients of the same objective.
- **Which examples are in the global batch.** The training loader draws the global batch
  from a CPU generator seeded identically on every rank and each rank keeps its own
  shard, so the union over ranks is the batch one process would have drawn.
- **Everything decided from validation.** Evaluation runs on every rank and its result is
  averaged, so early stopping and ReduceLROnPlateau see one number.

What must *differ* between ranks is the per-example randomness -- the order permutations
and the augmentation draw -- or the ranks would duplicate each other's work. The global
torch RNG is therefore seeded per rank.
"""
from dataclasses import dataclass
import inspect
import logging
import os
import random
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    """Where this process sits among the training processes.

    The default instance is the single-process case, which every code path treats as "no
    process group": no collectives, no sharding, every file written.
    """
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    #: Drawn by rank 0 and broadcast. Seeds the streams every rank must share. None for a
    #: single process, which keeps using the global `random` and torch RNGs as it always has.
    shared_seed: Optional[int] = None
    backend: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Whether this process writes files and logs to W&B."""
        return self.rank == 0

    def rank_seed(self, stream: int = 0) -> int:
        """A seed distinct per rank and per `stream`, derived from the shared seed."""
        if self.shared_seed is None:
            raise ValueError("A single-process context has no shared seed")
        return (self.shared_seed + 1_000_003 * (stream + 1) + self.rank) % 2**63

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """The mean of `tensor` over ranks, as a new tensor; `tensor` itself when single.

        SUM and a division rather than ReduceOp.AVG, which gloo does not implement.
        """
        if not self.enabled:
            return tensor
        reduced = tensor.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced / self.world_size

    def broadcast_object(self, obj: Any) -> Any:
        """Rank 0's `obj` on every rank."""
        if not self.enabled:
            return obj
        box = [obj if self.is_main else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def all_gather_object(self, obj: Any) -> List[Any]:
        """Every rank's `obj`, indexed by rank."""
        if not self.enabled:
            return [obj]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


SINGLE_PROCESS = DistributedContext()


def world_size_from_env() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_distributed(
    device: torch.device,
    backend: Optional[str] = None,
) -> Tuple[DistributedContext, torch.device]:
    """Join the process group torchrun set up, and pick this rank's device.

    Args:
        device: The device given on the command line. Under torchrun it names the device
            *type*: `cuda` becomes `cuda:LOCAL_RANK`, so the ranks of one node take one card
            each, in `CUDA_VISIBLE_DEVICES` order.
        backend: The collective backend; NCCL for CUDA and gloo for the CPU by default.

    Returns:
        The context, and the device this rank trains on. Outside torchrun, the
        single-process context and `device` unchanged.
    """
    world_size = world_size_from_env()
    if world_size == 1:
        return SINGLE_PROCESS, device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if device.type == "cuda":
        if device.index is not None:
            raise ValueError(
                f"Under torchrun pass `cuda`, not {device}: each rank takes "
                "cuda:LOCAL_RANK. Select the cards with CUDA_VISIBLE_DEVICES.")
        if torch.cuda.device_count() < local_world_size:
            raise RuntimeError(
                f"{local_world_size} ranks on this node, but only "
                f"{torch.cuda.device_count()} CUDA devices are visible.")
        device = torch.device("cuda", local_rank)
        # Before the process group: NCCL and every CUDA allocation that follows default to
        # the current device, and leaving it at cuda:0 puts a context of every rank there.
        torch.cuda.set_device(device)
        backend = backend or "nccl"
    elif device.type == "cpu":
        backend = backend or "gloo"
    else:
        raise ValueError(f"Distributed training supports cuda and cpu, not {device.type}")
    bind_device = {}
    # Binding the rank to its card makes NCCL set up eagerly and on the right device; the
    # argument appeared in torch 2.3, and older builds on other platforms lack it.
    if (device.type == "cuda"
            and "device_id" in inspect.signature(dist.init_process_group).parameters):
        bind_device = {"device_id": device}
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **bind_device)
    context = DistributedContext(
        rank=rank, local_rank=local_rank, world_size=world_size, backend=backend)
    shared_seed = context.broadcast_object(random.SystemRandom().getrandbits(62))
    context = DistributedContext(
        rank=rank, local_rank=local_rank, world_size=world_size, backend=backend,
        shared_seed=shared_seed)
    # Per rank, so the permutations and augmentation draws of the ranks are independent.
    # Model initialisation differing between ranks is harmless: DDP broadcasts rank 0's.
    torch.manual_seed(context.rank_seed())
    logger.info("Rank %d of %d on %s, backend %s", rank, world_size, device, backend)
    return context, device


def shutdown_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()
