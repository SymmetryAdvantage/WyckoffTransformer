"""Learning rate schedules that are indexed by step rather than by a validation metric.

`ReduceLROnPlateau`, the only schedule the other configs use, reacts to the loss: it needs no
notion of how long the run is, but it also cannot anneal to zero on a deadline. A run with a
fixed compute budget wants the opposite -- a schedule that knows its own horizon.
"""
import math
from typing import Callable

import torch


def _decay_factor(progress: float, shape: str) -> float:
    """Multiplier falling from 1 to 0 as `progress` runs 0 -> 1."""
    if shape == "linear":
        return 1.0 - progress
    if shape == "1-sqrt":
        # Hägele et al. 2024 (arXiv:2405.18392) found this to beat linear and cosine-shaped
        # decays at equal budget. It lies below linear everywhere: the drop is steepest at the
        # very start of the decay (one step in, at 1/1000 of the way, it is already at 0.97)
        # and flattens out towards zero, so the rate spends most of the decay phase small.
        return 1.0 - math.sqrt(progress)
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown decay shape {shape!r}; expected linear, 1-sqrt or cosine")


def warmup_stable_decay(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = 0.02,
    decay_fraction: float = 0.1,
    final_lr_fraction: float = 0.0,
    decay_shape: str = "linear",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup-stable-decay (WSD), stepped once per optimiser step.

    Linear warmup to the configured lr, a long constant-rate phase, then a decay over the last
    `decay_fraction` of the budget. The constant phase is what makes this a drop-in for
    schedule-free training: neither needs the horizon fixed in advance to behave sensibly
    early on, so a run can be extended by re-deriving the schedule. Unlike schedule-free, the
    good weights here are the *final* iterate rather than a running average, so nothing depends
    on that average keeping up with a drifting iterate -- the failure that cost runs lu4xqw0w
    and t1c9ehzp ~0.4 nats each over their second halves.

    The decay is where most of the loss improvement appears; it is not a cool-down formality.

    Args:
        total_steps: Optimiser steps in the whole run, not epochs.
        warmup_fraction: Fraction of `total_steps` spent ramping up from 0.
        decay_fraction: Fraction of `total_steps` spent decaying at the end.
        final_lr_fraction: Floor, as a fraction of the peak lr. 0 anneals to nothing.
        decay_shape: linear, 1-sqrt or cosine.
    """
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(f"warmup_fraction must be in [0, 1), got {warmup_fraction}")
    if not 0.0 <= decay_fraction < 1.0:
        raise ValueError(f"decay_fraction must be in [0, 1), got {decay_fraction}")
    if warmup_fraction + decay_fraction > 1.0:
        raise ValueError(
            f"warmup_fraction + decay_fraction must not exceed 1, got "
            f"{warmup_fraction} + {decay_fraction}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= final_lr_fraction <= 1.0:
        raise ValueError(f"final_lr_fraction must be in [0, 1], got {final_lr_fraction}")
    # Eagerly, not on first use: the decay phase does not begin until ~90% of the way through a
    # multi-day run, and a typo there must not be allowed to surface 65 hours in.
    _decay_factor(0.5, decay_shape)

    warmup_steps = int(round(warmup_fraction * total_steps))
    decay_steps = int(round(decay_fraction * total_steps))
    decay_start = total_steps - decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # +1 so the very first step is not taken at lr 0, which would waste it.
            return (step + 1) / (warmup_steps + 1)
        if step < decay_start:
            return 1.0
        if decay_steps == 0:
            return 1.0
        # Past the horizon the schedule holds its floor rather than going negative, so
        # overrunning the budget degrades gracefully instead of reversing the updates.
        progress = min(1.0, (step - decay_start) / decay_steps)
        return final_lr_fraction + (1.0 - final_lr_fraction) * _decay_factor(progress, decay_shape)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#: Schedules in this module that need the run's step budget injected by the trainer.
NEEDS_TOTAL_STEPS = frozenset({"warmup_stable_decay"})
