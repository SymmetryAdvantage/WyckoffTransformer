"""Paired statistics and frozen-input inference for checkpoint comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.inference_mode()
def predict_frozen_batch(model, batch, device, augmentation_block=4):
    """Replay the exact same sampled token descriptions for every weight file."""
    samples = batch["cascade_samples"]
    n = len(batch["keys"])
    predictions = []
    for start in range(0, len(samples), augmentation_block):
        chosen = samples[start:start + augmentation_block]
        tokens = [torch.cat([draw[field] for draw in chosen], dim=0).to(device)
                  for field in range(len(chosen[0]))]
        starts = batch["start_tokens"].repeat(len(chosen), 1).to(device)
        padding = batch["padding_mask"].repeat(len(chosen), 1).to(device)
        prediction = model(starts, tokens, padding, None, cond=None)
        predictions.append(prediction.reshape(len(chosen), n).detach().float().cpu())
    return torch.cat(predictions, dim=0).numpy()


def paired_formula_bootstrap(error_early, error_final, formulas, *, repetitions=2000, seed=20261029):
    """CI for final minus earlier MAE, resampling whole reduced formulas.

    The point estimate remains a per-gene mean. A sampled formula contributes
    all its genes, preserving the dependence among polymorphs of a composition.
    Negative differences mean that the final checkpoint improves MAE.
    """
    error_early = np.asarray(error_early, dtype=float)
    error_final = np.asarray(error_final, dtype=float)
    formulas = np.asarray(formulas, dtype=str)
    valid = np.isfinite(error_early) & np.isfinite(error_final)
    difference = np.abs(error_final[valid]) - np.abs(error_early[valid])
    groups = pd.DataFrame(dict(formula=formulas[valid], delta=difference)).groupby("formula").delta.agg(["sum", "size"])
    if groups.empty:
        return dict(n=0, n_formulas=0)
    rng = np.random.default_rng(seed)
    totals, sizes = groups["sum"].to_numpy(), groups["size"].to_numpy()
    values = []
    # Chunk the bootstrap so a large formula vocabulary does not require a
    # repetitions × vocabulary matrix in memory at once.
    for start in range(0, repetitions, 64):
        indices = rng.integers(0, len(groups), size=(min(64, repetitions - start), len(groups)))
        values.extend(totals[indices].sum(axis=1) / sizes[indices].sum(axis=1))
    lower, upper = np.quantile(values, [.025, .975])
    return dict(n=len(difference), n_formulas=len(groups), final_minus_early_mae=float(difference.mean()),
                ci95_lower=float(lower), ci95_upper=float(upper), repetitions=repetitions)
