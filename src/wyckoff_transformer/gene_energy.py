"""Targets and inference conditions for the Wyckoff-gene energy screener.

The initial screener treats the lowest formation energy observed for a Wyckoff
gene as that gene's attainable energy.  This is deliberately simpler than the
censored model in :mod:`wyckoff_transformer.censored`: every row for a gene is
given the same observed-minimum target and the scalar head is fitted with MSE.

The minimum is calculated over the full dataset, not independently per split.
It is a target derived from the reference archive rather than an evaluation
label, and a split-local minimum would make a gene mean different things in
training and validation.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.evaluation.novelty import record_to_augmented_fingerprint

logger = logging.getLogger(__name__)

FORMATION_ENERGY_COLUMN = "formation_energy_per_atom"
GENE_MIN_FORMATION_ENERGY_COLUMN = "gene_min_formation_energy_per_atom"

_FINGERPRINT_COLUMNS = (
    "spacegroup_number",
    "elements",
    "site_symmetries",
    "sites_enumeration_augmented",
)


def _fingerprints(frame: pd.DataFrame):
    """Yield the canonical, augmentation-invariant fingerprint for each row."""
    missing = set(_FINGERPRINT_COLUMNS) - set(frame.columns)
    if missing:
        raise KeyError(
            "Cannot derive a gene target without "
            f"{sorted(missing)}; cache the full Wyckoff representation first.")
    columns = [frame[column].values for column in _FINGERPRINT_COLUMNS]
    for values in zip(*columns):
        yield record_to_augmented_fingerprint(dict(zip(_FINGERPRINT_COLUMNS, values)))


def add_observed_gene_minimum(
    frames: Mapping[str, pd.DataFrame],
    energy_column: str = FORMATION_ENERGY_COLUMN,
    target_column: str = GENE_MIN_FORMATION_ENERGY_COLUMN,
) -> Mapping[str, pd.DataFrame]:
    """Attach the observed formation-energy minimum for each Wyckoff gene.

    Equal genes are recognised through all equivalent Wyckoff enumerations, not
    just the arbitrary enumeration assigned while tokenising a particular row.
    Every split is included when finding a minimum so the target has one
    definition throughout the cache.
    """
    if not frames:
        raise ValueError("No dataset splits were supplied")

    minima: dict[Any, float] = {}
    n_rows = 0
    for split, frame in frames.items():
        if energy_column not in frame:
            raise KeyError(
                f"Split {split!r} has no {energy_column!r}; carry it from the source "
                "CSV before deriving the observed gene minimum.")
        energy = pd.to_numeric(frame[energy_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(energy).all():
            raise ValueError(
                f"Split {split!r} has non-finite {energy_column!r} values; filter them "
                "before deriving the observed gene minimum.")
        for fingerprint, value in zip(_fingerprints(frame), energy):
            previous = minima.get(fingerprint)
            if previous is None or value < previous:
                minima[fingerprint] = float(value)
        n_rows += len(frame)

    for frame in frames.values():
        frame[target_column] = np.fromiter(
            (minima[fingerprint] for fingerprint in _fingerprints(frame)),
            dtype=float,
            count=len(frame),
        )

    logger.info(
        "attached %s to %d rows from %d augmentation-invariant genes",
        target_column,
        n_rows,
        len(minima),
    )
    return frames


def build_clean_relaxation_condition(
    trainer,
    n_rows: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Build the regression condition representing a fully relaxed structure.

    ``max_force`` is observed during training because it explains how far a
    partially converged reference structure may sit above the energy its gene
    can attain.  A generated gene has no relaxation yet, so the screener asks
    the model for the zero-force limit instead.
    """
    features = tuple(getattr(trainer, "condition_features", ()) or ())
    if not features:
        return None
    if features != ("max_force",):
        raise ValueError(
            "The gene-energy screener requires exactly condition_feature: max_force; "
            f"this regressor expects {list(features)}.")
    return trainer.build_condition_from_values(
        {"max_force": 0.0},
        n_rows,
        device=device,
    )
