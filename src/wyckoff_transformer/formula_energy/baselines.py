"""The comparators the censored model has to beat.

Three of them are the original proposal, implemented faithfully so the comparison
is a measurement rather than an argument.

``g_C`` regresses the archive minimum on the formulas whose lowest polymorph is
an experimentally observed structure. Under the assumption that observation
implies ground state its labels *are* ``f*``, so it is unbiased -- and it has
13,836 formulas, 0.63% of the corpus, none of which is the kind of formula a
screener is pointed at. ``g_D`` regresses the same quantity on everything: eighty
times the data, labels biased upward by an unknown amount. ``g_D - g_C`` is the
headroom signal the two were meant to produce together.

The fourth is Magpie descriptors under gradient boosting, which is what a
practitioner would reach for first and is a real test of whether a learned
composition encoder earns its place. The fifth is the training-free lookup that
``scripts/analyse_gene_energy_critic.py`` already reports at MAE 0.266 eV/atom
over reduced formulas: if nothing beats that, none of this was worth building.

``g_C`` and ``g_D`` share the encoder, features and split with the censored model
and differ from it only in the objective, so the comparison isolates the
likelihood.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.formula_energy import train as T
from wyckoff_transformer.formula_energy.features import N_ELEMENT_SLOTS

logger = logging.getLogger(__name__)

#: Magpie's 22 elemental properties, as matminer's ``magpie`` preset lists them.
MAGPIE_PROPERTIES: Tuple[str, ...] = (
    "Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row",
    "CovalentRadius", "Electronegativity", "NsValence", "NpValence", "NdValence",
    "NfValence", "NValence", "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled",
    "NUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom", "SpaceGroupNumber",
)
#: The six statistics matminer's preset takes over each property.
MAGPIE_STATS: Tuple[str, ...] = ("minimum", "maximum", "range", "mean", "avg_dev", "mode")


@lru_cache(maxsize=1)
def magpie_table() -> np.ndarray:
    """``[119, 22]`` of elemental property values, indexed by atomic number.

    The values come from matminer's ``MagpieData`` so they match what the rest of
    the repo uses in ``evaluation/cdvae_metrics.py``; the statistics over them are
    taken here instead of through ``ElementProperty.featurize``, which would need
    a ``pymatgen.Composition`` per formula and cannot be vectorised over two
    million of them.
    """
    from matminer.utils.data import MagpieData  # noqa: PLC0415
    from pymatgen.core.periodic_table import Element  # noqa: PLC0415

    data = MagpieData(impute_nan=False)
    table = np.full((N_ELEMENT_SLOTS, len(MAGPIE_PROPERTIES)), np.nan)
    for number in range(1, N_ELEMENT_SLOTS):
        try:
            element = Element.from_Z(number)
        except ValueError:
            continue
        for column, name in enumerate(MAGPIE_PROPERTIES):
            try:
                table[number, column] = data.get_elemental_property(element, name)
            except (KeyError, ValueError):
                pass
    return table


def magpie_features(data: T.FormulaData) -> np.ndarray:
    """``[N, 132]`` Magpie descriptors, vectorised over the padded element arrays."""
    table = magpie_table()
    element_ids = data.element_ids.cpu().numpy()
    fractions = data.fractions.cpu().numpy().astype(np.float64)
    present = ~data.padding_mask.cpu().numpy()

    values = table[element_ids]                                   # [N, L, P]
    weights = np.where(present, fractions, 0.0)[..., None]
    filled = np.where(present[..., None] & np.isfinite(values), values, np.nan)

    mean = np.nansum(np.where(np.isfinite(filled), filled, 0.0) * weights, axis=1)
    # Magpie has no values for a few elements (the noble gases among them), so a
    # composition can have no finite entry for a property at all. Filling with
    # +/-inf before the reduction rather than reducing over an all-NaN slice keeps
    # numpy quiet and lands on the same place, since nan_to_num clips the
    # infinities at the end.
    deviation = np.nansum(np.abs(filled - mean[:, None, :]) * weights, axis=1)
    minimum = np.where(np.isfinite(filled), filled, np.inf).min(axis=1)
    maximum = np.where(np.isfinite(filled), filled, -np.inf).max(axis=1)
    minimum = np.where(np.isfinite(minimum), minimum, 0.0)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    # The mode is the property of the most prevalent element, matminer's convention.
    dominant = np.argmax(np.where(present, fractions, -1.0), axis=1)
    mode = values[np.arange(len(values)), dominant]

    stacked = {"minimum": minimum, "maximum": maximum, "range": maximum - minimum,
               "mean": mean, "avg_dev": deviation, "mode": mode}
    return np.nan_to_num(np.concatenate([stacked[name] for name in MAGPIE_STATS], axis=1))


def fit_gbdt(
    features: np.ndarray,
    target: np.ndarray,
    max_iter: int = 400,
    seed: int = 0,
):
    """Gradient-boosted trees on Magpie descriptors.

    ``HistGradientBoostingRegressor`` rather than LightGBM: scikit-learn is
    already a dependency, and on tabular features of this size the two are
    interchangeable.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    model = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed,
                                          early_stopping=True, validation_fraction=0.05)
    model.fit(features, target)
    return model


def chemsys_mean_lookup(train: pd.DataFrame, test: pd.DataFrame,
                        target_column: str = "e_form_min") -> np.ndarray:
    """The training-free baseline: the train mean over the formula's chemical system.

    ``analyse_gene_energy_critic.py`` reports the reduced-formula version of this
    at MAE 0.266 eV/atom. That exact key cannot be used here -- a formula-level
    table has one row per formula, and a formula-level split puts none of the test
    keys in training -- so the back-off to the chemical system is the honest form
    of the same idea, and the one anything learned has to beat.
    """
    by_system = train.groupby("chemsys")[target_column].mean()
    return test["chemsys"].map(by_system).fillna(train[target_column].mean()).to_numpy()


def fit_neural_baseline(
    table: pd.DataFrame,
    train_split: str,
    val_split: str,
    config: T.TrainConfig,
    device: torch.device,
    subset: Optional[pd.Series] = None,
    max_elements: Optional[int] = None,
) -> Tuple[Sequence, T.FormulaData]:
    """Fit ``g_C`` or ``g_D``: the same model under MSE, on a chosen subset.

    Args:
        table: The formula table.
        train_split: Which split to fit on.
        val_split: Which split to early-stop on.
        config: Must carry ``loss="mse"``.
        device: Where to train.
        subset: Boolean mask restricting the training rows -- ``table["has_icsd"] &
            (table["icsd_excess"] <= 0)`` gives g_C's population, formulas whose
            experimentally observed structure is the archive's own minimum.
        max_elements: Shared pad width.
    """
    if config.loss != "mse":
        raise ValueError("The g_C / g_D baselines are MSE fits; set config.loss='mse'")
    rows = table["split"] == train_split
    if subset is not None:
        rows &= subset
    if not rows.any():
        raise ValueError("The subset left no training formulas")
    logger.info("fitting on %d formulas", int(rows.sum()))
    train = T.prepare(table[rows], max_elements=max_elements)
    val_rows = table["split"] == val_split
    if subset is not None:
        val_rows &= subset
    val = T.prepare(table[val_rows if val_rows.any() else rows], max_elements=max_elements)
    model, _ = T.train_one(train.to(device), val.to(device), config, device)
    return [model], train


def headroom_signal(g_d: pd.DataFrame, g_c: pd.DataFrame) -> pd.Series:
    """``g_D - g_C``: the original proposal's headroom estimate.

    Positive means the all-data fit sits above the experimentally anchored one,
    which the proposal reads as room below the archive's best. Whether the sign
    survives off the support of ``g_C`` is the question the comparison settles.
    """
    return g_d["location"] - g_c["location"]
