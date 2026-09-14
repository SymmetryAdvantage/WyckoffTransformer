"""Scoring a screener: precision, enrichment, and whether the probabilities mean anything.

Mean absolute error is the wrong headline here. The decision is binary and
one-sided -- spend a structure search on this formula or don't -- and the error
that matters is concentrated in a window tens of meV/atom wide near the hull,
while the label distribution runs over several eV. A model can win on MAE and be
useless at the threshold.

So the metrics are the ones a screening campaign is judged on. **Precision** is
the share of flagged formulas that really were below the shallow hull.
**Enrichment** is precision divided by prevalence -- how many times better than
spending the same budget at random, which is the number Wren reports and the only
one that says whether the model is worth running. **Recall** says how much of the
opportunity was left on the table.

Two prediction rules are scored. The naive one flags a formula when the estimated
floor is below the hull. The uncertainty-adjusted one flags it only when the
floor *plus its epistemic standard deviation* is below the hull, which is Wren's
criterion and lifted its precision from 38% to 53%. That correction is not a
detail: ranking millions of candidates by a point estimate puts the largest
positive errors at the top of the list, so the winner's curse, not the signal,
decides what gets calculated.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def probability_below_hull(
    location: np.ndarray,
    epistemic_sigma: np.ndarray,
    hull: np.ndarray,
    floor_sigma: float = 1e-3,
) -> np.ndarray:
    """``P(f*(X) < E_hull(X))`` from the ensemble's spread over the floor.

    The floor is a parameter of the likelihood, not a draw from it, so the
    uncertainty that belongs in this probability is epistemic -- how much the
    ensemble members disagree about where the floor is -- and not the excess
    scale, which describes how far *observed structures* scatter above the floor.

    Args:
        location: Ensemble-mean estimate of the floor, eV/atom.
        epistemic_sigma: Ensemble standard deviation of the floor, eV/atom.
        hull: The hull's formation energy at that composition, eV/atom.
        floor_sigma: Lower bound on the spread, so a degenerate ensemble gives a
            sharp probability rather than a division by zero.

    Returns:
        Probabilities in [0, 1].
    """
    from scipy.stats import norm  # noqa: PLC0415

    sigma = np.maximum(np.asarray(epistemic_sigma, dtype=float), floor_sigma)
    return norm.cdf((np.asarray(hull, dtype=float) - np.asarray(location, dtype=float)) / sigma)


def screening_metrics(
    location: np.ndarray,
    hull: np.ndarray,
    discovered: np.ndarray,
    epistemic_sigma: Optional[np.ndarray] = None,
    margin: float = 0.0,
) -> Dict[str, float]:
    """Score one triage rule against the answer key.

    Args:
        location: Estimated floor, eV/atom.
        hull: Hull formation energy at the composition, eV/atom.
        discovered: Ground truth -- did the withheld search find something below
            this hull.
        epistemic_sigma: If given, the rule becomes ``location + sigma < hull``,
            Wren's uncertainty-adjusted criterion. If ``None``, the naive
            ``location < hull``.
        margin: Extra strictness, eV/atom; the flag needs ``... < hull - margin``.

    Returns:
        ``flagged``, ``precision``, ``recall``, ``prevalence``, ``enrichment``.
        Precision and enrichment are NaN when nothing was flagged, which is a
        real outcome and not a zero.
    """
    location = np.asarray(location, dtype=float)
    hull = np.asarray(hull, dtype=float)
    discovered = np.asarray(discovered, dtype=bool)
    if not (len(location) == len(hull) == len(discovered)):
        raise ValueError("location, hull and discovered must be the same length")
    if len(location) == 0:
        raise ValueError("Nothing to score")

    score = location if epistemic_sigma is None else location + np.asarray(epistemic_sigma, dtype=float)
    flagged = score < hull - margin
    n_flagged = int(flagged.sum())
    prevalence = float(discovered.mean())
    precision = float(discovered[flagged].mean()) if n_flagged else math.nan
    return {
        "flagged": n_flagged,
        "flagged_share": n_flagged / len(location),
        "precision": precision,
        "recall": float(discovered[flagged].sum() / discovered.sum()) if discovered.any() else math.nan,
        "prevalence": prevalence,
        "enrichment": precision / prevalence if n_flagged and prevalence > 0 else math.nan,
    }


def enrichment_curve(
    score: np.ndarray,
    discovered: np.ndarray,
    budgets: Sequence[int] = (100, 500, 1_000, 5_000, 10_000),
) -> pd.DataFrame:
    """Precision and enrichment among the top-``k`` formulas, for several budgets.

    This is the shape of the question a screening campaign actually asks: not
    "how many are there" but "if I can afford k structure searches, how many of
    them pay off". Lower ``score`` is better, so pass ``location - hull``.
    """
    score = np.asarray(score, dtype=float)
    discovered = np.asarray(discovered, dtype=bool)
    prevalence = float(discovered.mean())
    order = np.argsort(score, kind="stable")
    hits = np.cumsum(discovered[order])
    rows = []
    for budget in budgets:
        if budget > len(score):
            continue
        precision = hits[budget - 1] / budget
        rows.append({
            "budget": budget,
            "hits": int(hits[budget - 1]),
            "precision": float(precision),
            "enrichment": float(precision / prevalence) if prevalence > 0 else math.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted probability against observed frequency, in equal-width bins.

    A screener whose probabilities are not calibrated cannot be used to allocate a
    budget across chemistries, however good its ranking is.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    frame = pd.DataFrame({"bin": which, "p": probabilities, "y": outcomes})
    table = frame.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    table["bin_low"] = edges[table.index.to_numpy()]
    return table.reset_index(drop=True)


def expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between predicted and observed frequency, weighted by bin size."""
    table = calibration_table(probabilities, outcomes, n_bins)
    return float((table["n"] / table["n"].sum() * (table["predicted"] - table["observed"]).abs()).sum())
