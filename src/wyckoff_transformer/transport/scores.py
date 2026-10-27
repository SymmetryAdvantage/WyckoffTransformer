"""Scores of a transport descriptor against conductivity labels.

The primary score is Maevskiy et al.'s: the area under the ROC curve over the lowest
10% of false-positive rate, divided by 0.1 so that a perfect ranking scores 1 and a
random one 0.05.  Labels may be soft: every sample carries a positive and a negative
weight (their smoothed labels, or 1/0 for hard ones), and enters the true-positive sum
with the first and the false-positive sum with the second.

The rediscovery score asks where a family's known fast members land among all
candidates: the percentile rank of each member, 0 for the best-ranked compound.
"""
from __future__ import annotations

import numpy as np


def weighted_roc(scores: np.ndarray, w_pos: np.ndarray, w_neg: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """``(fpr, tpr)`` with ties grouped, from (0, 0) to (1, 1), higher score = positive."""
    scores = np.asarray(scores, dtype=float)
    w_pos = np.asarray(w_pos, dtype=float)
    w_neg = np.asarray(w_neg, dtype=float)
    keep = np.isfinite(scores)
    scores, w_pos, w_neg = scores[keep], w_pos[keep], w_neg[keep]
    order = np.argsort(-scores, kind="mergesort")
    scores, w_pos, w_neg = scores[order], w_pos[order], w_neg[order]
    last_of_tie = np.r_[scores[1:] != scores[:-1], True]
    tp = np.cumsum(w_pos)[last_of_tie]
    fp = np.cumsum(w_neg)[last_of_tie]
    total_pos, total_neg = w_pos.sum(), w_neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        raise ValueError("need both positive and negative weight")
    return np.r_[0.0, fp / total_neg], np.r_[0.0, tp / total_pos]


def partial_auc(scores, w_pos, w_neg, max_fpr: float = 0.1, normalise: bool = True) -> float:
    """Area under the ROC curve for FPR in [0, max_fpr], trapezoidal; /max_fpr if normalised."""
    fpr, tpr = weighted_roc(scores, w_pos, w_neg)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x = fpr[:stop]
    y = tpr[:stop]
    if stop < len(fpr) and x[-1] < max_fpr:
        # Interpolate the curve at max_fpr.
        x0, x1, y0, y1 = fpr[stop - 1], fpr[stop], tpr[stop - 1], tpr[stop]
        y_end = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0) if x1 > x0 else y1
        x = np.r_[x, max_fpr]
        y = np.r_[y, y_end]
    area = float(np.trapezoid(y, x))
    return area / max_fpr if normalise else area


def random_partial_auc(max_fpr: float = 0.1) -> float:
    """The normalised partial AUC of a random ranking."""
    return max_fpr / 2


def bootstrap_partial_auc(scores, w_pos, w_neg, max_fpr: float = 0.1, n: int = 1000,
                          seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """``(point, low, high)``: the score and its percentile bootstrap interval."""
    scores = np.asarray(scores, dtype=float)
    w_pos = np.asarray(w_pos, dtype=float)
    w_neg = np.asarray(w_neg, dtype=float)
    point = partial_auc(scores, w_pos, w_neg, max_fpr)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(scores), len(scores))
        if w_pos[pick].sum() <= 0 or w_neg[pick].sum() <= 0:
            continue
        draws.append(partial_auc(scores[pick], w_pos[pick], w_neg[pick], max_fpr))
    alpha = (1 - level) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return point, float(low), float(high)


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Each compound's rank as a fraction of all, 0 for the best; ties share the mean rank.

    Missing scores rank last.
    """
    scores = np.asarray(scores, dtype=float)
    filled = np.where(np.isfinite(scores), scores, -np.inf)
    from scipy.stats import rankdata

    rank = rankdata(-filled, method="average") - 1
    return rank / max(len(scores) - 1, 1)


def rediscovery(percentiles: np.ndarray, family: np.ndarray, is_fast_member: np.ndarray,
                min_members: int = 5, top_fraction: float = 0.1) -> dict:
    """Median percentile of each family's fast members, and the G1 rediscovery verdict.

    Returns ``{"families": {name: {"n": .., "median_percentile": .., "in_top": bool}},
    "eligible": k, "rediscovered": m, "fraction": m / k}``.
    """
    out = {}
    family = np.asarray(family, dtype=object)
    is_fast_member = np.asarray(is_fast_member, dtype=bool)
    for name in sorted({f for f in family[is_fast_member] if f is not None}):
        members = is_fast_member & (family == name)
        if members.sum() < min_members:
            continue
        median = float(np.median(percentiles[members]))
        out[name] = {"n": int(members.sum()), "median_percentile": median,
                     "in_top": median <= top_fraction}
    eligible = len(out)
    rediscovered = sum(entry["in_top"] for entry in out.values())
    return {"families": out, "eligible": eligible, "rediscovered": rediscovered,
            "fraction": rediscovered / eligible if eligible else float("nan")}
