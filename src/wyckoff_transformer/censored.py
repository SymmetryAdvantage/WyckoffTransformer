"""Censored likelihood for regressing ``min(E | gene)``.

A Wyckoff gene fixes the space group, the species and the occupied Wyckoff
positions.  It does *not* fix the free coordinates or the cell, so it does not
determine an energy: it determines a whole manifold of structures.  For crystal
structure prediction the quantity worth predicting is the bottom of that
manifold, ``m(g) = min(E | g)`` -- the energy of the best structure the gene can
realise -- because the downstream step (PyXtal + relaxation) is a stochastic
search that can be run repeatedly.

The training labels do not contain ``m(g)``.  A dataset entry with gene ``g``
is *some* structure on the manifold, so its energy is an **upper bound**:

    E_obs >= m(g)

Fitting those labels with a mean-squared error, as a plain regressor does,
estimates ``E[E | g]`` instead, whose distance from ``m(g)`` grows with the
gene's positional degrees of freedom, and which is confounded with how often the
gene appears in the dataset.  A gene seen once has one loose upper bound; a gene
seen fifty times has fifty, and its lowest is nearly tight.  MSE treats both the
same and so learns, in part, a popularity prior wearing an energy costume.

This module implements the likelihood that treats the bound literally.

The model
---------

For a gene ``g``, a dataset structure's energy is the gene's optimum plus a
non-negative excess, observed through label noise::

    E_obs = m(g) + eps + eta,    eps ~ Exponential(rate = 1 / s(g)),
                                 eta ~ Normal(0, sigma^2)

``s(g)`` is the mean excess above the gene's optimum: how much room the gene has
to be off its own minimum, which is essentially a function of its degrees of
freedom, and so is itself predictable from the gene.  ``sigma`` is the label
noise of the reference data (DFT/MLIP disagreement, relaxation tolerance); it is
a property of the dataset, not of the gene, and it also smooths the otherwise
hard boundary at ``E = m`` into something a gradient can cross.

``E_obs`` is then exponentially modified Gaussian, with log density

    log p = log(lam) - a*t + a^2/2 + log Phi(t - a)

where ``t = (E - m) / sigma``, ``a = lam * sigma`` and ``lam = 1 / s``.  As
``sigma -> 0`` this collapses to the truncated form ``log(lam) - lam*(E - m)``
for ``E > m`` and ``-inf`` below, which is the constraint stated exactly.

Why this solves the frequency confound
--------------------------------------

Nothing here groups the data by gene: every row contributes its own term, and
the loss is a drop-in replacement for MSE.  The grouping happens implicitly and
correctly.  A gene appearing ``n`` times contributes ``n`` terms, each pushing
``m`` down by ``lam`` per unit and each blocking it from rising above that row's
energy.  The maximum-likelihood ``m`` for a gene observed in isolation is
therefore its observed minimum, and the finite-sample bias of that estimate is
analytic::

    E[min_i E_i] - m(g) = s(g) / n        (see `expected_min_bias`)

so a gene seen once is fitted to a bound that is loose by about ``s``, a gene
seen ten times to one loose by ``s / 10``, and the likelihood knows the
difference.  Genes are not fitted in isolation in practice -- the parameters are
shared, so the fitted ``m`` is the lower envelope of the whole cloud rather than
a per-gene order statistic -- but the same weighting applies.

A caveat worth stating: for a gene observed exactly once the location-scale pair
is formally degenerate (``m -> E_obs``, ``s -> 0`` sends the density to a
spike).  What prevents it is that the parameters are shared across genes, plus
the hard floor `min_scale` below.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.special import log_ndtr

#: Below this the exponential rate ``1/s`` overflows and the boundary at ``E=m``
#: becomes a hard constraint with no usable gradient.  In eV/atom.
DEFAULT_MIN_SCALE = 1e-3

#: Label noise of the reference energies, in eV/atom.  Doubles as the width over
#: which the ``E >= m`` boundary is smoothed, so it must stay well above the
#: float32 resolution of the energy scale.  10 meV/atom is the right order for
#: MLIP-relaxed reference data.
DEFAULT_NOISE = 1e-2


def censored_min_nll(
    observed: Tensor,
    location: Tensor,
    log_scale: Tensor,
    noise: float | Tensor = DEFAULT_NOISE,
    min_scale: float = DEFAULT_MIN_SCALE,
) -> Tensor:
    """Negative log likelihood of ``observed`` under the censored-minimum model.

    Every observation is read as evidence that ``location <= observed``, with
    the density of the excess supplying the opposing pressure that keeps
    ``location`` from running off to minus infinity.

    Args:
        observed: Observed energies, any shape.
        location: Predicted ``m(g)``, broadcastable to ``observed``.
        log_scale: Predicted ``log s(g)``, the log mean excess above the gene's
            optimum, broadcastable to ``observed``.  Clamped from below by
            ``log(min_scale)``.
        noise: ``sigma``, the label noise. Scalar or broadcastable tensor.
        min_scale: Floor on ``s``; see `DEFAULT_MIN_SCALE`.

    Returns:
        Elementwise NLL, the shape of the broadcast of the inputs.
    """
    sigma = torch.as_tensor(noise, dtype=observed.dtype, device=observed.device)
    if torch.any(sigma <= 0):
        raise ValueError("noise must be positive")
    log_scale = log_scale.clamp(min=math.log(min_scale))
    scale = log_scale.exp()

    # t is the observation in units of label noise, measured from the predicted
    # minimum; a is the label noise in units of the excess scale.
    t = (observed - location) / sigma
    a = sigma / scale
    # log p = -log(s) - a*t + a^2/2 + log Phi(t - a). log_ndtr is stable into the
    # far left tail, where it goes as -(t-a)^2/2 and swamps the quadratic term,
    # which is what makes a prediction above an observation expensive.
    return log_scale + a * t - 0.5 * a * a - log_ndtr(t - a)


def expected_min_bias(scale: Tensor | float, n: Tensor | int) -> Tensor:
    """How far the minimum of ``n`` observations sits above the true minimum.

    Under the exponential-excess model the order statistic ``min_i E_i`` is
    itself exponential with rate ``n / s``, so it overshoots ``m(g)`` by ``s/n``
    in expectation.  Useful for reading harvested labels: it says how much of
    the gap between two genes' observed minima is real and how much is an
    artefact of one having been sampled more often.
    """
    scale = torch.as_tensor(scale, dtype=torch.get_default_dtype())
    n = torch.as_tensor(n, dtype=torch.get_default_dtype())
    if torch.any(n < 1):
        raise ValueError("n must be at least 1")
    return scale / n


class CensoredMinLoss(nn.Module):
    """`censored_min_nll` as a criterion over a two-column model output.

    The model emits ``[location, log_scale]`` per example: predicting the scale
    as well as the location costs one extra output and lets the excess width
    track the gene's degrees of freedom, which is the whole reason it varies.
    Pass ``predict_scale=False`` to hold a single global scale instead, learned
    as a parameter of the criterion.

    Args:
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        noise: ``sigma``. Fixed rather than learned: it is weakly identified
            (it only shapes the density within a noise width of the boundary)
            and shrinking it is nearly free, so learning it tends to drive the
            boundary hard and the gradients with it.
        min_scale: Floor on ``s``.
        predict_scale: If True the criterion reads ``log s`` from the model's
            second output column; if False it uses its own parameter and the
            model emits one column.
        init_scale: Starting value of the global scale when ``predict_scale``
            is False, and the initial guess documented for the head otherwise.
    """

    def __init__(
        self,
        reduction: str = "mean",
        noise: float = DEFAULT_NOISE,
        min_scale: float = DEFAULT_MIN_SCALE,
        predict_scale: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
        if init_scale < min_scale:
            raise ValueError("init_scale must be at least min_scale")
        self.reduction = reduction
        self.noise = noise
        self.min_scale = min_scale
        self.predict_scale = predict_scale
        if predict_scale:
            self.register_parameter("global_log_scale", None)
        else:
            self.global_log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    #: Number of output columns the model must emit for this criterion.
    @property
    def n_outputs(self) -> int:
        return 2 if self.predict_scale else 1

    def split(self, prediction: Tensor) -> tuple[Tensor, Tensor]:
        """Split a raw model output into ``(location, log_scale)``."""
        if self.predict_scale:
            if prediction.dim() < 2 or prediction.size(-1) != 2:
                raise ValueError(
                    "CensoredMinLoss with predict_scale=True needs a model with "
                    f"outputs=2; got shape {tuple(prediction.shape)}")
            return prediction[..., 0], prediction[..., 1]
        location = prediction.squeeze(-1) if prediction.dim() > 1 else prediction
        return location, self.global_log_scale.expand_as(location)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        location, log_scale = self.split(prediction)
        nll = censored_min_nll(
            target, location, log_scale, noise=self.noise, min_scale=self.min_scale)
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll


class CensoredMinDiagnostics(nn.Module):
    """Interpretable companions to the NLL, which is not comparable across runs.

    The NLL moves with the fitted scale, so it cannot be read as an error the
    way an MAE can.  These three can:

    ``violation`` -- the fraction of observations that fall *below* the
    predicted minimum.  The model asserts this is impossible up to label noise,
    so it should sit near zero; a rising violation rate is the signature of a
    location head that has been pulled up towards the conditional mean.

    ``excess`` -- the mean of ``E_obs - m``, which the model claims equals
    ``s``.  Comparing it with ``scale`` below is the calibration check.

    ``scale`` -- the mean predicted ``s``, the model's own estimate of how much
    room each gene has above its optimum.  Expected to grow with degrees of
    freedom, and a useful thing to bin by dof when reading a trained model.
    """

    def __init__(self, criterion: CensoredMinLoss):
        super().__init__()
        self.criterion = criterion

    @torch.no_grad()
    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        location, log_scale = self.criterion.split(prediction)
        scale = log_scale.clamp(min=math.log(self.criterion.min_scale)).exp()
        return {
            "violation": (target < location).to(target.dtype).mean(),
            "excess": (target - location).mean(),
            "scale": scale.mean(),
        }


def censored_min_mae(prediction: Tensor, target: Tensor,
                     criterion: Optional[CensoredMinLoss] = None) -> Tensor:
    """Mean absolute error of the location head against the observed energies.

    Reported only so that a censored run stays comparable with the MSE runs
    already on the board.  It is *not* the quantity being optimised and should
    not be expected to fall below the mean excess: the location head is aiming
    at the bottom of each gene's manifold, and the labels are scattered above
    it, so a perfect ``m`` has an MAE of about ``s``.
    """
    if criterion is not None:
        location, _ = criterion.split(prediction)
    else:
        location = prediction[..., 0] if prediction.dim() > 1 else prediction
    return (location - target).abs().mean()


def gene_level_polymorph_delta(
    gene_ids,
    energies,
    composition_ids,
):
    """Build the ``Delta_E_polymorph`` conditioning label at the level of the gene.

    The obvious label -- a structure's energy minus the lowest energy among the
    polymorphs of its composition -- is noise on the conditioning channel of a
    model that reads genes.  The gene does not contain the free coordinates, so
    two dataset entries with the *same* gene and different energies present the
    model with one input and two targets, and the channel that CSP then
    conditions at zero is the one blurred by it.

    Assigning the label from ``min(E | gene)`` instead removes that entirely:
    every structure sharing a gene gets one label, and the zero of the scale
    means "the gene whose best realisation is the ground-state polymorph",
    which is the CSP target stated exactly.

    Args:
        gene_ids: Per-structure gene fingerprint. Anything hashable and equal
            for equal genes -- the augmented Wyckoff fingerprint is the right one.
        energies: Per-structure energy, in eV/atom.
        composition_ids: Per-structure composition key. Must be the *reduced*
            formula: Na2Cl2 and NaCl are the same set of polymorphs.

    Returns:
        A DataFrame indexed as the inputs are, with

        ``gene_min`` -- the observed minimum energy over structures with that
        gene. An upper bound on ``min(E | gene)``, loose by about ``s / n``
        (see `expected_min_bias`); the point of fitting the censored likelihood
        rather than regressing this column directly.

        ``delta_e_polymorph`` -- ``gene_min`` less the lowest ``gene_min`` among
        the genes of that composition. Zero for the best gene of each
        composition, and what to condition on for CSP.

        ``gene_count`` -- structures sharing the gene, and so how tight the
        bound is.

        ``polymorph_count`` -- distinct genes for the composition. One means the
        composition is unopposed and its zero says only that nothing better was
        *seen*; carry it as a second conditioning channel, or restrict training
        to compositions with more than one, rather than letting those zeros
        dilute the value CSP samples at.
    """
    import pandas as pd  # noqa: PLC0415

    frame = pd.DataFrame({
        "gene": list(gene_ids), "energy": list(energies),
        "composition": list(composition_ids)})
    if len(frame) == 0:
        raise ValueError("No structures given")
    by_gene = frame.groupby("gene", sort=False)
    frame["gene_min"] = by_gene["energy"].transform("min")
    frame["gene_count"] = by_gene["energy"].transform("size")
    # The composition's ground state is the best gene's minimum, not the best
    # structure's energy: both are estimates of the same thing, and the first is
    # the one the model can be asked to reproduce.
    frame["delta_e_polymorph"] = frame["gene_min"] - frame.groupby(
        "composition", sort=False)["gene_min"].transform("min")
    frame["polymorph_count"] = frame.groupby("composition", sort=False)["gene"].transform("nunique")
    return frame[["gene_min", "delta_e_polymorph", "gene_count", "polymorph_count"]]
