"""Narrow many cheap starts down to the usual number of expensive ones.

The wide-then-narrow variant of the ranking protocol spends its extra budget
where the budget is cheap.  Instead of drawing the schedule's one to three
PyXtal starts per gene and relaxing each on the scoring potential, it draws ten
times as many, relaxes all of them on NEP89 under a fixed-symmetry schedule,
throws away the ones that landed on the same structure, and hands the scoring
potential the *same* number of starts the schedule always allotted -- the best
survivors rather than a blind draw.

The trade is favourable because the two costs differ by orders of magnitude: a
NEP89 force call is ~1 ms where an ORB one is tens to hundreds, so ten cheap
relaxations cost a fraction of one expensive one.  What the arm buys is a better
*choice* of start, and what it risks is choosing on the wrong potential -- NEP89
ordering the basins differently from ORB.  That is the question the study in
``docs/de_novo_ranking_protocol_nep89_variants.md`` measures rather than
assumes.

Two things happen here, in this order:

**Deduplicate.**  Ten draws from one gene frequently relax into one structure,
and relaxing that structure three times on the scoring potential is three times
the cost for one answer.  Duplicates are found with ``StructureMatcher`` at
pymatgen's defaults -- the same matcher, at the same tolerances, that the
protocol's uniqueness and novelty filters use, so a pair this stage calls
distinct is a pair the funnel would also call distinct.

**Select.**  From the surviving distinct structures, the lowest NEP89 energies
up to the gene's usual trial budget.  Energy is the only ranking signal the
stage has, and it is the same signal the protocol already uses to pick a gene's
kept trial; the difference is that here it is read off the cheap potential.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: Energy gap above which two relaxed structures cannot be the same one, in
#: eV/atom, used to skip a ``StructureMatcher`` call that cannot succeed.
#:
#: Two structures relaxed to the same minimum on the same potential agree in
#: energy to the optimiser's convergence, which at ``fmax = 0.1 eV/A`` is well
#: under a meV/atom for the cells this sees.  5 meV/atom therefore leaves a
#: 5-to-50x margin, and the direction of the error is the safe one: too tight a
#: gate keeps a duplicate, which costs one extra relaxation, while too loose a
#: gate would merge two genuinely different structures and lose one.
DEDUP_ENERGY_TOL_EV_PER_ATOM = 0.005


@dataclass
class Candidate:
    """One cheaply relaxed draw, as the selection sees it.

    Attributes:
        trial: The trial index the draw was filed under.
        energy_per_atom: NEP89 (or whichever pre-relaxation potential) energy.
        structure: The relaxed geometry, as a pymatgen ``Structure``.
    """

    trial: int
    energy_per_atom: float
    structure: object


@dataclass
class Selection:
    """What the selection decided for one gene.

    Attributes:
        selected: Trials to relax on the scoring potential, best first.
        duplicate_of: Trial -> the trial it duplicates, for the ones dropped as
            duplicates.  The representative is always the lower-energy one.
        rejected: Trials that are distinct but did not make the budget.
        budget: Trials the schedule allots this gene.
        n_distinct: Distinct structures found among the candidates.
    """

    selected: list[int] = field(default_factory=list)
    duplicate_of: dict[int, int] = field(default_factory=dict)
    rejected: list[int] = field(default_factory=list)
    budget: int = 0
    n_distinct: int = 0


def select_candidates(
    candidates: Iterable[Candidate],
    budget: int,
    matcher=None,
    energy_tol: float = DEDUP_ENERGY_TOL_EV_PER_ATOM,
) -> Selection:
    """Deduplicate *candidates* and keep the *budget* lowest-energy survivors.

    Greedy in ascending energy: each candidate is compared against the
    representatives already kept, and becomes a representative itself if it
    matches none of them.  Ascending order is what makes the representative of
    a duplicate group its lowest-energy member, which is the one the scoring
    potential should start from.

    The comparison is skipped -- and the pair declared distinct -- when the two
    energies differ by more than *energy_tol*, since two relaxations of the same
    minimum on the same potential cannot.  On ten draws of one gene that turns a
    quadratic number of matcher calls into a handful.

    Args:
        candidates: The cheaply relaxed draws.  Order is irrelevant.
        budget: How many distinct structures to select.  Non-positive selects
            none, which is a schedule error rather than a valid request.
        matcher: A ``StructureMatcher``.  Built at pymatgen's defaults when
            omitted -- the same tolerances the protocol's uniqueness and novelty
            filters run at.  Pass ``False`` to deduplicate on energy alone.
        energy_tol: Energy gap, eV/atom, above which the matcher is not called.

    Returns:
        A :class:`Selection`.

    Raises:
        ValueError: If *budget* is not positive.
    """
    if budget < 1:
        raise ValueError(f"a gene needs at least one selected trial, got {budget}")

    ordered = sorted(candidates, key=lambda c: c.energy_per_atom)
    if matcher is None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher()

    selection = Selection(budget=budget)
    representatives: list[Candidate] = []
    for candidate in ordered:
        duplicate = _representative_of(candidate, representatives, matcher, energy_tol)
        if duplicate is not None:
            selection.duplicate_of[candidate.trial] = duplicate.trial
            continue
        representatives.append(candidate)

    selection.n_distinct = len(representatives)
    selection.selected = [c.trial for c in representatives[:budget]]
    selection.rejected = [c.trial for c in representatives[budget:]]
    return selection


def _representative_of(
    candidate: Candidate,
    representatives: Sequence[Candidate],
    matcher,
    energy_tol: float,
) -> Optional[Candidate]:
    """The kept candidate *candidate* duplicates, or ``None`` if it is new."""
    for representative in representatives:
        if abs(candidate.energy_per_atom - representative.energy_per_atom) > energy_tol:
            continue
        if matcher is False:
            return representative
        try:
            if matcher.fit(representative.structure, candidate.structure):
                return representative
        except Exception as exc:  # noqa: BLE001 - a matcher failure must not drop a trial
            logger.warning(
                "StructureMatcher failed on trials %s/%s (%s); keeping both",
                representative.trial, candidate.trial, exc,
            )
    return None
