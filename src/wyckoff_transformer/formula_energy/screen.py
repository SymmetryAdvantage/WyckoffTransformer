"""Screening: rank formulas, and exclude the ones that would falsify the record.

Two signals, pointing opposite ways.

The model gives an **upper**-bound-flavoured estimate: where the floor probably
is, with an epistemic spread, which turns into ``P(f*(X) < E_hull(X))`` and a
triage rule. Everything else in this package produces that.

This module adds the only **lower** bound available. If a compound at composition
X really had energy ``e``, the convex hull would move, and some compound that
people have actually made might be pushed above it. Compounds that exist tend not
to be unstable, so an ``e`` that would displace one is evidence against itself.
Formally, for each protected entry ``Z`` the displacement condition is affine in
``e`` -- adding a point at ``(x_X, e)`` changes the hull at ``x_Z`` to a minimum
over simplices contributing ``lambda*e + sum(mu_i E_i)`` -- so there is a critical
energy below which ``Z`` comes off the hull, and ``L(X)`` is the largest such
critical energy over the protected set.

Which simplex binds depends on ``e``, so rather than enumerate them this is found
by bisection on a predicate that is monotone in ``e``: lowering a point can only
lower the hull, so once ``Z`` is displaced it stays displaced. Thirty phase
diagrams per candidate, which is affordable on a shortlist and is not affordable
on the 2.2M-formula corpus -- as intended. Only 3.7% of formulas define the hull
at all and only an eighth of those have an experimentally observed structure, so
this is a post-filter on a ranked list, never a primary screen.
"""
from __future__ import annotations

import logging
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: How far below the hull the bisection is willing to look, eV/atom. Formation
#: energies live inside a few eV and the reference hull is built from the same
#: scale, so a candidate this far down is out of the physical range anyway.
DEEPEST = -10.0


def _phase_diagram(entries):
    from pymatgen.analysis.phase_diagram import PhaseDiagram  # noqa: PLC0415

    return PhaseDiagram(entries)


def entries_for_chemsys(reference: pd.DataFrame, elements: FrozenSet[str]) -> List:
    """Every reference entry whose elements fit inside ``elements``.

    A phase diagram over a chemical system needs its subsystems -- the binaries
    and the elemental references -- not just the entries at the same arity, which
    is what the set-containment test gives.
    """
    from pymatgen.analysis.phase_diagram import PDEntry  # noqa: PLC0415

    inside = reference["chemsys"].map(lambda text: set(str(text).split("-")) <= elements)
    return [
        PDEntry(row.full_formula, row.energy_corrected, str(row.Index))
        for row in reference[inside.to_numpy()].itertuples()
    ]


class HullLookup:
    """Hull formation energy at arbitrary compositions, with the per-system work cached.

    :func:`hull_energy_per_atom` rescans the whole reference for every call, which
    is fine for a handful of candidates and hopeless for a thousand. This does the
    element containment test once as a matrix operation -- the same device
    ``evaluation/hull_energy.py`` uses -- and keeps one phase diagram per chemical
    system, so a batch of candidates pays for each system once however many
    compositions fall inside it.
    """

    #: Hydrogen through oganesson, indexed by ``Z - 1``.
    N_ELEMENTS = 118

    def __init__(self, reference: pd.DataFrame) -> None:
        from pymatgen.core.periodic_table import Element  # noqa: PLC0415

        self.reference = reference
        rows = np.zeros((len(reference), self.N_ELEMENTS), dtype=bool)
        for position, chemsys in enumerate(reference["chemsys"].astype(str)):
            for symbol in chemsys.split("-"):
                try:
                    rows[position, Element(symbol).Z - 1] = True
                except (ValueError, KeyError):
                    pass
        self._matrix = rows
        self._diagrams: Dict[FrozenSet[str], object] = {}

    def _mask(self, elements: FrozenSet[str]) -> np.ndarray:
        from pymatgen.core.periodic_table import Element  # noqa: PLC0415

        wanted = np.zeros(self.N_ELEMENTS, dtype=bool)
        for symbol in elements:
            wanted[Element(symbol).Z - 1] = True
        # A row belongs in the subspace when it introduces no element the target
        # does not have: rows & ~wanted must be empty.
        return ~(self._matrix & ~wanted).any(axis=1)

    def diagram(self, elements: FrozenSet[str]):
        """The phase diagram over ``elements`` and all its subsystems."""
        from pymatgen.analysis.phase_diagram import PDEntry  # noqa: PLC0415

        if elements not in self._diagrams:
            inside = self.reference[self._mask(elements)]
            if inside.empty:
                raise ValueError(f"No reference entries cover {sorted(elements)}")
            entries = [
                PDEntry(row.full_formula, row.energy_corrected, str(row.Index))
                for row in inside.itertuples()
            ]
            self._diagrams[elements] = _phase_diagram(entries)
        return self._diagrams[elements]

    def hull_energy_per_atom(self, formula: str) -> float:
        """What a structure at ``formula`` must beat, in eV/atom of formation energy."""
        from pymatgen.core.composition import Composition  # noqa: PLC0415

        composition = Composition(formula)
        elements = frozenset(str(element) for element in composition.elements)
        return formation_hull_energy_per_atom(self.diagram(elements), composition)


def formation_hull_energy_per_atom(diagram, composition) -> float:
    """Convert pymatgen's absolute hull energy to formation energy per atom.

    ``PhaseDiagram.get_hull_energy_per_atom`` returns an absolute energy even
    though the phase diagram also knows its elemental references. The screeners
    predict formation energies, so comparing that value directly would mix scales.
    """
    elemental_reference = sum(
        composition[element] * diagram.el_refs[element].energy_per_atom
        for element in composition.elements
    ) / composition.num_atoms
    return diagram.get_hull_energy_per_atom(composition) - elemental_reference


def hull_energy_per_atom(reference: pd.DataFrame, formula: str) -> float:
    """The formation energy a structure at ``formula`` must beat to be on the hull.

    For a formula the archive already holds, ``e_form - e_hull`` gives this for
    free. For a formula nobody has computed there is no row to subtract, so the
    phase diagram has to be built and evaluated at that composition -- which is
    the case the screener exists for.
    """
    from pymatgen.core.composition import Composition  # noqa: PLC0415

    composition = Composition(formula)
    elements = frozenset(str(element) for element in composition.elements)
    entries = entries_for_chemsys(reference, elements)
    if not entries:
        raise ValueError(f"No reference entries cover {formula}")
    diagram = _phase_diagram(entries)
    return formation_hull_energy_per_atom(diagram, composition)


def total_energy_for(diagram, composition, formation_energy_per_atom: float) -> float:
    """Invert ``get_form_energy_per_atom``: a formation energy back to a total energy.

    The model speaks in formation energy per atom; a phase diagram is built from
    total energies. The elemental references that connect them belong to the
    diagram, so the conversion has to happen against the same one the candidate
    will be added to.
    """
    reference = sum(
        composition[element] * diagram.el_refs[element].energy_per_atom
        for element in composition.elements
    )
    return formation_energy_per_atom * composition.num_atoms + reference


def displaces_anything(
    diagram_entries: Sequence,
    composition,
    formation_energy_per_atom: float,
    protected: Sequence,
    tolerance: float,
) -> bool:
    """Would a compound this low push a protected entry off the hull?"""
    from pymatgen.analysis.phase_diagram import PDEntry  # noqa: PLC0415

    base = _phase_diagram(list(diagram_entries))
    candidate = PDEntry(composition, total_energy_for(base, composition, formation_energy_per_atom),
                        "__candidate__")
    widened = _phase_diagram([*diagram_entries, candidate])
    return any(widened.get_e_above_hull(entry) > tolerance for entry in protected)


def displacement_bound(
    reference: pd.DataFrame,
    formula: str,
    protected_ids: Optional[Iterable[str]] = None,
    tolerance: float = 1e-3,
    precision: float = 1e-3,
    deepest: float = DEEPEST,
) -> float:
    """``L(X)``: the lowest formation energy per atom that falsifies nothing.

    Args:
        reference: Hull entries, indexed by id, with ``full_formula``,
            ``energy_corrected`` and ``chemsys``.
        formula: The candidate composition.
        protected_ids: Ids of entries that must stay on the hull -- the
            experimentally observed ones. Defaults to every entry in the
            candidate's chemical system that is currently on the hull.
        tolerance: How far above the hull counts as displaced, eV/atom. Not zero:
            about half of ICSD entries sit slightly above a computed hull anyway.
        precision: Bisection precision, eV/atom.
        deepest: Lower end of the search.

    Returns:
        ``L(X)`` in eV/atom, or ``-inf`` when nothing in this chemical system can
        be displaced and the bound says nothing.
    """
    from pymatgen.core.composition import Composition  # noqa: PLC0415

    composition = Composition(formula)
    elements = frozenset(str(element) for element in composition.elements)
    entries = entries_for_chemsys(reference, elements)
    if not entries:
        raise ValueError(f"No reference entries cover {formula}")

    base = _phase_diagram(entries)
    if protected_ids is None:
        protected = [entry for entry in base.stable_entries if entry.name != "__candidate__"]
    else:
        wanted = set(protected_ids)
        protected = [entry for entry in entries if entry.name in wanted]
    if not protected:
        return float("-inf")

    # Monotone in e: a lower candidate can only lower the hull, so displacement,
    # once true, stays true. Bisect the boundary.
    if not displaces_anything(entries, composition, deepest, protected, tolerance):
        return float("-inf")
    low, high = deepest, formation_hull_energy_per_atom(base, composition)
    if displaces_anything(entries, composition, high, protected, tolerance):
        return high
    while high - low > precision:
        middle = 0.5 * (low + high)
        if displaces_anything(entries, composition, middle, protected, tolerance):
            low = middle
        else:
            high = middle
    return high


def shortlist(
    prediction: pd.DataFrame,
    margin: bool = True,
    top: Optional[int] = None,
) -> pd.DataFrame:
    """Rank formulas by how confidently their floor clears the hull.

    Args:
        prediction: Output of :func:`~.train.predict`, carrying ``location``,
            ``sigma_epistemic`` and ``hull``.
        margin: Apply Wren's uncertainty adjustment, ranking on
            ``location + sigma`` rather than ``location``. Ranking millions of
            candidates on a point estimate puts the largest positive errors at
            the top, so this is the default.
        top: Keep only this many rows.
    """
    from wyckoff_transformer.formula_energy.metrics import probability_below_hull  # noqa: PLC0415

    frame = prediction.copy()
    frame["score"] = frame["location"] + (frame["sigma_epistemic"] if margin else 0.0) - frame["hull"]
    frame["p_below_hull"] = probability_below_hull(
        frame["location"].to_numpy(), frame["sigma_epistemic"].to_numpy(), frame["hull"].to_numpy()
    )
    frame = frame.sort_values("score")
    return frame.head(top) if top else frame


def apply_displacement_filter(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    protected_ids: Optional[Iterable[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Add ``L(X)`` and a ``consistent`` flag to a shortlist.

    ``consistent`` is False where the predicted floor is so low that some
    experimentally observed compound would have to be unstable -- which is
    evidence the prediction is wrong, not that the compound is.
    """
    bounds = []
    for formula in candidates.index:
        try:
            bounds.append(displacement_bound(reference, formula, protected_ids, **kwargs))
        except (ValueError, KeyError) as problem:
            logger.warning("no displacement bound for %s: %s", formula, problem)
            bounds.append(float("-inf"))
    out = candidates.copy()
    out["displacement_bound"] = bounds
    out["consistent"] = out["location"] >= out["displacement_bound"]
    return out
