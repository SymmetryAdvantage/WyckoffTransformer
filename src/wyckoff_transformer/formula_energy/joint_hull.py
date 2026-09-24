"""A hull that the candidates are on, not only compared against.

:class:`~wyckoff_transformer.formula_energy.screen.HullLookup` answers "how far is
this gene below the DFT hull", one gene at a time. Candidates never compete: two
hundred polymorphs of one formula predicted below the hull all look like hits,
and a ternary predicted below the DFT hull still looks like one when a binary in
the same cohort is predicted lower still. This module puts the candidates' own
(predicted) energies into the phase diagram next to the DFT entries and scores
each one against **the hull of everything else**:

    score(c) = E(c) - H_{-c}(x_c)

where ``H_{-c}`` is the lower hull of the DFT reference and every other candidate.
For a candidate that is not a vertex of the joint hull this is its ordinary
``e_above_hull``, >= 0. For a vertex it is negative, how far below everything
else it sits -- so the best polymorph of a formula is scored against the next
best, and the same formula cannot fill a budget twice over.

Unaries are scored against the reference and never added: the elemental
references anchor every formation energy, and a generated element predicted
below one would silently re-base the whole diagram.

The energies are formation energies per atom, the scale the regressor predicts,
converted to total energies against the reference diagram's own elemental
references (:func:`~wyckoff_transformer.formula_energy.screen.total_energy_for`).
"""
from __future__ import annotations

import logging
from typing import Dict, FrozenSet

import numpy as np
import pandas as pd

from wyckoff_transformer.formula_energy.screen import (
    HullLookup,
    _phase_diagram,
    formation_hull_energy_per_atom,
    total_energy_for,
)

logger = logging.getLogger(__name__)

_CANDIDATE_PREFIX = "__candidate__"


def _elements(composition) -> FrozenSet[str]:
    return frozenset(str(element) for element in composition.elements)


def joint_hull_scores(
    candidates: pd.DataFrame,
    lookup: HullLookup,
    energy_column: str = "energy",
    formula_column: str = "formula",
) -> pd.DataFrame:
    """Score every candidate against the hull of the reference and every other candidate.

    Args:
        candidates: One row per *distinct* gene -- duplicates would shadow each
            other -- with a formula and a formation energy per atom. Rows with a
            missing energy or formula are returned undecided.
        lookup: The DFT reference, whose per-system diagrams are reused.

    Returns:
        ``joint_hull_energy`` (``H_{-c}`` at the candidate's composition, as a
        formation energy) and ``joint_e_hull`` (the score), indexed like
        *candidates*; NaN where no reference diagram covers the system.
    """
    from pymatgen.analysis.phase_diagram import PDEntry  # noqa: PLC0415
    from pymatgen.core.composition import Composition  # noqa: PLC0415

    out = pd.DataFrame(
        {"joint_hull_energy": np.nan, "joint_e_hull": np.nan}, index=candidates.index)
    usable = candidates[candidates[energy_column].notna() & candidates[formula_column].notna()]
    if usable.empty:
        return out
    compositions = {index: Composition(formula)
                    for index, formula in usable[formula_column].items()}
    element_sets = {index: _elements(comp) for index, comp in compositions.items()}
    energies = usable[energy_column].astype(float).to_dict()

    # Unaries: against the reference only.
    for index, elements in element_sets.items():
        if len(elements) == 1:
            try:
                hull = lookup.hull_energy_per_atom(usable.at[index, formula_column])
            except (KeyError, ValueError) as error:
                logger.debug("No reference hull for %s: %s", usable.at[index, formula_column], error)
                continue
            out.at[index, "joint_hull_energy"] = hull
            out.at[index, "joint_e_hull"] = energies[index] - hull

    compound = {index: elements for index, elements in element_sets.items() if len(elements) > 1}
    systems: Dict[FrozenSet[str], list] = {}
    for index, elements in compound.items():
        systems.setdefault(elements, []).append(index)

    for system, members in sorted(systems.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
        try:
            base = lookup.diagram(system)
        except (KeyError, ValueError) as error:
            logger.warning("No reference diagram for %s: %s", "-".join(sorted(system)), error)
            continue
        # Reference entries that are off the DFT hull stay off it however the
        # candidates move, so only its vertices are needed.
        reference_entries = list(base.stable_entries)
        inside = [index for index, elements in compound.items() if elements <= system]
        entries = {}
        for index in inside:
            composition = compositions[index]
            entries[index] = PDEntry(
                composition, total_energy_for(base, composition, energies[index]),
                f"{_CANDIDATE_PREFIX}{index}")
        try:
            joint = _phase_diagram(reference_entries + list(entries.values()))
        except Exception as error:  # noqa: BLE001 - qhull raises several types
            logger.warning("Joint diagram for %s failed: %s", "-".join(sorted(system)), error)
            continue
        stable_names = {entry.name for entry in joint.stable_entries}
        for index in members:
            composition = compositions[index]
            entry = entries[index]
            if entry.name in stable_names:
                others = reference_entries + [e for i, e in entries.items() if i != index]
                hull = formation_hull_energy_per_atom(_phase_diagram(others), composition)
            else:
                hull = formation_hull_energy_per_atom(joint, composition)
            out.at[index, "joint_hull_energy"] = hull
            out.at[index, "joint_e_hull"] = energies[index] - hull
    return out
