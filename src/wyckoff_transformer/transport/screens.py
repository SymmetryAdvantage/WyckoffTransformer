"""Family-blind chemistry screens: generic anion roles and the coordination check.

The families campaign fixed each family's anions (``FAMILY_ANIONS`` in
``scripts/solid_electrolyte_campaign.py``). Outside the families the roles have to be
generic. The plan's two stages:

* **gene stage** (:func:`generic_oxidation_states`): one oxidation state per element,
  summing to zero, every anion more electronegative than every cation (the families
  campaign's charge balance), and in addition halogens and O are negative, N, S, Se and
  Te are positive only when O or F is present (sulfates and selenites stay; molecular
  N-Cl and S-I solids and halogen oxoanions go), and the metals are positive;
* **after relaxation** (:func:`bond_valence_roles`): the bond-valence oxidation states of
  the structure (``BVAnalyzer``) must exist and obey the same sign rules, which tells a
  phosphate P(+5) from a phosphide P(-3).

The coordination check compares every clean cation environment (species, CN, polyhedron)
of :func:`wyckoff_transformer.evaluation.coordination_motifs.analyse` with LeMat's
frequency table, as ``scripts/screen_coordination_motifs.py report`` builds it.
"""
from __future__ import annotations

from functools import cache
from itertools import product

HALOGENS = frozenset({"F", "Cl", "Br", "I"})
#: Negative unless O or F is present.
CONDITIONAL_ANIONS = frozenset({"N", "S", "Se", "Te"})
#: Either sign: hydride or proton, boride or borate, phosphide or phosphate.
AMPHOTERIC = frozenset({"H", "B", "C", "Si", "Ge", "Sn", "P", "As", "Sb", "Bi"})


def allowed_signs(element: str, elements: set[str]) -> set[int]:
    """The signs *element* may take in a compound of *elements*."""
    if element in HALOGENS or element == "O":
        return {-1}
    if element in CONDITIONAL_ANIONS:
        return {-1, 1} if elements & {"O", "F"} else {-1}
    if element in AMPHOTERIC:
        return {-1, 1}
    return {1}


@cache
def generic_oxidation_states(formula: str) -> str | None:
    """One oxidation state per element under the generic roles, e.g. ``Cl-1 Li+1 Y+3``.

    States are pymatgen's common and ICSD-frequent ones (Cu+1 and Ti+3 are ICSD-frequent
    but not "common"); the first assignment that balances, obeys
    :func:`allowed_signs` and puts every anion above every cation in electronegativity
    is returned, or None.
    """
    from pymatgen.core import Composition

    composition = Composition(formula).reduced_composition
    elements = list(composition.elements)
    symbols = {e.symbol for e in elements}
    choices = []
    for element in elements:
        signs = allowed_signs(element.symbol, symbols)
        states = sorted({s for s in (*element.common_oxidation_states,
                                     *element.icsd_oxidation_states)
                         if s != 0 and (1 if s > 0 else -1) in signs},
                        key=lambda s: (s not in element.common_oxidation_states, abs(s)))
        if not states:
            return None
        choices.append(states)
    for states in product(*choices):
        if abs(sum(composition[e] * s for e, s in zip(elements, states))) > 1e-6:
            continue
        anions = [e.X for e, s in zip(elements, states) if s < 0]
        cations = [e.X for e, s in zip(elements, states) if s > 0]
        if not anions or (cations and min(anions) < max(cations)):
            continue
        return " ".join(f"{e}{s:+d}" for e, s in sorted(zip(elements, states),
                                                         key=lambda pair: pair[0].symbol))
    return None


def bond_valence_roles(structure) -> tuple[str | None, str]:
    """``(states, reason)``: BVAnalyzer's per-element states if they obey the roles.

    *states* is None when BVAnalyzer finds no assignment ("not_ionic"), when an element
    takes two signs ("mixed_sign") or when a sign breaks :func:`allowed_signs`
    ("role:<element><sign>").
    """
    from pymatgen.analysis.bond_valence import BVAnalyzer

    try:
        valences = BVAnalyzer().get_valences(structure)
    except ValueError:
        return None, "not_ionic"
    symbols = {site.specie.symbol for site in structure}
    by_element: dict[str, set[int]] = {}
    for site, valence in zip(structure, valences):
        by_element.setdefault(site.specie.symbol, set()).add(int(valence))
    for element, states in by_element.items():
        signs = {1 if s > 0 else -1 for s in states if s != 0}
        if 0 in states:
            return None, f"role:{element}0"
        if len(signs) > 1:
            return None, "mixed_sign"
        if not signs <= allowed_signs(element, symbols):
            return None, f"role:{element}{'+' if 1 in signs else '-'}"
    states = " ".join(f"{e}{'/'.join(f'{s:+d}' for s in sorted(v))}"
                      for e, v in sorted(by_element.items()))
    return states, "ok"


#: ``screen_coordination_motifs.py report``'s clean site: the next anion is at least 10%
#: farther than the last bonded one, and the shape fits its polyhedron within 10 deg RMS.
CLEAN_GAP, CLEAN_ANGLE_RMS = 1.10, 10.0


def environment_keys(site_rows: list[dict]) -> list[tuple[str, int, str]]:
    """``(species, CN, polyhedron)`` of the clean sites among ``analyse``'s site rows.

    A site whose coordination number is ill defined has no row in the frequency table
    either, so it is not checked.
    """
    return [(f"{row['element']}{int(row['ox']):+d}", int(row["cn"]), row["geometry"])
            for row in site_rows
            if row["gap"] >= CLEAN_GAP and row["angle_rms"] <= CLEAN_ANGLE_RMS]
