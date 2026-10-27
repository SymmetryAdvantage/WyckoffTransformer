"""The Li-free framework of a structure, and grouping structures by it.

A *framework* is what is left of a structure when the mobile ion is removed and every
other species is anonymised to its role: one label for the non-Li cations and one for
the anions.  Jun et al. (Nat. Mater. 21, 924, 2022) classify Li conductors by the
connectivity of exactly this sublattice; two compounds with matching frameworks
belong to the same structural family whatever their chemistry, so Li3YCl6 and
Li3ErCl6 match, LGPS matches Li10SnP2S12, and the doped garnets match the ordered
Li7La3Zr2O12 parent in LeMat.

Disordered structures (most OBELiX CIFs) are handled site by site: a site whose
non-Li occupancy is at least one half is a framework site, labelled by the role of its
majority non-Li species; anything else is dropped with the Li.  A dopant at 0.2 on a Li
site therefore leaves no trace, and a mixed Ge/P site stays one cation site.

The anion/cation role comes from bond-valence oxidation states when the structure is
ordered and :class:`~pymatgen.analysis.bond_valence.BVAnalyzer` can assign them, and
from :func:`anion_elements` otherwise.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from pymatgen.core import Element, Structure

logger = logging.getLogger(__name__)

#: Placeholder species of the anonymised framework.  Any two distinct elements would
#: do; these are chosen so that a framework is never mistaken for a real compound.
CATION_LABEL = "Xe"
ANION_LABEL = "Rn"

HALOGENS = frozenset({"F", "Cl", "Br", "I"})
CHALCOGENS_PNICTOGENS = frozenset({"N", "S", "Se", "Te"})
#: Elements that are anions only when no conventional anion is present: phosphides,
#: arsenides, carbides, silicides and the like, and hydrides.
WEAK_ANIONS = frozenset({"P", "As", "Sb", "Bi", "C", "Si", "Ge", "Sn", "B", "H"})


def anion_elements(elements, mobile: str = "Li") -> frozenset[str]:
    """Which elements of a composition are anions, by a composition-only rule.

    * O and the halogens always are.
    * N, S, Se and Te are, unless O or F is present (sulfates, selenites, nitrates).
    * Without any of those, the most electronegative of P, As, Sb, Bi, C, Si, Ge, Sn,
      B and H is the anion (Li3P, Li9AlP4, LiBH4, Li2C2).
    * H is an anion only in that last case, so LiNH2 and LiOH keep it a cation.

    This is the generic anion-role rule of the plan's gene screen, extended to the
    elements the palette allows.  It cannot tell S2- from S6+ in an oxysulfide; a
    structure-level assignment should be preferred where there is one.
    """
    found = {str(element) for element in elements} - {mobile}
    anions = {element for element in found if element in HALOGENS or element == "O"}
    with_oxide_or_fluoride = bool(found & {"O", "F"})
    if not with_oxide_or_fluoride:
        anions |= found & CHALCOGENS_PNICTOGENS
    if not anions:
        weak = [element for element in found if element in WEAK_ANIONS]
        if weak:
            anions.add(max(weak, key=lambda element: Element(element).X))
    return frozenset(anions)


def structure_anions(structure: Structure, mobile: str = "Li") -> frozenset[str]:
    """Anion elements from bond-valence states if possible, else :func:`anion_elements`.

    An element counts as an anion when the majority of its sites are negative.
    """
    if structure.is_ordered:
        try:
            from pymatgen.analysis.bond_valence import BVAnalyzer

            valences = BVAnalyzer().get_valences(structure)
        except (ValueError, KeyError, TypeError):
            valences = None
        if valences is not None:
            negative: dict[str, list[bool]] = defaultdict(list)
            for site, valence in zip(structure, valences):
                negative[site.specie.symbol].append(valence < 0)
            return frozenset(element for element, flags in negative.items()
                             if element != mobile and np.mean(flags) > 0.5)
    return anion_elements(structure.composition.elements, mobile)


def framework(structure: Structure, mobile: str = "Li",
              anions: frozenset[str] | None = None,
              min_occupancy: float = 0.5) -> Structure | None:
    """The anonymised framework: mobile ion removed, cations and anions relabelled.

    Args:
        structure: Ordered or disordered.
        mobile: The ion removed.
        anions: The anion elements; :func:`structure_anions` by default.
        min_occupancy: A site is kept if its non-mobile occupancy reaches this.

    Returns:
        An ordered structure of :data:`CATION_LABEL` and :data:`ANION_LABEL` sites, or
        ``None`` when nothing is left (Li metal, or a structure of Li and vacancies).
    """
    if anions is None:
        anions = structure_anions(structure, mobile)
    species, coords = [], []
    for site in structure:
        others = {sp: occ for sp, occ in site.species.items()
                  if getattr(sp, "symbol", str(sp)) != mobile}
        if sum(others.values()) < min_occupancy:
            continue
        anion_share = sum(occ for sp, occ in others.items()
                          if getattr(sp, "symbol", str(sp)) in anions)
        species.append(ANION_LABEL if anion_share * 2 >= sum(others.values()) else CATION_LABEL)
        coords.append(site.frac_coords)
    if not species:
        return None
    return Structure(structure.lattice, species, coords)


def framework_formula(frame: Structure) -> str:
    """The framework's reduced formula, e.g. ``XeRn4`` for a PS4 or GeS4 framework."""
    return frame.composition.reduced_formula


def default_matcher(ltol: float = 0.2, stol: float = 0.3, angle_tol: float = 5.0):
    """The framework matcher, with tolerances to be calibrated at gate G1b.

    ``scale=True`` because removing Li changes the volume the framework had to hold;
    ``attempt_supercell`` lets a primitive cell match a conventional one.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=True,
                            scale=True, attempt_supercell=True)


def group_frameworks(frames: dict[str, Structure], matcher=None) -> dict[str, int]:
    """Group structures by matching frameworks: key -> group number.

    Frameworks are compared only within the same reduced formula, which is what makes
    grouping ten thousand of them affordable.
    """
    matcher = matcher or default_matcher()
    by_formula: dict[str, list[str]] = defaultdict(list)
    for key, frame in frames.items():
        by_formula[framework_formula(frame)].append(key)
    groups: dict[str, int] = {}
    next_group = 0
    for keys in by_formula.values():
        representatives: list[tuple[int, Structure]] = []
        for key in keys:
            frame = frames[key]
            for group, representative in representatives:
                if matcher.fit(representative, frame):
                    groups[key] = group
                    break
            else:
                representatives.append((next_group, frame))
                groups[key] = next_group
                next_group += 1
    return groups


def match_to_reference(frame: Structure, references: dict[str, Structure],
                       matcher=None) -> list[str]:
    """Keys of the reference frameworks that *frame* matches (same reduced formula)."""
    matcher = matcher or default_matcher()
    formula = framework_formula(frame)
    return [key for key, reference in references.items()
            if framework_formula(reference) == formula and matcher.fit(reference, frame)]
