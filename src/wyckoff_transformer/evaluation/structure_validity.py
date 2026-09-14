"""Structure validity, as LeMat-GenBench computes it.

A structure is valid only if it passes all three of the benchmark's checks:
charge neutrality, minimum interatomic distance, and physical plausibility.
This is a port of ``lemat_genbench.metrics.validity_metrics`` narrowed to the
per-structure predicates -- the aggregation and config plumbing around them is
not needed here, since :mod:`wyckoff_transformer.evaluation.protocol` does its
own accounting.

The logic is reproduced faithfully rather than improved, including behaviour
that looks accidental, because the point is to reproduce the benchmark's
numbers.  Two such quirks are marked inline.  Defaults match the values in
LeMat-GenBench's ``comprehensive_multi_mlip_hull.yaml``.

``tests/test_genbench_equivalence.py`` pins the result against the benchmark's
own implementation structure by structure.

Attribution: derived from LeMaterial/lemat-genbench (Apache-2.0),
https://github.com/LeMaterial/lemat-genbench.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact.metallicity import metallicity_score

from wyckoff_transformer.evaluation.oxidation_state import (
    OXI_STATE_MAPPING_FILE,
    _load_json,
    compositional_oxi_state_guesses,
    get_inequivalent_site_info,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidityThresholds:
    """Thresholds from LeMat-GenBench's leaderboard configuration."""

    charge_tolerance: float = 0.1
    distance_scaling: float = 0.5
    min_atomic_density: float = 0.00001
    max_atomic_density: float = 0.5
    min_mass_density: float = 0.01
    max_mass_density: float = 25.0
    check_format: bool = True
    check_symmetry: bool = True


DEFAULT_THRESHOLDS = ValidityThresholds()

#: Above this metallicity, charge balance is not meaningful and is not required.
METALLICITY_CUTOFF = 0.7

#: Returned when no charge-balanced assignment is credible.  Any value above
#: ``charge_tolerance`` fails the check; the magnitude is not otherwise used.
LARGE_CHARGE_DEVIATION = 10.0


@lru_cache(maxsize=1)
def _element_radii() -> dict[str, float]:
    """Atomic radii by element symbol, defaulting to 1.0 A where unknown."""
    return {
        str(element): element.atomic_radius or 1.0
        for element in Element
        if hasattr(element, "atomic_radius")
    }


def charge_deviation(
    structure: Structure, bv_analyzer: Optional[BVAnalyzer] = None
) -> float:
    """Absolute deviation from charge neutrality; 0.0 means neutral.

    Tried in order: metallicity, bond-valence sums, pymatgen's oxidation-state
    decoration, then compositional guessing against LeMat-Bulk's ICSD priors.
    """
    bv_analyzer = bv_analyzer or BVAnalyzer()

    # 1. Metals need no charge balance.
    try:
        if metallicity_score(Composition(structure.formula)) > METALLICITY_CUTOFF:
            return 0.0
        sites = get_inequivalent_site_info(structure)
        bond_valences = [
            calculate_bv_sum(
                structure[site_index],
                get_neighbors_of_site_with_index(structure, site_index),
            )
            for site_index in sites["sites"]
        ]
        if all(abs(value) < 1e-15 for value in bond_valences):
            return 0.0  # every bond valence vanishes: metallic
    except Exception as exc:
        logger.debug("Bond valence analysis failed: %s", exc)

    # 2. Pymatgen's own oxidation-state decoration.
    try:
        decorated = bv_analyzer.get_oxi_state_decorated_structure(structure)
        return abs(sum(site.specie.oxi_state for site in decorated.sites))
    except ValueError:
        logger.debug("Could not decorate oxidation states")

    # 3. Compositional guessing, first against the ICSD priors.
    composition = Composition(structure.composition)
    mapping = _load_json(OXI_STATE_MAPPING_FILE)
    override = {
        str(element): mapping[str(element)]
        for element in composition.elements
        if str(element) in mapping
    }
    try:
        compositional_oxi_state_guesses(
            composition,
            all_oxi_states=False,
            max_sites=-1,
            target_charge=0,
            oxi_states_override=override,
        )[2][0]
        # Quirk, preserved: LeMat-GenBench branches on this score but returns
        # 0.0 either way, so reaching here at all means "charge balanced".
        return 0.0
    except IndexError:
        pass

    # 4. Fall back to all oxidation states, ranked by how well they
    #    anti-correlate with electronegativity.
    try:
        correlation = -compositional_oxi_state_guesses(
            composition,
            all_oxi_states=True,
            max_sites=-1,
            target_charge=0,
            oxi_states_override=None,
        )[2][0]
    except IndexError:
        return LARGE_CHARGE_DEVIATION
    return 0.0 if correlation > 0.0 else LARGE_CHARGE_DEVIATION


def distances_valid(
    structure: Structure, scaling_factor: float = DEFAULT_THRESHOLDS.distance_scaling
) -> bool:
    """True when every interatomic distance clears the scaled radius sum."""
    if len(structure) <= 1:
        return True  # a single atom has no pair to check

    radii = _element_radii()
    distances = structure.distance_matrix
    symbols = [str(site.specie) for site in structure]
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            minimum = (
                0.7 + radii.get(symbols[i], 1.0) + radii.get(symbols[j], 1.0)
            ) * scaling_factor
            if distances[i, j] < minimum:
                logger.debug(
                    "%s-%s distance %.3f A is below the minimum %.3f A",
                    symbols[i], symbols[j], distances[i, j], minimum,
                )
                return False
    return True


def physically_plausible(
    structure: Structure, thresholds: ValidityThresholds = DEFAULT_THRESHOLDS
) -> bool:
    """True when density, lattice, CIF round-trip and symmetry all pass.

    Every check must pass; a check that raises counts as failed.
    """
    passed = 0
    required = 3

    try:
        if thresholds.min_mass_density <= structure.density <= thresholds.max_mass_density:
            passed += 1
    except Exception as exc:
        logger.debug("Could not compute mass density: %s", exc)

    try:
        atomic_density = len(structure) / structure.volume
        if thresholds.min_atomic_density <= atomic_density <= thresholds.max_atomic_density:
            passed += 1
    except Exception as exc:
        logger.debug("Could not compute atomic density: %s", exc)

    try:
        lattice = structure.lattice
        if (
            lattice.volume > 1.0
            and all(1.0 <= p <= 100.0 for p in lattice.abc)
            and all(0 < angle < 180 for angle in lattice.angles)
        ):
            passed += 1
    except Exception as exc:
        logger.debug("Could not validate the lattice: %s", exc)

    if thresholds.check_format:
        required += 1
        try:
            str(CifWriter(structure))
            passed += 1
        except Exception as exc:
            logger.debug("CIF round-trip failed: %s", exc)

    if thresholds.check_symmetry:
        required += 1
        try:
            # Quirk, preserved: this uses SpacegroupAnalyzer's default symprec
            # (0.01), not the tolerance used anywhere else.
            if 1 <= SpacegroupAnalyzer(structure).get_space_group_number() <= 230:
                passed += 1
        except Exception as exc:
            logger.debug("Symmetry check failed: %s", exc)

    return passed == required


def is_valid(
    structure: Structure,
    thresholds: ValidityThresholds = DEFAULT_THRESHOLDS,
    bv_analyzer: Optional[BVAnalyzer] = None,
) -> bool:
    """True when *structure* passes charge, distance and plausibility checks.

    Any check that raises counts as failed, so a pathological structure is
    invalid rather than an error.
    """
    try:
        charge_ok = charge_deviation(structure, bv_analyzer) <= thresholds.charge_tolerance
    except Exception:
        charge_ok = False

    try:
        distance_ok = distances_valid(structure, thresholds.distance_scaling)
    except Exception:
        distance_ok = False

    try:
        plausible_ok = physically_plausible(structure, thresholds)
    except Exception:
        plausible_ok = False

    return bool(charge_ok and distance_ok and plausible_ok)
