"""The two families of hull energy in this repository must be convertible.

`docs/e_hull_definitions.md` lists six things called some variant of `e_hull`.
Two of them are load-bearing and live in different conventions:

* the de novo protocol's :class:`HullEnergyCalculator`, which takes an
  *absolute* MLIP total energy and returns the distance above the hull;
* the formula screeners' :class:`HullLookup`, which returns the hull's own
  *formation* energy at a composition, because that is the scale a predicted
  formation energy can be compared against.

They are related by ``e_hull = e_form - e_hull_at_composition``, the elemental
references cancelling in the difference.  If that identity ever breaks -- a
changed one-hot encoding, a different subspace rule, a pymatgen convention
shift -- a screener's threshold and the protocol's threshold stop meaning the
same thing, silently.  So it is pinned here, on a hull small enough to check by
hand.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

#: A two-element system with one binary, so the hull is two straight segments.
#: Energies are total, in eV, as both conventions expect.
REFERENCE = [
    # species_at_sites            energy  full_formula  chemsys
    (["Li"],                       -2.0,  "Li1",        "Li"),
    (["O", "O"],                   -8.0,  "O2",         "O"),
    (["Li", "Li", "O"],           -18.0,  "Li2O1",      "Li-O"),
]


def _hull_parquet(directory: Path) -> Path:
    """The reference in the shape :class:`HullEnergyCalculator` reads."""
    path = directory / "hull.parquet"
    pd.DataFrame(
        {
            "species_at_sites": [row[0] for row in REFERENCE],
            "energy": [row[1] for row in REFERENCE],
        }
    ).to_parquet(path)
    return path


def _reference_frame() -> pd.DataFrame:
    """The same reference in the shape :class:`HullLookup` reads."""
    return pd.DataFrame(
        {
            "full_formula": [row[2] for row in REFERENCE],
            "chemsys": [row[3] for row in REFERENCE],
            "energy_corrected": [row[1] for row in REFERENCE],
        },
        index=pd.Index(["ref-1", "ref-2", "ref-3"], name="immutable_id"),
    )


class TestHullConventions(unittest.TestCase):
    #: Query structures: one above the hull, one on it, one below it.
    QUERIES = (
        ("Li2O2", -20.0),   # formula, total energy in eV
        ("Li2O1", -18.0),   # exactly a reference entry: distance 0
        ("Li2O1", -19.0),   # below the hull: the negative case
    )

    def _absolute(self, parquet: Path, formula: str, energy: float) -> float:
        from pymatgen.core import Composition

        from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator

        # orb_conserv_inf names a runnable hull; the parquet overrides its data.
        hull = HullEnergyCalculator("orb_conserv_inf", parquet=parquet)
        return hull.energy_above_hull(energy, Composition(formula))

    def _formation(self, formula: str, energy: float):
        """``(e_form, e_hull_at_composition)`` in the screeners' convention."""
        from pymatgen.analysis.phase_diagram import PDEntry
        from pymatgen.core import Composition

        from wyckoff_transformer.formula_energy.screen import HullLookup

        lookup = HullLookup(_reference_frame())
        composition = Composition(formula)
        diagram = lookup.diagram(
            frozenset(str(element) for element in composition.elements)
        )
        e_form = diagram.get_form_energy_per_atom(PDEntry(composition, energy))
        return e_form, lookup.hull_energy_per_atom(formula)

    def test_distance_equals_formation_energy_minus_hull_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = _hull_parquet(Path(tmp))
            for formula, energy in self.QUERIES:
                with self.subTest(formula=formula, energy=energy):
                    distance = self._absolute(parquet, formula, energy)
                    e_form, level = self._formation(formula, energy)
                    self.assertAlmostEqual(distance, e_form - level, places=9)

    def test_the_absolute_convention_reports_below_hull_as_negative(self):
        """The protocol must not clip: a structure below the reference is news."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet = _hull_parquet(Path(tmp))
            self.assertAlmostEqual(
                self._absolute(parquet, "Li2O1", -19.0), -1.0 / 3.0, places=9
            )
            self.assertAlmostEqual(
                self._absolute(parquet, "Li2O1", -18.0), 0.0, places=9
            )

    def test_a_reference_entry_sits_exactly_on_its_own_hull(self):
        """Which is why the published 1 meV/atom slice keeps every vertex."""
        e_form, level = self._formation("Li2O1", -18.0)
        self.assertAlmostEqual(e_form, level, places=9)


if __name__ == "__main__":
    unittest.main()
