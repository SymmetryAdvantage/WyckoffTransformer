"""Tests for the archive-wide hull table.

The numbers below are chosen so the hull can be checked by hand: Li at −2 eV/atom
and O at −4 eV/atom are the elemental references, so Li2O at −18 eV total has
formation energy (−18 − 2·(−2) − 1·(−4)) / 3 = −10/3 eV/atom and, being the only
binary, defines the hull at its composition.
"""
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.formula_energy import hull_table as ht

ARCHIVE = pd.DataFrame(
    {
        "immutable_id": ["li", "o", "li2o", "li2o-poor", "yb2o3", "uo2"],
        "full_formula": ["Li1", "O2", "Li2O1", "Li2O1", "Yb2O3", "U1O2"],
        "chemsys": ["Li", "O", "Li-O", "Li-O", "O-Yb", "O-U"],
        "energy_corrected": [-2.0, -8.0, -18.0, -15.0, -40.0, -20.0],
    }
)
#: Yb and U need their own elemental references, or their systems cannot be
#: placed at all -- the case the exclusions used to hide.
ELEMENTS = pd.DataFrame(
    {
        "immutable_id": ["yb", "u"],
        "full_formula": ["Yb1", "U1"],
        "chemsys": ["Yb", "U"],
        "energy_corrected": [-1.5, -5.0],
    }
)


def _archive() -> pd.DataFrame:
    return pd.concat([ARCHIVE, ELEMENTS], ignore_index=True)


class TestHullEnergies(unittest.TestCase):
    def test_the_hull_defining_rows_sit_at_zero(self):
        labels = ht.hull_energies(_archive())
        by_id = labels.set_index(_archive()["immutable_id"])
        for name in ("li", "o", "li2o"):
            with self.subTest(row=name):
                self.assertAlmostEqual(by_id.loc[name, "e_hull"], 0.0, places=9)

    def test_formation_energy_is_measured_against_the_elemental_references(self):
        labels = ht.hull_energies(_archive())
        by_id = labels.set_index(_archive()["immutable_id"])
        self.assertAlmostEqual(by_id.loc["li2o", "e_form"], -10.0 / 3.0, places=9)
        # A worse Li2O1: 3 eV total above the hull entry, over 3 atoms.
        self.assertAlmostEqual(by_id.loc["li2o-poor", "e_hull"], 1.0, places=9)
        self.assertAlmostEqual(by_id.loc["li", "e_form"], 0.0, places=9)

    def test_elements_the_old_script_excluded_are_labelled(self):
        """Yb and Z >= 84 cost the archive 589,250 labels for no stated reason."""
        labels = ht.hull_energies(_archive())
        by_id = labels.set_index(_archive()["immutable_id"])
        for name in ("yb2o3", "uo2"):
            with self.subTest(row=name):
                self.assertFalse(np.isnan(by_id.loc[name, "e_hull"]))
                self.assertAlmostEqual(by_id.loc[name, "e_hull"], 0.0, places=9)

    def test_a_row_below_a_separate_reference_hull_is_negative(self):
        """The answer key's case: a deep row against the shallow world's hull.

        `compute_e_hull.py` returned `(None, None)` here, because
        `get_e_above_hull` raises below the hull -- which lost `e_form` too.
        """
        archive = _archive()
        # A shallow world that never found either Li2O1: its hull over Li-O is
        # the tie line between the elements, so the real compound is below it by
        # exactly its formation energy.
        shallow = archive[~archive["immutable_id"].isin(["li2o", "li2o-poor"])]
        deep = ARCHIVE[ARCHIVE["immutable_id"] == "li2o"]
        labels = ht.hull_energies(deep, reference=shallow)
        self.assertAlmostEqual(labels["e_hull"].iloc[0], -10.0 / 3.0, places=9)
        self.assertAlmostEqual(labels["e_form"].iloc[0], -10.0 / 3.0, places=9)

    def test_a_system_with_no_elemental_reference_is_counted_not_hidden(self):
        rows = pd.concat([
            ARCHIVE,
            pd.DataFrame({
                "immutable_id": ["ncl"], "full_formula": ["Na1Cl1"],
                "chemsys": ["Cl-Na"], "energy_corrected": [-7.0],
            }),
        ], ignore_index=True)
        with self.assertLogs(ht.logger, level="INFO") as logs:
            labels = ht.hull_energies(rows)
        self.assertTrue(np.isnan(labels["e_hull"].iloc[-1]))
        self.assertTrue(
            any("no phase diagram" in line for line in logs.output),
            f"the failure was not reported: {logs.output}",
        )

    def test_workers_do_not_change_the_answer(self):
        one = ht.hull_energies(_archive(), workers=1)
        many = ht.hull_energies(_archive(), workers=2)
        pd.testing.assert_frame_equal(one, many)


class TestAnnotateCsv(unittest.TestCase):
    def test_every_column_and_the_row_order_survive(self):
        frame = _archive()
        # A heavy passenger column, as the archive's CIF text is.
        frame["cif"] = [f"# cif {name}" for name in frame["immutable_id"]]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.csv.gz"
            frame.to_csv(source, index=False)
            out = ht.annotate_csv(
                source, Path(tmp) / "out.csv.gz", workers=1, chunk_size=3
            )
            with gzip.open(out, "rt") as handle:
                written = pd.read_csv(handle)

        self.assertEqual(
            list(written.columns), list(frame.columns) + list(ht.OUTPUT_COLUMNS)
        )
        pd.testing.assert_frame_equal(written[frame.columns], frame)
        self.assertAlmostEqual(
            float(written.loc[written.immutable_id == "li2o", "e_form"].iloc[0]),
            -10.0 / 3.0, places=6,
        )

    def test_the_parser_can_print_its_own_help(self):
        ht.build_parser().format_help()


if __name__ == "__main__":
    unittest.main()
