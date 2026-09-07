"""Tests for the formula-level table.

Three properties carry the design. A formula is the *reduced* composition, so
the same compound written two ways is one row with two polymorphs. The hull
energy is a property of the composition, so it must agree across a formula's
rows. And the split is by formula, because a row-random split lets a
formula-level model read its answers out of the training set.
"""
import unittest

import numpy as np
import pandas as pd

from wyckoff_transformer.formula_energy import dataset as ds


def _rows():
    """Two compounds: BaTiO3 written two ways, and NaCl written two ways."""
    frame = pd.DataFrame({
        "immutable_id": ["mp-1", "mp-2", "agm9", "oqmd-7", "agm8"],
        "full_formula": ["Ba1 Ti1 O3", "Ba2 Ti2 O6", "Ba1 Ti1 O3", "Na1 Cl1", "Na2 Cl2"],
        "chemsys": ["Ba-O-Ti"] * 3 + ["Cl-Na"] * 2,
        "max_force": [0.001, 0.002, 0.003, 0.004, 0.005],
        "e_form": [-3.0, -2.9, -3.1, -2.0, -2.2],
        "e_hull": [0.1, 0.2, 0.0, 0.2, 0.0],
    })
    provenance = pd.DataFrame({
        "material_id": ["mp-1", "mp-2"], "theoretical": [False, True], "n_icsd": [3, 0],
    }).set_index("material_id")
    frame["kind"] = ds._kinds(frame["immutable_id"], provenance).to_numpy()
    frame["n_icsd"] = frame["immutable_id"].map(provenance["n_icsd"]).fillna(0).astype(np.int32)
    frame["formula"], frame["cell_size"] = ds._keys_and_sizes(frame["full_formula"])
    return frame


class TestFormulaKey(unittest.TestCase):
    def test_cell_multiples_are_one_compound(self):
        # Z is not chemistry: both spellings share a floor, so they must share a key.
        self.assertEqual(
            ds.formula_key(ds.parse_full_formula("Ba2 Ti2 O6")),
            ds.formula_key(ds.parse_full_formula("Ba1 Ti1 O3")),
        )

    def test_key_is_canonical_and_ordered(self):
        self.assertEqual(ds.formula_key(ds.parse_full_formula("Li12 As4 H64 S16 O32")), "As1H16Li3O8S4")

    def test_element_order_does_not_matter(self):
        self.assertEqual(
            ds.formula_key(ds.parse_full_formula("O3 Ti1 Ba1")),
            ds.formula_key(ds.parse_full_formula("Ba1 Ti1 O3")),
        )

    def test_spaces_are_required_to_parse(self):
        # csp.parse_formula rejects gaps between tokens, which is how it catches a
        # typo; pymatgen's Composition.formula always has them.
        with self.assertRaises(ValueError):
            ds.parse_formula("Ba1 Ti1 O3")


class TestKinds(unittest.TestCase):
    def test_theoretical_flag_splits_materials_project(self):
        frame = _rows()
        self.assertEqual(
            frame["kind"].tolist(),
            ["mp_icsd", "mp_theoretical", "agm", "oqmd", "agm"],
        )


    def test_an_unresolved_mp_id_is_not_called_alexandria(self):
        # The API no longer resolves about a thousand of LeMat-Bulk's mp- ids.
        # Falling through to the default would credit them to the wrong search
        # process entirely, and to the one whose bounds are loosest.
        frame = pd.DataFrame({"immutable_id": ["mp-999999", "agm1", "oqmd-1"]})
        provenance = pd.DataFrame({"material_id": ["mp-1"], "theoretical": [False],
                                   "n_icsd": [1]}).set_index("material_id")
        kinds = ds._kinds(frame["immutable_id"], provenance)
        self.assertEqual(kinds.tolist(), ["mp_theoretical", "agm", "oqmd"])


class TestBuildFormulaTable(unittest.TestCase):
    def setUp(self):
        self.table = ds.build_formula_table(_rows())

    def test_polymorphs_collapse_to_one_row(self):
        self.assertEqual(sorted(self.table.index), ["Ba1O3Ti1", "Cl1Na1"])
        self.assertEqual(self.table.loc["Ba1O3Ti1", "n_rows"], 3)
        # Ba1Ti1O3 and Ba2Ti2O6 are the same formula at two cell sizes.
        self.assertEqual(self.table.loc["Ba1O3Ti1", "n_cell_sizes"], 2)

    def test_label_is_the_minimum(self):
        self.assertAlmostEqual(self.table.loc["Ba1O3Ti1", "e_form_min"], -3.1)

    def test_hull_energy_is_a_property_of_the_composition(self):
        # e_form - e_hull is the same number from every row of a formula.
        self.assertAlmostEqual(self.table.loc["Ba1O3Ti1", "e_hull_at_composition"], -3.1)
        self.assertAlmostEqual(self.table.loc["Cl1Na1", "e_hull_at_composition"], -2.2)

    def test_inconsistent_hull_energies_are_refused(self):
        # Two rows of one formula disagreeing means they came from two different
        # hulls, or were keyed wrongly. Either way the table would be nonsense.
        rows = _rows()
        rows.loc[0, "e_hull"] = 0.9
        with self.assertRaises(ValueError) as caught:
            ds.build_formula_table(rows)
        self.assertIn("not constant within a formula", str(caught.exception))

    def test_icsd_excess_is_the_test_of_assumption_c(self):
        # The ICSD-backed entry sits at -3.0 while Alexandria found -3.1, so the
        # experimentally observed structure is not the archive's ground state.
        self.assertTrue(self.table.loc["Ba1O3Ti1", "has_icsd"])
        self.assertAlmostEqual(self.table.loc["Ba1O3Ti1", "icsd_excess"], 0.1)
        self.assertEqual(self.table.loc["Ba1O3Ti1", "argmin_kind"], "agm")

    def test_formulas_without_experimental_backing_are_marked(self):
        self.assertFalse(self.table.loc["Cl1Na1", "has_icsd"])
        self.assertTrue(np.isnan(self.table.loc["Cl1Na1", "icsd_excess"]))

    def test_provenance_counts_are_per_process(self):
        row = self.table.loc["Ba1O3Ti1"]
        self.assertEqual((row["n_mp_icsd"], row["n_mp_theoretical"], row["n_oqmd"], row["n_agm"]),
                         (1, 1, 0, 1))

    def test_empty_input_is_refused(self):
        with self.assertRaises(ValueError):
            ds.build_formula_table(_rows().iloc[:0])


class TestAssignSplit(unittest.TestCase):
    def test_split_is_deterministic(self):
        keys = [f"X{i}Y{i}" for i in range(500)]
        self.assertTrue((ds.assign_split(keys) == ds.assign_split(keys)).all())

    def test_split_does_not_depend_on_the_other_keys(self):
        # A formula must land in the same fold whether or not the shallow world
        # contains its neighbours, or the shallow and deep runs are incomparable.
        keys = [f"X{i}Y{i}" for i in range(500)]
        whole = ds.assign_split(keys)
        half = ds.assign_split(keys[:100])
        self.assertTrue((whole.iloc[:100] == half).all())

    def test_salt_changes_the_split(self):
        keys = [f"X{i}Y{i}" for i in range(500)]
        self.assertFalse((ds.assign_split(keys) == ds.assign_split(keys, salt="other")).all())

    def test_proportions_are_about_right(self):
        keys = [f"X{i}Y{i}" for i in range(20_000)]
        split = ds.assign_split(keys, val_permille=50, test_permille=50)
        self.assertAlmostEqual((split == "val").mean(), 0.05, delta=0.01)
        self.assertAlmostEqual((split == "test").mean(), 0.05, delta=0.01)

    def test_nonsensical_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            ds.assign_split(["A1"], val_permille=600, test_permille=600)


if __name__ == "__main__":
    unittest.main()
