"""Tests for the shallow-world answer key.

Two things have to be right or the evaluation is measuring noise. A discovery is
a structure that falls below the hull the *shallow* world knew about, and the two
worlds have to be on one energy scale before that comparison means anything --
withholding a source raises the elemental references, and a formation energy is
measured against them.
"""
import unittest

import numpy as np
import pandas as pd

from wyckoff_transformer.formula_energy import answer_key as ak


def _tables(deep_min=-1.10):
    shallow = pd.DataFrame(
        {"e_form_min": [-1.00, -0.50], "e_hull_at_composition": [-1.05, -0.50],
         "n_rows": [3, 1], "split": ["test", "test"]},
        index=["Cl1Na1", "Ba1O3Ti1"],
    )
    deep = pd.DataFrame(
        {"e_form_min": [deep_min, -0.50], "n_rows": [9, 4]},
        index=["Cl1Na1", "Ba1O3Ti1"],
    )
    return deep, shallow


class TestFrameShift(unittest.TestCase):
    def test_shift_is_the_fraction_weighted_reference_move(self):
        delta = pd.Series({"Na": -0.01, "Cl": -0.04})
        np.testing.assert_allclose(ak.frame_shift(["Cl1Na1"], delta), [-0.025])

    def test_elements_with_no_recorded_move_contribute_nothing(self):
        delta = pd.Series({"Cl": -0.04})
        np.testing.assert_allclose(ak.frame_shift(["Cl1Na1"], delta), [-0.02])

    def test_shift_respects_stoichiometry(self):
        # Three quarters chlorine, so three quarters of chlorine's move.
        delta = pd.Series({"Na": 0.0, "Cl": -0.04})
        np.testing.assert_allclose(ak.frame_shift(["Cl3Na1"], delta), [-0.03])


class TestBuildAnswerKey(unittest.TestCase):
    def test_a_structure_below_the_shallow_hull_is_a_discovery(self):
        key = ak.build_answer_key(*_tables())
        self.assertTrue(bool(key.loc["Cl1Na1", "discovered"]))
        self.assertFalse(bool(key.loc["Ba1O3Ti1", "discovered"]))

    def test_the_drop_is_how_far_the_withheld_search_lowered_the_floor(self):
        key = ak.build_answer_key(*_tables())
        self.assertAlmostEqual(key.loc["Cl1Na1", "drop"], 0.10)
        self.assertAlmostEqual(key.loc["Cl1Na1", "headroom"], 0.05)

    def test_the_frame_shift_can_overturn_a_discovery(self):
        # Deep at -1.10 clears a hull at -1.05, but only by 50 meV. Put the two
        # worlds on one scale with a 100 meV move and it no longer does.
        delta = pd.Series({"Na": 0.10, "Cl": 0.10})
        key = ak.build_answer_key(*_tables(), delta=delta)
        self.assertAlmostEqual(key.loc["Cl1Na1", "deep_e_form_min"], -1.00)
        self.assertFalse(bool(key.loc["Cl1Na1", "discovered"]))

    def test_worlds_that_share_nothing_are_refused(self):
        deep, shallow = _tables()
        with self.assertRaises(ValueError):
            ak.build_answer_key(deep, shallow.rename(index={"Cl1Na1": "X1", "Ba1O3Ti1": "Y1"}))

    def test_describe_reports_the_impossible_case(self):
        # A negative drop means the two worlds are still on different scales.
        summary = ak.describe(ak.build_answer_key(*_tables(deep_min=-0.90)))
        self.assertGreater(summary["drop_negative"], 0.0)
        self.assertEqual(summary["formulas"], 2)


if __name__ == "__main__":
    unittest.main()
