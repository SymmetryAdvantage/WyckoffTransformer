"""Tests for the displacement bound.

This is the only lower bound in the scheme. Everything else says where the floor
probably is; this says how low it cannot be without contradicting a compound
somebody has actually made. The geometry is checkable by hand on a binary, which
is what these tests do.
"""
import unittest

import pandas as pd

from wyckoff_transformer.formula_energy import screen


def _reference():
    """A Na-Cl system whose hull is Na(0) - NaCl(-1 eV/atom) - Cl(0).

    Elemental references sit at zero total energy, so a total energy divided by
    the atom count is already a formation energy per atom and the arithmetic in
    the tests is visible.
    """
    return pd.DataFrame(
        {
            "full_formula": ["Na1", "Cl1", "Na1 Cl1"],
            "energy_corrected": [0.0, 0.0, -2.0],
            "chemsys": ["Na", "Cl", "Cl-Na"],
        },
        index=["na", "cl", "nacl"],
    )


class TestDisplacementBound(unittest.TestCase):
    def test_the_bound_is_where_the_geometry_says_it_is(self):
        # A candidate at NaCl3 sits at x_Cl = 0.75. The tie-line from Na through
        # it passes x_Cl = 0.5 at (0.5/0.75) * h, so NaCl at -1 eV/atom comes off
        # the hull exactly when h < -1.5.
        bound = screen.displacement_bound(_reference(), "Na1Cl3", protected_ids=["nacl"])
        self.assertAlmostEqual(bound, -1.5, delta=0.01)

    def test_a_candidate_above_the_bound_displaces_nothing(self):
        reference = _reference()
        bound = screen.displacement_bound(reference, "Na1Cl3", protected_ids=["nacl"])
        self.assertGreater(bound, -1.6)
        self.assertLess(bound, -1.4)

    def test_nothing_protected_means_no_bound(self):
        # Silence, not a constraint of zero: with nothing to falsify, the test
        # says nothing about the candidate.
        self.assertEqual(
            screen.displacement_bound(_reference(), "Na1Cl3", protected_ids=[]),
            float("-inf"),
        )

    def test_an_uncovered_chemistry_is_refused(self):
        with self.assertRaises(ValueError):
            screen.displacement_bound(_reference(), "K1Br1", protected_ids=["nacl"])

    def test_entries_for_chemsys_pulls_in_the_subsystems(self):
        # A binary diagram needs its elemental endpoints, not just entries of the
        # same arity.
        entries = screen.entries_for_chemsys(_reference(), frozenset({"Na", "Cl"}))
        self.assertEqual(len(entries), 3)
        self.assertEqual(len(screen.entries_for_chemsys(_reference(), frozenset({"Na"}))), 1)


class TestShortlist(unittest.TestCase):
    def setUp(self):
        self.prediction = pd.DataFrame(
            {"location": [-1.0, -1.0, 0.0], "sigma_epistemic": [0.01, 0.9, 0.01],
             "hull": [-0.5, -0.5, -0.5], "scale": [0.1, 0.1, 0.1], "target": [0.0, 0.0, 0.0]},
            index=["confident", "vague", "hopeless"],
        )

    def test_the_margin_demotes_the_uncertain_candidate(self):
        # Both estimate the same floor; only one of them means it.
        naive = screen.shortlist(self.prediction, margin=False)
        adjusted = screen.shortlist(self.prediction, margin=True)
        self.assertEqual(list(naive.index[:2]), ["confident", "vague"])
        self.assertEqual(adjusted.index[0], "confident")
        self.assertGreater(
            adjusted.loc["vague", "score"], adjusted.loc["confident", "score"]
        )

    def test_probabilities_track_confidence(self):
        frame = screen.shortlist(self.prediction)
        self.assertGreater(frame.loc["confident", "p_below_hull"], 0.99)
        self.assertLess(frame.loc["hopeless", "p_below_hull"], 0.01)

    def test_top_truncates(self):
        self.assertEqual(len(screen.shortlist(self.prediction, top=2)), 2)


class TestDisplacementFilter(unittest.TestCase):
    def test_a_prediction_below_the_bound_is_marked_inconsistent(self):
        candidates = pd.DataFrame(
            {"location": [-2.0, -1.0], "sigma_epistemic": [0.01, 0.01], "hull": [-0.5, -0.5]},
            index=["Na1Cl3", "Na1Cl3"],
        ).iloc[:1]
        out = screen.apply_displacement_filter(candidates, _reference(), protected_ids=["nacl"])
        # -2.0 is below the -1.5 the geometry allows, so the record says no.
        self.assertFalse(bool(out["consistent"].iloc[0]))
        self.assertAlmostEqual(out["displacement_bound"].iloc[0], -1.5, delta=0.01)
