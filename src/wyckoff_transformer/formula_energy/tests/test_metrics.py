"""Tests for the screening metrics.

Enrichment against a random baseline is the number a campaign is judged on, and
the uncertainty-adjusted rule is what stops the ranking being led by its own
largest errors. Both have to behave correctly at the edges -- flagging nothing is
a real outcome, not a precision of zero.
"""
import math
import unittest

import numpy as np

from wyckoff_transformer.formula_energy import metrics as m


class TestScreeningMetrics(unittest.TestCase):
    def setUp(self):
        self.location = np.array([-1.0, -0.5, 0.0, 0.5])
        self.hull = np.zeros(4)
        self.discovered = np.array([True, True, False, False])

    def test_naive_rule(self):
        got = m.screening_metrics(self.location, self.hull, self.discovered)
        self.assertEqual(got["flagged"], 2)
        self.assertAlmostEqual(got["precision"], 1.0)
        self.assertAlmostEqual(got["recall"], 1.0)
        self.assertAlmostEqual(got["prevalence"], 0.5)
        self.assertAlmostEqual(got["enrichment"], 2.0)

    def test_uncertainty_adjusted_rule_is_stricter(self):
        # Wren's criterion: only flag where the floor plus its spread still clears
        # the hull. Trades recall for precision, which is the right trade when
        # each flag costs a structure search.
        sigma = np.array([0.2, 0.6, 0.1, 0.1])
        got = m.screening_metrics(self.location, self.hull, self.discovered, epistemic_sigma=sigma)
        self.assertEqual(got["flagged"], 1)
        self.assertAlmostEqual(got["precision"], 1.0)
        self.assertAlmostEqual(got["recall"], 0.5)

    def test_flagging_nothing_is_not_a_precision_of_zero(self):
        got = m.screening_metrics(self.location, self.hull, self.discovered, margin=10.0)
        self.assertEqual(got["flagged"], 0)
        self.assertTrue(math.isnan(got["precision"]))
        self.assertTrue(math.isnan(got["enrichment"]))

    def test_enrichment_below_one_means_worse_than_random(self):
        # A model that flags exactly the wrong half.
        got = m.screening_metrics(np.array([0.5, 0.5, -0.5, -0.5]), self.hull, self.discovered)
        self.assertAlmostEqual(got["precision"], 0.0)
        self.assertAlmostEqual(got["enrichment"], 0.0)

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            m.screening_metrics(self.location, self.hull[:2], self.discovered)

    def test_empty_input_is_refused(self):
        with self.assertRaises(ValueError):
            m.screening_metrics(np.array([]), np.array([]), np.array([], dtype=bool))


class TestProbabilityBelowHull(unittest.TestCase):
    def test_a_floor_on_the_hull_is_a_coin_flip(self):
        got = m.probability_below_hull(np.array([0.0]), np.array([0.1]), np.array([0.0]))
        self.assertAlmostEqual(got[0], 0.5)

    def test_more_spread_moves_a_confident_call_towards_a_half(self):
        far = m.probability_below_hull(np.array([-0.5]), np.array([0.01]), np.array([0.0]))[0]
        vague = m.probability_below_hull(np.array([-0.5]), np.array([1.0]), np.array([0.0]))[0]
        self.assertGreater(far, 0.99)
        self.assertLess(vague, far)
        self.assertGreater(vague, 0.5)

    def test_a_degenerate_ensemble_does_not_divide_by_zero(self):
        got = m.probability_below_hull(np.array([-1.0, 1.0]), np.zeros(2), np.zeros(2))
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertGreater(got[0], got[1])


class TestEnrichmentCurve(unittest.TestCase):
    def test_ranking_by_margin_finds_the_positives_first(self):
        score = np.array([-1.0, -0.9, 0.5, 0.6, 0.7, 0.8])
        discovered = np.array([True, True, False, False, False, False])
        curve = m.enrichment_curve(score, discovered, budgets=(2, 4))
        self.assertEqual(curve.loc[0, "hits"], 2)
        self.assertAlmostEqual(curve.loc[0, "precision"], 1.0)
        self.assertAlmostEqual(curve.loc[0, "enrichment"], 3.0)
        self.assertAlmostEqual(curve.loc[1, "precision"], 0.5)

    def test_budgets_beyond_the_candidate_pool_are_dropped(self):
        curve = m.enrichment_curve(np.zeros(3), np.array([True, False, False]), budgets=(2, 100))
        self.assertEqual(list(curve["budget"]), [2])


class TestCalibration(unittest.TestCase):
    def test_calibrated_probabilities_score_near_zero(self):
        rng = np.random.default_rng(0)
        probabilities = rng.uniform(size=200_000)
        outcomes = rng.uniform(size=200_000) < probabilities
        self.assertLess(m.expected_calibration_error(probabilities, outcomes), 0.01)

    def test_overconfidence_is_detected(self):
        # Claim 0.9 everywhere, be right half the time.
        rng = np.random.default_rng(1)
        probabilities = np.full(50_000, 0.9)
        outcomes = rng.uniform(size=50_000) < 0.5
        self.assertGreater(m.expected_calibration_error(probabilities, outcomes), 0.35)

    def test_table_covers_the_populated_bins(self):
        table = m.calibration_table(np.array([0.05, 0.15, 0.95]), np.array([False, True, True]), n_bins=10)
        self.assertEqual(len(table), 3)
        self.assertTrue((table["n"] == 1).all())


if __name__ == "__main__":
    unittest.main()
