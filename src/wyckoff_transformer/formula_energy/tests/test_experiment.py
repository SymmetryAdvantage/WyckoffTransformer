"""Wiring test for the comparison run.

Not a test of whether the model works -- that needs the real archive. This checks
that all five comparators can be trained, predicted and scored against an answer
key without the shapes, indices or splits coming apart, which is the failure mode
that otherwise shows up an hour into a real run.
"""
import unittest

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.formula_energy import experiment
from wyckoff_transformer.formula_energy.tests.test_train import _synthetic

torch.set_num_threads(min(4, torch.get_num_threads()))


def _world(n=300):
    """A shallow table and a key in which the deep world found something."""
    table = _synthetic(n)
    table["chemsys"] = ["-".join(sorted(key[:2] for key in [formula[:2]])) for formula in table.index]
    table["split"] = np.where(np.arange(n) % 5 == 0, "test",
                              np.where(np.arange(n) % 5 == 1, "val", "train"))
    rng = np.random.default_rng(0)
    # A fifth of the formulas hide something below the shallow hull.
    deep = table["e_form_min"] - np.where(rng.uniform(size=n) < 0.2, 0.3, 0.0)
    key = pd.DataFrame({
        "shallow_e_form_min": table["e_form_min"],
        "shallow_e_hull": table["e_hull_at_composition"],
        "shallow_n_rows": table["n_rows"],
        "deep_e_form_min": deep,
        "deep_n_rows": table["n_rows"] * 2,
        "split": table["split"],
    })
    key["drop"] = key["shallow_e_form_min"] - key["deep_e_form_min"]
    key["discovered"] = key["deep_e_form_min"] < key["shallow_e_hull"]
    key["headroom"] = key["shallow_e_form_min"] - key["shallow_e_hull"]
    return table, key


class TestExperimentWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        table, key = _world()
        cls.frame = experiment.run(table, key, torch.device("cpu"), quick=True)
        cls.ranking = pd.DataFrame(cls.frame.attrs["ranking"])

    def test_every_floor_estimator_is_scored_under_both_rules(self):
        # Only models producing an energy on the hull's scale get a threshold row;
        # the censored fit and g_D also get the uncertainty-adjusted rule.
        self.assertEqual(
            set(self.frame["model"]),
            {"censored", "g_D", "g_C", "magpie+gbdt", "chemsys mean"},
        )
        adjusted = self.frame[self.frame["rule"] == "uncertainty-adjusted"]
        self.assertEqual(set(adjusted["model"]), {"censored", "g_D"})

    def test_the_headroom_signal_is_ranked_not_thresholded(self):
        # g_D - g_C is evidence that room exists, not an estimate of where the
        # floor is, so it appears in the ranking comparison and nowhere else.
        self.assertIn("g_D - g_C", set(self.ranking["model"]))
        self.assertNotIn("g_D - g_C", set(self.frame["model"]))

    def test_metrics_are_in_range(self):
        finite = self.frame.dropna(subset=["precision"])
        self.assertTrue(((finite["precision"] >= 0) & (finite["precision"] <= 1)).all())
        self.assertTrue(((finite["recall"] >= 0) & (finite["recall"] <= 1)).all())
        self.assertTrue((self.ranking["enrichment"] >= 0).all())

    def test_calibration_is_reported(self):
        self.assertTrue(0.0 <= self.frame.attrs["calibration"] <= 1.0)

    def test_prevalence_matches_the_key(self):
        self.assertAlmostEqual(self.frame["prevalence"].iloc[0], 0.2, delta=0.15)


if __name__ == "__main__":
    unittest.main()
