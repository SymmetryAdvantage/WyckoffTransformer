"""Tests for the training loop.

The end-to-end property is the one that distinguishes this from a regression on
the archive minimum: fitted to bounds scattered above a floor, the model must put
its estimate *under* the bounds, and must attribute the scatter to the search
process that produced it rather than to the chemistry.
"""
import unittest

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.formula_energy import train as T
from wyckoff_transformer.formula_energy.features import PROVENANCE_FEATURES

# These models are tiny, and on a 48-core box the default thread pool spends more
# time synchronising than computing -- a 32-wide matmul does not want 48 threads.
torch.set_num_threads(min(4, torch.get_num_threads()))


def _synthetic(n_formulas=400, seed=0):
    """Binaries with a known floor, observed through a search of known depth.

    Half the formulas were searched hard (a tight bound) and half barely (a loose
    one). The floor depends only on chemistry; the looseness only on effort.
    """
    rng = np.random.default_rng(seed)
    elements = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Al",
                "Ga", "Si", "Ge", "Ti", "Zr", "Fe", "Ni", "Cu", "Zn", "O"]
    counts = 6
    # Only len(elements) * (len(elements) - 1) * counts distinct keys are
    # reachable; asking for more than that would spin here forever.
    if n_formulas > len(elements) * (len(elements) - 1) * counts:
        raise ValueError(f"Cannot draw {n_formulas} distinct formulas from this alphabet")
    rows, seen = [], set()
    while len(rows) < n_formulas:
        first, second = rng.choice(len(elements), size=2, replace=False)
        count = int(rng.integers(1, counts + 1))
        key = "".join(sorted([f"{elements[first]}1", f"{elements[second]}{count}"]))
        if key in seen:
            continue
        seen.add(key)
        deep = bool(len(rows) % 2)
        # The floor is a smooth function of composition; the excess is not.
        floor = -1.0 - 0.5 * (first / len(elements)) - 0.3 * (count / counts)
        excess = rng.exponential(0.02 if deep else 0.5)
        rows.append({
            "formula": key, "e_form_min": floor + excess, "true_floor": floor,
            "e_hull_at_composition": floor - 0.05, "deep": deep,
            "n_mp_icsd": 2 if deep else 0, "n_mp_theoretical": 0,
            "n_oqmd": 0, "n_agm": 1 if deep else 1,
            "n_rows": 30 if deep else 1, "n_cell_sizes": 5 if deep else 1,
            "n_icsd": 2 if deep else 0, "has_icsd": deep,
            "max_force_min": 0.0, "max_force_median": 0.001,
            "n_elements": 2, "icsd_excess": 0.0 if deep else np.nan,
            # What SystemDensity.attach adds: a denser neighbourhood for the
            # well-searched half, so the fixture mirrors the real table.
            "log1p_sys_entries_per_binary": 3.0 if deep else 0.5,
            "log1p_sys_entries_per_ternary": 0.0,
            "log1p_sys_hull_per_binary": 1.5 if deep else 0.2,
            "log1p_sys_hull_per_ternary": 0.0,
        })
    return pd.DataFrame(rows).set_index("formula")


class TestPrepare(unittest.TestCase):
    def test_tensors_line_up_with_the_table(self):
        table = _synthetic(40)
        data = T.prepare(table)
        self.assertEqual(len(data), len(table))
        self.assertEqual(data.provenance.shape, (len(table), len(PROVENANCE_FEATURES)))
        np.testing.assert_allclose(data.target.numpy(), table["e_form_min"].to_numpy(), rtol=1e-6)

    def test_dropping_a_provenance_feature_zeroes_it(self):
        data = T.prepare(_synthetic(40), drop_provenance=["has_icsd"])
        self.assertEqual(data.provenance[:, PROVENANCE_FEATURES.index("has_icsd")].abs().max().item(), 0.0)

    def test_unknown_provenance_feature_is_refused(self):
        with self.assertRaises(ValueError):
            T.prepare(_synthetic(40), drop_provenance=["how_famous_the_element_is"])

    def test_index_and_to_preserve_length(self):
        data = T.prepare(_synthetic(40))
        selected = data.index(torch.arange(10))
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(data.to(torch.device("cpu"))), 40)


class TestRecoversTheFloor(unittest.TestCase):
    """The real-data analogue of ``test_censored.TestRecoversTheMinimum``."""

    @classmethod
    def setUpClass(cls):
        table = _synthetic(400)
        cls.table = table
        config = T.TrainConfig(
            d_model=32, n_layers=2, n_heads=4, dim_feedforward=64, head_widths=(32, 16),
            dropout=0.0, epochs=60, patience=60, batch_size=64, learning_rate=3e-3, seed=0,
        )
        data = T.prepare(table)
        device = torch.device("cpu")
        cls.model, cls.history = T.train_one(data.to(device), data.to(device), config, device)
        cls.prediction = T.predict([cls.model], data.to(device))

    def test_the_floor_lands_below_the_bound(self):
        # An MSE fit would land on the mean of the observations. The censored fit
        # must sit under them, which is what "excess > 0" says.
        excess = self.prediction["target"] - self.prediction["location"]
        self.assertGreater(excess.mean(), 0.05)

    def test_the_floor_is_closer_to_the_truth_than_the_bound_is(self):
        truth = self.table["true_floor"].to_numpy()
        model_error = np.abs(self.prediction["location"].to_numpy() - truth).mean()
        bound_error = np.abs(self.prediction["target"].to_numpy() - truth).mean()
        self.assertLess(model_error, bound_error)

    def test_the_excess_is_attributed_to_the_search_not_the_chemistry(self):
        # Formulas that were barely searched must get the wider excess scale.
        deep = self.table["deep"].to_numpy()
        self.assertGreater(
            self.prediction["scale"].to_numpy()[~deep].mean(),
            self.prediction["scale"].to_numpy()[deep].mean(),
        )

    def test_training_reduced_the_loss(self):
        self.assertLess(self.history[-1]["nll"], self.history[0]["nll"])


class TestEnsembleAndPersistence(unittest.TestCase):
    def setUp(self):
        self.data = T.prepare(_synthetic(60))
        self.config = T.TrainConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32,
                                    head_widths=(16,), epochs=2, patience=2, batch_size=32)

    def test_members_disagree_so_the_spread_is_usable(self):
        models, _ = T.train_ensemble(self.data, self.data, self.config, torch.device("cpu"), n_models=3)
        prediction = T.predict(models, self.data)
        self.assertEqual(len(prediction), len(self.data))
        self.assertGreater(prediction["sigma_epistemic"].mean(), 0.0)

    def test_a_single_model_has_no_epistemic_spread(self):
        model, _ = T.train_one(self.data, self.data, self.config, torch.device("cpu"))
        prediction = T.predict([model], self.data)
        self.assertTrue(np.isfinite(prediction["sigma_epistemic"]).all())
        self.assertEqual(prediction["sigma_epistemic"].abs().max(), 0.0)

    def test_predicting_with_no_models_is_refused(self):
        with self.assertRaises(ValueError):
            T.predict([], self.data)

    def test_round_trip_through_disk(self):
        import tempfile
        from pathlib import Path

        models, _ = T.train_ensemble(self.data, self.data, self.config, torch.device("cpu"), n_models=2)
        before = T.predict(models, self.data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            T.save_ensemble(models, path, self.config)
            restored, config, names = T.load_ensemble(path, torch.device("cpu"))
        self.assertEqual(config.d_model, self.config.d_model)
        # The feature list travels with the checkpoint so a model cannot be
        # handed a provenance vector of the wrong width or order.
        self.assertEqual(names, list(PROVENANCE_FEATURES))
        pd.testing.assert_frame_equal(before, T.predict(restored, self.data))


if __name__ == "__main__":
    unittest.main()
