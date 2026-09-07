"""Tests for the two-head composition model.

The property that has to hold is the exclusion restriction: the estimate of the
floor is a function of chemistry and of nothing else. If provenance can reach the
location head, the model learns that formulas nobody has tried have high energy
-- true of this archive, and exactly backwards for screening.

The rest is representation hygiene: a compound is a multiset of elements with
proportions, so the encoder must not care about element order, about whether the
formula was written per cell or per formula unit, or about how much padding it
was given.
"""
import unittest

import torch

from wyckoff_transformer.censored import CensoredMinLoss
from wyckoff_transformer.formula_energy.encoder import CompositionEncoder, FormulaEnergyModel
from wyckoff_transformer.formula_energy.features import PROVENANCE_FEATURES, composition_tensors


def _model(**kwargs):
    torch.manual_seed(0)
    model = FormulaEnergyModel(
        n_provenance=len(PROVENANCE_FEATURES), d_model=32, n_layers=2,
        n_heads=4, dim_feedforward=64, head_widths=(16, 8), **kwargs,
    )
    return model.eval()


def _batch(formulas=("Ba1O3Ti1", "Cl1Na1", "As1H16Li3O8S4")):
    ids, fractions, mask = composition_tensors(formulas)
    provenance = torch.zeros(len(formulas), len(PROVENANCE_FEATURES))
    return ids, fractions, mask, provenance


class TestExclusionRestriction(unittest.TestCase):
    def test_provenance_cannot_move_the_floor(self):
        # The whole design in one assertion: change how hard anyone looked, and
        # the estimate of what is down there must not move.
        model = _model()
        ids, fractions, mask, provenance = _batch()
        quiet = model(ids, fractions, mask, provenance)
        busy = model(ids, fractions, mask, torch.randn_like(provenance) * 5)
        torch.testing.assert_close(quiet[:, 0], busy[:, 0])
        # ...while the excess scale must move, or the channel is doing nothing.
        self.assertFalse(torch.allclose(quiet[:, 1], busy[:, 1]))

    def test_a_named_feature_reaches_the_floor_and_the_others_still_cannot(self):
        # The relaxation is deliberate and scoped: naming one column lets the
        # floor read that column and nothing else.
        torch.manual_seed(0)
        model = FormulaEnergyModel(
            n_provenance=len(PROVENANCE_FEATURES), d_model=32, n_layers=2, n_heads=4,
            dim_feedforward=64, head_widths=(16, 8), location_feature_indices=[0],
        ).eval()
        ids, fractions, mask, provenance = _batch()
        base = model(ids, fractions, mask, provenance)[:, 0]

        moved_named = provenance.clone(); moved_named[:, 0] = 3.0
        self.assertFalse(torch.allclose(base, model(ids, fractions, mask, moved_named)[:, 0]))

        moved_other = provenance.clone(); moved_other[:, 1:] = 3.0
        torch.testing.assert_close(base, model(ids, fractions, mask, moved_other)[:, 0])

    def test_a_floor_that_reads_provenance_demands_it(self):
        torch.manual_seed(0)
        model = FormulaEnergyModel(
            n_provenance=len(PROVENANCE_FEATURES), d_model=32, n_layers=2, n_heads=4,
            dim_feedforward=64, head_widths=(16, 8), location_feature_indices=[0],
        ).eval()
        ids, fractions, mask, _ = _batch()
        with self.assertRaises(ValueError):
            model.predict_floor(ids, fractions, mask)

    def test_an_out_of_range_index_is_refused(self):
        with self.assertRaises(ValueError):
            FormulaEnergyModel(n_provenance=3, d_model=32, n_layers=1, n_heads=4,
                               dim_feedforward=32, head_widths=(8,),
                               location_feature_indices=[7])

    def test_predict_floor_needs_no_provenance(self):
        # A formula nobody has computed has none to supply.
        model = _model()
        ids, fractions, mask, provenance = _batch()
        torch.testing.assert_close(
            model.predict_floor(ids, fractions, mask), model(ids, fractions, mask, provenance)[:, 0]
        )

    def test_detached_scale_head_leaves_the_trunk_alone(self):
        model = _model(detach_scale_trunk=True)
        ids, fractions, mask, provenance = _batch()
        model(ids, fractions, mask, provenance)[:, 1].sum().backward()
        grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
        self.assertTrue(all(grad.abs().max() == 0 for grad in grads) or not grads)

    def test_wrong_provenance_width_is_refused(self):
        model = _model()
        ids, fractions, mask, _ = _batch()
        with self.assertRaises(ValueError):
            model(ids, fractions, mask, torch.zeros(3, 2))


class TestRepresentationInvariances(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder = CompositionEncoder(d_model=32, n_layers=2, n_heads=4, dim_feedforward=64).eval()

    def test_element_order_does_not_matter(self):
        # No positional encoding, and pooling is a weighted sum, so the encoder
        # sees a multiset. Built by hand because composition_tensors always sorts.
        ids = torch.tensor([[56, 8, 22], [22, 56, 8]])
        fractions = torch.tensor([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        mask = torch.zeros(2, 3, dtype=torch.bool)
        trunk = self.encoder(ids, fractions, mask)
        torch.testing.assert_close(trunk[0], trunk[1])

    def test_cell_multiples_are_identical(self):
        # BaTiO3 and Ba2Ti2O6 are one compound with one floor.
        one, two = composition_tensors(["Ba1O3Ti1"]), composition_tensors(["Ba2O6Ti2"])
        torch.testing.assert_close(self.encoder(*one), self.encoder(*two))

    def test_padding_is_ignored(self):
        # A binary encoded in a batch of quaternaries must give what it gives alone.
        narrow = composition_tensors(["Cl1Na1"])
        wide = composition_tensors(["Cl1Na1"], max_elements=5)
        torch.testing.assert_close(self.encoder(*narrow), self.encoder(*wide))

    def test_a_formula_wider_than_the_pad_is_refused(self):
        with self.assertRaises(ValueError):
            composition_tensors(["As1H16Li3O8S4"], max_elements=2)


class TestFitsTheCensoredLoss(unittest.TestCase):
    def test_output_is_the_shape_the_loss_splits(self):
        model, criterion = _model(), CensoredMinLoss()
        ids, fractions, mask, provenance = _batch()
        prediction = model(ids, fractions, mask, provenance)
        self.assertEqual(prediction.shape, (3, criterion.n_outputs))
        location, log_scale = criterion.split(prediction)
        torch.testing.assert_close(location, prediction[:, 0])
        torch.testing.assert_close(log_scale, prediction[:, 1])

    def test_loss_is_finite_and_backpropagates(self):
        model, criterion = _model(), CensoredMinLoss()
        model.train()
        ids, fractions, mask, provenance = _batch()
        loss = criterion(model(ids, fractions, mask, provenance), torch.tensor([-3.1, -2.2, -1.0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_location_bias_starts_where_it_is_told(self):
        # Targets are raw eV/atom, so the head has to start near the data mean
        # rather than at zero; the loss constants are in those units too.
        model = _model(location_bias=-1.25)
        ids, fractions, mask, provenance = _batch()
        self.assertLess(model(ids, fractions, mask, provenance)[:, 0].abs().max(), 5.0)
        self.assertAlmostEqual(model.location_head[-1].bias.item(), -1.25)


class TestFractionalEncoder(unittest.TestCase):
    def test_nearby_fractions_are_nearby_vectors(self):
        from wyckoff_transformer.formula_energy.encoder import FractionalEncoder

        encoder = FractionalEncoder(32)
        values = encoder(torch.tensor([[0.20, 0.21, 0.90]]))
        near = (values[0, 0] - values[0, 1]).norm()
        far = (values[0, 0] - values[0, 2]).norm()
        self.assertLess(near, far)

    def test_width_must_divide(self):
        from wyckoff_transformer.formula_energy.encoder import FractionalEncoder

        with self.assertRaises(ValueError):
            FractionalEncoder(30)


if __name__ == "__main__":
    unittest.main()


class TestSystemDensity(unittest.TestCase):
    """The arity control is the whole point of the combinatoric normalisation."""

    def _density(self):
        from wyckoff_transformer.formula_energy.features import SystemDensity

        # Every binary subsystem holds ten entries, two of them on the hull.
        entries = {frozenset(pair): 10 for pair in (("Na", "Cl"), ("Na", "O"), ("Cl", "O"))}
        hull = {frozenset(pair): 2 for pair in (("Na", "Cl"), ("Na", "O"), ("Cl", "O"))}
        return SystemDensity(entries, hull)

    def test_the_same_density_scores_the_same_at_any_arity(self):
        # A ternary contains C(3,2) = 3 binary subsystems holding 30 entries
        # between them; a binary contains one holding 10. Both are ten entries
        # per binary subsystem, and the feature must say so.
        import math

        columns = self._density().columns(["Cl1Na1O1", "Cl1Na1"])
        per_binary = columns["log1p_sys_entries_per_binary"]
        self.assertAlmostEqual(per_binary.iloc[0], math.log1p(10.0), places=6)
        self.assertAlmostEqual(per_binary.iloc[1], math.log1p(10.0), places=6)

    def test_a_raw_count_would_not_have(self):
        # Stated explicitly because this is what the normalisation buys: the raw
        # totals differ threefold for the same underlying density.
        raw_ternary = sum(10 for _ in range(3))
        raw_binary = 10
        self.assertNotEqual(raw_ternary, raw_binary)

    def test_subsystems_larger_than_the_system_score_zero(self):
        columns = self._density().columns(["Cl1Na1"])
        self.assertEqual(columns["log1p_sys_entries_per_ternary"].iloc[0], 0.0)

    def test_hull_entries_are_counted_separately(self):
        import math

        columns = self._density().columns(["Cl1Na1O1"])
        self.assertAlmostEqual(columns["log1p_sys_hull_per_binary"].iloc[0], math.log1p(2.0), places=6)

    def test_an_unseen_system_is_empty_not_missing(self):
        columns = self._density().columns(["K1Br1"])
        self.assertTrue((columns.to_numpy() == 0.0).all())
