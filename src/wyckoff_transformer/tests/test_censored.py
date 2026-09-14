"""Tests for the censored likelihood that regresses min(E | gene).

The property that matters is the one that distinguishes it from MSE: fitted to
observations scattered above a floor, it must find the floor and not the mean.
"""
import math
import unittest

import torch

from wyckoff_transformer.censored import (
    CensoredMinDiagnostics,
    CensoredMinLoss,
    censored_min_mae,
    censored_min_nll,
    expected_min_bias,
    gene_level_polymorph_delta,
)


def _fit(observations, steps=3000, noise=1e-3):
    """Fit a single (location, log_scale) pair by gradient descent."""
    location = torch.zeros(1, requires_grad=True)
    log_scale = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([location, log_scale], lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad()
        censored_min_nll(observations, location, log_scale, noise=noise).mean().backward()
        optimizer.step()
    return location.item(), log_scale.exp().item()


class TestCensoredMinNLL(unittest.TestCase):
    def test_density_normalises(self):
        # The NLL must be a proper log density, or the fitted scale means nothing.
        grid = torch.linspace(-6, 8, 200_001, dtype=torch.float64)
        location = torch.full_like(grid, -1.5)
        log_scale = torch.full_like(grid, math.log(0.3))
        density = torch.exp(-censored_min_nll(grid, location, log_scale, noise=0.05))
        self.assertAlmostEqual(torch.trapz(density, grid).item(), 1.0, places=5)

    def test_mean_is_location_plus_scale(self):
        grid = torch.linspace(-6, 8, 200_001, dtype=torch.float64)
        location = torch.full_like(grid, -1.5)
        log_scale = torch.full_like(grid, math.log(0.3))
        density = torch.exp(-censored_min_nll(grid, location, log_scale, noise=0.05))
        self.assertAlmostEqual(torch.trapz(density * grid, grid).item(), -1.2, places=4)

    def test_collapses_to_exponential_as_noise_vanishes(self):
        location, scale = -1.5, 0.3
        observed = torch.tensor([location + 0.2, location + 1.0], dtype=torch.float64)
        got = censored_min_nll(observed, torch.tensor(location, dtype=torch.float64),
                               torch.tensor(math.log(scale), dtype=torch.float64), noise=1e-7)
        want = math.log(scale) + (observed - location) / scale
        self.assertLess((got - want).abs().max().item(), 1e-9)

    def test_prediction_above_observation_is_expensive_but_finite(self):
        # A hard constraint would give an infinite loss and no usable gradient.
        nll = censored_min_nll(
            torch.zeros(3), torch.tensor([0.05, 0.5, 5.0]),
            torch.full((3,), math.log(0.3)), noise=0.01)
        self.assertTrue(torch.isfinite(nll).all())
        self.assertTrue((nll.diff() > 0).all())
        self.assertGreater(nll[-1].item(), 100.)

    def test_gradients_are_finite_far_from_the_optimum(self):
        location = torch.tensor([-50.0, 0.0, 50.0], requires_grad=True)
        log_scale = torch.zeros(3, requires_grad=True)
        censored_min_nll(torch.zeros(3), location, log_scale, noise=0.01).sum().backward()
        self.assertTrue(torch.isfinite(location.grad).all())
        self.assertTrue(torch.isfinite(log_scale.grad).all())

    def test_rejects_non_positive_noise(self):
        with self.assertRaises(ValueError):
            censored_min_nll(torch.zeros(1), torch.zeros(1), torch.zeros(1), noise=0.0)


class TestRecoversTheMinimum(unittest.TestCase):
    """The whole point: labels are upper bounds, and the fit must respect that."""

    def test_finds_the_floor_not_the_mean(self):
        torch.manual_seed(0)
        true_min, true_scale = -1.5, 0.3
        observations = true_min + torch.distributions.Exponential(1 / true_scale).sample((200,))
        location, scale = _fit(observations)
        self.assertAlmostEqual(location, true_min, places=2)
        self.assertAlmostEqual(scale, true_scale, delta=0.05)
        # An MSE fit would land on the mean, a whole scale higher.
        self.assertGreater(observations.mean().item() - location, 0.5 * true_scale)

    def test_a_gene_seen_more_often_is_pinned_more_tightly(self):
        """The frequency handling, which is what makes this better than min-of-observed.

        The gap for any one gene is a random variable; only its mean shrinks as
        ``s / n``, so this averages over replicates rather than trusting a draw.
        """
        torch.manual_seed(1)
        true_min, true_scale, replicates = 0.0, 0.4, 12
        gaps = {}
        for n in (1, 4, 64):
            drawn = [_fit(true_min + torch.distributions.Exponential(1 / true_scale).sample((n,)),
                          steps=1500)[0] - true_min
                     for _ in range(replicates)]
            gaps[n] = sum(drawn) / replicates
        self.assertGreater(gaps[1], gaps[4])
        self.assertGreater(gaps[4], gaps[64])
        # Only the ordering is asserted. The gap does not match `expected_min_bias`
        # here: at n = 1 the isolated location-scale fit is the degenerate case the
        # module documents, and where it lands depends on the optimiser rather than
        # on the likelihood. The s/n law itself is checked against the order
        # statistic directly, in test_expected_min_bias_matches_the_order_statistic.

    def test_expected_min_bias_matches_the_order_statistic(self):
        torch.manual_seed(2)
        scale = 0.3
        for n in (1, 4, 16):
            draws = torch.distributions.Exponential(1 / scale).sample((20000, n))
            self.assertAlmostEqual(
                draws.min(dim=1).values.mean().item(),
                expected_min_bias(scale, n).item(), delta=0.01)

    def test_expected_min_bias_rejects_zero_samples(self):
        with self.assertRaises(ValueError):
            expected_min_bias(0.3, 0)


class TestCensoredMinLoss(unittest.TestCase):
    def test_two_column_output_splits_into_location_and_scale(self):
        criterion = CensoredMinLoss()
        self.assertEqual(criterion.n_outputs, 2)
        prediction = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        location, log_scale = criterion.split(prediction)
        self.assertTrue(torch.equal(location, torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.equal(log_scale, torch.tensor([-2.0, -4.0])))

    def test_a_batch_of_one_keeps_both_columns(self):
        # A bare squeeze() would turn [1, 2] into [2] and read the scale as a location.
        criterion = CensoredMinLoss()
        loss = criterion(torch.tensor([[0.5, -1.0]]), torch.tensor([1.0]))
        self.assertTrue(torch.isfinite(loss))

    def test_global_scale_is_a_trainable_parameter(self):
        criterion = CensoredMinLoss(predict_scale=False, init_scale=0.2)
        self.assertEqual(criterion.n_outputs, 1)
        self.assertEqual([p.numel() for p in criterion.parameters()], [1])
        criterion(torch.tensor([0.0, 0.1]), torch.tensor([1.0, 1.2])).backward()
        self.assertIsNotNone(criterion.global_log_scale.grad)

    def test_predicted_scale_needs_two_columns(self):
        with self.assertRaises(ValueError):
            CensoredMinLoss().split(torch.tensor([1.0, 2.0, 3.0]))

    def test_scale_floor_is_enforced(self):
        criterion = CensoredMinLoss(min_scale=1e-2)
        pinned = criterion(torch.tensor([[0.0, -50.0]]), torch.tensor([0.5]))
        floored = criterion(torch.tensor([[0.0, math.log(1e-2)]]), torch.tensor([0.5]))
        self.assertAlmostEqual(pinned.item(), floored.item(), places=5)

    def test_reductions(self):
        prediction = torch.tensor([[0.0, -1.0], [0.0, -1.0]])
        target = torch.tensor([1.0, 2.0])
        none = CensoredMinLoss(reduction="none")(prediction, target)
        self.assertEqual(none.shape, (2,))
        self.assertAlmostEqual(
            CensoredMinLoss(reduction="sum")(prediction, target).item(), none.sum().item(), 5)
        self.assertAlmostEqual(
            CensoredMinLoss(reduction="mean")(prediction, target).item(), none.mean().item(), 5)

    def test_rejects_unknown_reduction(self):
        with self.assertRaises(ValueError):
            CensoredMinLoss(reduction="median")


class TestDiagnostics(unittest.TestCase):
    def test_violation_rate_catches_a_location_pulled_to_the_mean(self):
        torch.manual_seed(3)
        observations = torch.distributions.Exponential(1 / 0.3).sample((500,))
        diagnostics = CensoredMinDiagnostics(CensoredMinLoss())
        at_floor = torch.stack(
            [torch.zeros(500), torch.full((500,), math.log(0.3))], dim=-1)
        at_mean = torch.stack(
            [torch.full((500,), observations.mean().item()), torch.full((500,), math.log(0.3))],
            dim=-1)
        self.assertLess(diagnostics(at_floor, observations)["violation"].item(), 0.01)
        self.assertGreater(diagnostics(at_mean, observations)["violation"].item(), 0.5)

    def test_excess_matches_the_scale_when_calibrated(self):
        torch.manual_seed(4)
        scale = 0.3
        observations = torch.distributions.Exponential(1 / scale).sample((20000,))
        diagnostics = CensoredMinDiagnostics(CensoredMinLoss())
        prediction = torch.stack(
            [torch.zeros(20000), torch.full((20000,), math.log(scale))], dim=-1)
        reported = diagnostics(prediction, observations)
        self.assertAlmostEqual(reported["excess"].item(), reported["scale"].item(), delta=0.02)

    def test_mae_reads_the_location_column(self):
        prediction = torch.tensor([[1.0, -3.0], [2.0, -3.0]])
        target = torch.tensor([1.5, 2.5])
        self.assertAlmostEqual(
            censored_min_mae(prediction, target, CensoredMinLoss()).item(), 0.5, places=6)


class TestGeneLevelPolymorphDelta(unittest.TestCase):
    """The conditioning label, assigned per gene rather than per structure."""

    def setUp(self):
        # NaCl has two genes, g1 seen twice; KBr has one gene and so is unopposed.
        self.frame = gene_level_polymorph_delta(
            gene_ids=["g1", "g1", "g2", "g3"],
            energies=[-1.0, -0.7, -0.4, -2.0],
            composition_ids=["NaCl", "NaCl", "NaCl", "KBr"])

    def test_gene_min_is_the_observed_minimum(self):
        self.assertEqual(list(self.frame["gene_min"]), [-1.0, -1.0, -0.4, -2.0])

    def test_structures_sharing_a_gene_share_one_label(self):
        # The point: the model sees one input for these two rows, so it must see
        # one target. A per-structure label would give 0.0 and 0.3.
        self.assertEqual(self.frame["delta_e_polymorph"][0],
                         self.frame["delta_e_polymorph"][1])

    def test_best_gene_of_a_composition_is_zero(self):
        self.assertEqual(self.frame["delta_e_polymorph"][0], 0.0)
        self.assertAlmostEqual(self.frame["delta_e_polymorph"][2], 0.6)

    def test_unopposed_composition_is_flagged(self):
        # KBr's zero says only that nothing better was seen, not that it is a
        # ground state, and polymorph_count is what distinguishes the two.
        self.assertEqual(self.frame["delta_e_polymorph"][3], 0.0)
        self.assertEqual(self.frame["polymorph_count"][3], 1)
        self.assertEqual(self.frame["polymorph_count"][0], 2)

    def test_gene_count_records_how_tight_the_bound_is(self):
        self.assertEqual(list(self.frame["gene_count"]), [2, 2, 1, 1])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            gene_level_polymorph_delta([], [], [])


if __name__ == "__main__":
    unittest.main()
