"""Tests for conditioning on the target chemical formula.

The representation has to be a faithful encoding of the composition -- if two
different formulas can collide, or the same formula can encode two ways, the
conditioning teaches the model something untrue -- and it has to reach the model
in the same layout at training and at sampling time.
"""
import math
import unittest
import unittest.mock

import torch

from wyckoff_transformer.composition import (
    COMPOSITION_FIELD,
    attach_composition_vector,
    composition_conditioning_dim,
    composition_vector,
    composition_vectors,
    describe,
)

N_ELEMENTS = 8


class TestCompositionVector(unittest.TestCase):
    def test_width_is_the_vocabulary_plus_the_size_channel(self):
        self.assertEqual(composition_conditioning_dim(N_ELEMENTS), N_ELEMENTS + 1)
        self.assertEqual(
            composition_vector([0], [1], N_ELEMENTS).shape, (N_ELEMENTS + 1,))

    def test_fractions_sum_to_one_and_size_is_log1p(self):
        vector = composition_vector([1, 3], [1, 3], N_ELEMENTS)
        self.assertAlmostEqual(vector[:N_ELEMENTS].sum().item(), 1.0, places=6)
        self.assertAlmostEqual(vector[1].item(), 0.25, places=6)
        self.assertAlmostEqual(vector[3].item(), 0.75, places=6)
        self.assertAlmostEqual(vector[N_ELEMENTS].item(), math.log1p(4), places=6)

    def test_absent_elements_are_zero(self):
        vector = composition_vector([2], [4], N_ELEMENTS)
        self.assertEqual(vector[:N_ELEMENTS].nonzero().reshape(-1).tolist(), [2])

    def test_fractions_are_z_invariant_and_the_size_channel_is_not(self):
        """BaTiO3 and Ba2Ti2O6 are the same chemistry in a cell of a different size."""
        one = composition_vector([0, 1, 2], [1, 1, 3], N_ELEMENTS)
        two = composition_vector([0, 1, 2], [2, 2, 6], N_ELEMENTS)
        self.assertTrue(torch.allclose(one[:N_ELEMENTS], two[:N_ELEMENTS]))
        self.assertGreater(two[N_ELEMENTS].item(), one[N_ELEMENTS].item())

    def test_different_formulas_do_not_collide(self):
        # The pair a conditioning signal most needs to separate: same elements,
        # different stoichiometry.
        rutile = composition_vector([0, 1], [1, 2], N_ELEMENTS)
        monoxide = composition_vector([0, 1], [1, 1], N_ELEMENTS)
        self.assertFalse(torch.allclose(rutile, monoxide))

    def test_counts_are_recoverable_from_the_vector(self):
        # Fractions plus log size are a bijection with the raw counts, so nothing
        # the model might need has been thrown away.
        counts = [2, 1, 5]
        vector = composition_vector([0, 3, 6], counts, N_ELEMENTS)
        total = math.expm1(vector[N_ELEMENTS].item())
        recovered = [round(vector[i].item() * total) for i in (0, 3, 6)]
        self.assertEqual(recovered, counts)

    def test_accepts_tensors_as_well_as_sequences(self):
        from_lists = composition_vector([1, 2], [3, 1], N_ELEMENTS)
        from_tensors = composition_vector(
            torch.tensor([1, 2]), torch.tensor([3.0, 1.0]), N_ELEMENTS)
        self.assertTrue(torch.allclose(from_lists, from_tensors))

    def test_a_repeated_element_sums(self):
        together = composition_vector([1], [4], N_ELEMENTS)
        apart = composition_vector([1, 1], [1, 3], N_ELEMENTS)
        self.assertTrue(torch.allclose(together, apart))

    def test_rejects_a_token_outside_the_vocabulary(self):
        with self.assertRaises(ValueError):
            composition_vector([N_ELEMENTS], [1], N_ELEMENTS)
        with self.assertRaises(ValueError):
            composition_vector([-1], [1], N_ELEMENTS)

    def test_rejects_non_positive_counts(self):
        with self.assertRaises(ValueError):
            composition_vector([0], [0], N_ELEMENTS)

    def test_rejects_empty_and_mismatched_input(self):
        with self.assertRaises(ValueError):
            composition_vector([], [], N_ELEMENTS)
        with self.assertRaises(ValueError):
            composition_vector([0, 1], [1], N_ELEMENTS)


class TestCompositionVectors(unittest.TestCase):
    def test_stacks_ragged_compositions_into_a_dense_block(self):
        # The tokeniser stores one ragged tensor per structure; this is where they
        # become something the conditioning path can index by row.
        block = composition_vectors(
            [[0], [1, 2], [3, 4, 5]], [[2], [1, 1], [1, 2, 3]], N_ELEMENTS)
        self.assertEqual(block.shape, (3, N_ELEMENTS + 1))
        self.assertTrue(torch.allclose(
            block[:, :N_ELEMENTS].sum(dim=1), torch.ones(3), atol=1e-6))

    def test_rejects_mismatched_or_empty_input(self):
        with self.assertRaises(ValueError):
            composition_vectors([[0]], [[1], [1]], N_ELEMENTS)
        with self.assertRaises(ValueError):
            composition_vectors([], [], N_ELEMENTS)


class TestAttachCompositionVector(unittest.TestCase):
    def _data(self):
        return {"composition_tokens": [torch.tensor([0, 1]), torch.tensor([2])],
                "composition_counts": [torch.tensor([1, 1]), torch.tensor([4])]}

    def test_adds_the_dense_field(self):
        data = attach_composition_vector(self._data(), N_ELEMENTS)
        self.assertEqual(data[COMPOSITION_FIELD].shape, (2, N_ELEMENTS + 1))

    def test_is_idempotent(self):
        data = attach_composition_vector(self._data(), N_ELEMENTS)
        first = data[COMPOSITION_FIELD]
        attach_composition_vector(data, N_ELEMENTS)
        self.assertIs(data[COMPOSITION_FIELD], first)

    def test_missing_counters_say_how_to_fix_it(self):
        with self.assertRaises(KeyError) as caught:
            attach_composition_vector({"elements": torch.zeros(2)}, N_ELEMENTS)
        self.assertIn("counters", str(caught.exception))


class _Tokeniser(dict):
    """Enough of an element tokeniser for the symbol-keyed lookups."""
    def __init__(self, symbols):
        super().__init__({symbol: index for index, symbol in enumerate(symbols)})
        self.to_token = {index: symbol for symbol, index in self.items()}


class TestDescribe(unittest.TestCase):
    def test_reads_a_vector_back_as_symbols(self):
        tokeniser = _Tokeniser(["Ba", "Ti", "O", "H"])
        vector = composition_vector([0, 1, 2], [1, 1, 3], len(tokeniser))
        rendered = " ".join(describe(vector, tokeniser))
        self.assertIn("Ba=0.200", rendered)
        self.assertIn("O=0.600", rendered)
        self.assertNotIn("H=", rendered)
        self.assertIn("log1p(atoms)", rendered)


class TestTrainerConditioningAssembly(unittest.TestCase):
    """`build_cond` is the single place the scalar and the composition are joined."""

    def _trainer(self, condition_feature, composition_conditioning):
        from wyckoff_transformer.trainer import WyckoffTrainer
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.condition_feature = condition_feature
        trainer.condition_transform = "log1p" if condition_feature else None
        trainer.composition_conditioning = composition_conditioning
        trainer.n_elements = N_ELEMENTS
        return trainer

    def _dataset(self):
        dataset = unittest.mock.MagicMock()
        dataset.data = {
            "energy_above_hull": torch.tensor([[0.0], [1.0]]),
            COMPOSITION_FIELD: composition_vectors(
                [[0], [1, 2]], [[4], [1, 1]], N_ELEMENTS),
        }
        return dataset

    def test_scalar_only_is_unchanged(self):
        cond = self._trainer("energy_above_hull", False).build_cond(self._dataset())
        self.assertEqual(cond.shape, (2, 1))
        # Still transformed on the way in, and still in log1p.
        self.assertAlmostEqual(cond[1, 0].item(), math.log1p(1.0), places=6)

    def test_composition_only(self):
        cond = self._trainer(None, True).build_cond(self._dataset())
        self.assertEqual(cond.shape, (2, N_ELEMENTS + 1))

    def test_scalar_comes_first_then_the_composition(self):
        # The order the model was built around; swapping it would silently make
        # every learned weight point at the wrong input.
        cond = self._trainer("energy_above_hull", True).build_cond(self._dataset())
        self.assertEqual(cond.shape, (2, N_ELEMENTS + 2))
        self.assertAlmostEqual(cond[1, 0].item(), math.log1p(1.0), places=6)
        self.assertAlmostEqual(cond[:, 1:1 + N_ELEMENTS].sum(dim=1)[0].item(), 1.0, places=6)

    def test_unconditional_builds_nothing(self):
        self.assertIsNone(self._trainer(None, False).build_cond(self._dataset()))

    def test_condition_dim_matches_what_build_cond_produces(self):
        for feature, composition in ((None, True), ("energy_above_hull", True),
                                     ("energy_above_hull", False)):
            trainer = self._trainer(feature, composition)
            with self.subTest(feature=feature, composition=composition):
                self.assertEqual(
                    trainer.condition_dim,
                    trainer.build_cond(self._dataset()).shape[-1])

    def test_batch_selection_is_honoured(self):
        cond = self._trainer("energy_above_hull", True).build_cond(
            self._dataset(), torch.tensor([1]))
        self.assertEqual(cond.shape, (1, N_ELEMENTS + 2))


if __name__ == "__main__":
    unittest.main()
