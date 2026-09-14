"""Tests for conditioning on the chemical system.

Two properties carry this mode. The vector has to be the indicator of a *set* --
if a count can leak into it the relaxed mode has quietly become the strict one --
and it has to reach the model in the same layout at training and at sampling
time, in the same slot the composition would have occupied.
"""
import unittest
import unittest.mock

import torch

from wyckoff_transformer.chemical_system import (
    CHEMICAL_SYSTEM_FIELD,
    attach_chemical_system_vector,
    chemical_system_conditioning_dim,
    chemical_system_vector,
    chemical_system_vectors,
    describe,
    parse_chemical_system,
)

N_ELEMENTS = 8


class TestChemicalSystemVector(unittest.TestCase):
    def test_width_is_the_vocabulary(self):
        self.assertEqual(chemical_system_conditioning_dim(N_ELEMENTS), N_ELEMENTS)
        self.assertEqual(chemical_system_vector([0], N_ELEMENTS).shape, (N_ELEMENTS,))

    def test_present_elements_are_one_and_the_rest_zero(self):
        vector = chemical_system_vector([1, 3], N_ELEMENTS)
        self.assertEqual(vector.nonzero().reshape(-1).tolist(), [1, 3])
        self.assertEqual(vector[1].item(), 1.0)
        self.assertEqual(vector[3].item(), 1.0)

    def test_the_norm_is_the_arity(self):
        # The model is told how many elements it is working with, which is the one
        # count the set representation does keep.
        self.assertEqual(chemical_system_vector([0, 2, 5], N_ELEMENTS).sum().item(), 3.0)

    def test_counts_do_not_reach_the_vector(self):
        """BaTiO3, Ba2TiO4 and TiO2Ba are one chemical system, and encode identically."""
        once = chemical_system_vector([0, 1, 2], N_ELEMENTS)
        repeated = chemical_system_vector([0, 1, 1, 2, 2, 2], N_ELEMENTS)
        reordered = chemical_system_vector([2, 0, 1], N_ELEMENTS)
        self.assertTrue(torch.equal(once, repeated))
        self.assertTrue(torch.equal(once, reordered))

    def test_different_systems_do_not_collide(self):
        self.assertFalse(torch.equal(
            chemical_system_vector([0, 1], N_ELEMENTS),
            chemical_system_vector([0, 2], N_ELEMENTS)))

    def test_a_subsystem_is_a_different_input(self):
        # Ba-Ti-O is not Ba-O: the conditioning names the elements the structure has,
        # so asking for the larger system is asking for a compound of all of them.
        self.assertFalse(torch.equal(
            chemical_system_vector([0, 1, 2], N_ELEMENTS),
            chemical_system_vector([0, 2], N_ELEMENTS)))

    def test_accepts_tensors_as_well_as_sequences(self):
        self.assertTrue(torch.equal(
            chemical_system_vector([1, 2], N_ELEMENTS),
            chemical_system_vector(torch.tensor([1, 2]), N_ELEMENTS)))

    def test_rejects_a_token_outside_the_vocabulary(self):
        with self.assertRaises(ValueError):
            chemical_system_vector([N_ELEMENTS], N_ELEMENTS)
        with self.assertRaises(ValueError):
            chemical_system_vector([-1], N_ELEMENTS)

    def test_rejects_an_empty_system(self):
        with self.assertRaises(ValueError):
            chemical_system_vector([], N_ELEMENTS)


class TestChemicalSystemVectors(unittest.TestCase):
    def test_stacks_ragged_systems_into_a_dense_block(self):
        block = chemical_system_vectors([[0], [1, 2], [3, 4, 5]], N_ELEMENTS)
        self.assertEqual(block.shape, (3, N_ELEMENTS))
        self.assertEqual(block.sum(dim=1).tolist(), [1.0, 2.0, 3.0])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            chemical_system_vectors([], N_ELEMENTS)


class TestAttach(unittest.TestCase):
    def _data(self):
        return {"composition_tokens": [torch.tensor([0, 1]), torch.tensor([2])],
                "composition_counts": [torch.tensor([1.0, 3.0]), torch.tensor([2.0])]}

    def test_adds_the_dense_field_from_the_counter_keys(self):
        data = attach_chemical_system_vector(self._data(), N_ELEMENTS)
        self.assertEqual(data[CHEMICAL_SYSTEM_FIELD].shape, (2, N_ELEMENTS))
        self.assertEqual(data[CHEMICAL_SYSTEM_FIELD].sum(dim=1).tolist(), [2.0, 1.0])

    def test_ignores_the_counts(self):
        """The same cache serves both modes, so the counts are present and unused."""
        data = self._data()
        heavier = self._data()
        heavier["composition_counts"] = [torch.tensor([9.0, 9.0]), torch.tensor([9.0])]
        self.assertTrue(torch.equal(
            attach_chemical_system_vector(data, N_ELEMENTS)[CHEMICAL_SYSTEM_FIELD],
            attach_chemical_system_vector(heavier, N_ELEMENTS)[CHEMICAL_SYSTEM_FIELD]))

    def test_is_idempotent(self):
        data = attach_chemical_system_vector(self._data(), N_ELEMENTS)
        marker = data[CHEMICAL_SYSTEM_FIELD]
        self.assertIs(attach_chemical_system_vector(data, N_ELEMENTS)[CHEMICAL_SYSTEM_FIELD],
                      marker)

    def test_missing_counters_say_how_to_fix_it(self):
        with self.assertRaises(KeyError) as caught:
            attach_chemical_system_vector({}, N_ELEMENTS)
        self.assertIn("counters", str(caught.exception))


class TestParseAndDescribe(unittest.TestCase):
    def _tokeniser(self):
        tokeniser = {"Ba": 0, "Ti": 1, "O": 2, "Na": 3}
        # Mimics EnumeratingTokeniser: dict plus a token-id-indexed inverse.
        tokeniser = unittest.mock.MagicMock(wraps=tokeniser)
        tokeniser.__contains__ = lambda _, key: key in ("Ba", "Ti", "O", "Na")
        tokeniser.__getitem__ = lambda _, key: {"Ba": 0, "Ti": 1, "O": 2, "Na": 3}[key]
        tokeniser.__len__ = lambda _: 4
        tokeniser.to_token = ["Ba", "Ti", "O", "Na"]
        return tokeniser

    def test_parses_a_dashed_system(self):
        symbols, tokens = parse_chemical_system("Ba-Ti-O", self._tokeniser())
        self.assertEqual(symbols, ("Ba", "Ti", "O"))
        self.assertEqual(tokens, (0, 1, 2))

    def test_tokens_come_out_sorted_and_deduplicated(self):
        symbols, tokens = parse_chemical_system("O-Ba-O", self._tokeniser())
        self.assertEqual(symbols, ("O", "Ba"))
        self.assertEqual(tokens, (0, 2))

    def test_rejects_an_empty_system(self):
        with self.assertRaises(ValueError):
            parse_chemical_system("-", self._tokeniser())

    def test_rejects_an_element_outside_the_vocabulary(self):
        with self.assertRaises(KeyError):
            parse_chemical_system("Ba-Xe", self._tokeniser())

    def test_describe_reads_the_vector_back(self):
        tokeniser = self._tokeniser()
        vector = chemical_system_vector([0, 2], 4)
        self.assertEqual(describe(vector, tokeniser), "Ba-O")


class TestTrainerLayout(unittest.TestCase):
    """The block lands in the conditioning vector after the scalars, once."""

    def _trainer(self, condition_feature, chemical_system_conditioning):
        from wyckoff_transformer.trainer import WyckoffTrainer

        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.condition_feature = condition_feature
        trainer.composition_conditioning = False
        trainer.chemical_system_conditioning = chemical_system_conditioning
        trainer.condition_on_cell_size = True
        trainer.n_elements = N_ELEMENTS
        return trainer

    def _dataset(self, num_examples=4):
        dataset = unittest.mock.MagicMock()
        dataset.data = {
            "energy_above_hull": torch.arange(num_examples, dtype=torch.float32).unsqueeze(1),
            CHEMICAL_SYSTEM_FIELD: torch.eye(num_examples, N_ELEMENTS),
        }
        return dataset

    def test_width_is_the_scalars_plus_the_vocabulary(self):
        trainer = self._trainer("energy_above_hull", True)
        self.assertEqual(trainer.condition_dim, 1 + N_ELEMENTS)
        self.assertEqual(trainer.formula_conditioning_field, CHEMICAL_SYSTEM_FIELD)

    def test_scalars_come_first_then_the_system(self):
        trainer = self._trainer("energy_above_hull", True)
        built = trainer.build_cond(self._dataset())
        self.assertEqual(built.shape, (4, 1 + N_ELEMENTS))
        self.assertTrue(torch.equal(built[:, 0], torch.arange(4, dtype=torch.float32)))
        self.assertTrue(torch.equal(built[:, 1:], torch.eye(4, N_ELEMENTS)))

    def test_condition_dim_matches_what_build_cond_produces(self):
        trainer = self._trainer("energy_above_hull", True)
        self.assertEqual(trainer.build_cond(self._dataset()).shape[-1], trainer.condition_dim)

    def test_system_only(self):
        trainer = self._trainer(None, True)
        self.assertEqual(trainer.condition_dim, N_ELEMENTS)
        self.assertEqual(trainer.build_cond(self._dataset()).shape, (4, N_ELEMENTS))

    def test_the_two_formula_modes_are_refused_together(self):
        from wyckoff_transformer.trainer import formula_conditioning_width

        with self.assertRaises(ValueError):
            formula_conditioning_width(N_ELEMENTS, True, True)

    def test_batch_selection_is_honoured(self):
        trainer = self._trainer("energy_above_hull", True)
        selection = torch.tensor([2, 0])
        built = trainer.build_cond(self._dataset(), selection)
        self.assertEqual(built.shape, (2, 1 + N_ELEMENTS))
        self.assertTrue(torch.equal(built[:, 0], torch.tensor([2.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
