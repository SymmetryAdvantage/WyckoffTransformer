"""Tests for the (chemical system, space group) sampler.

Three properties carry this module, and the rest of the file is edge cases around
them. Every draw has to satisfy the caller's constraint -- a sampler that can emit
a system outside the palette is worse than the rejection loop it replaces, because
nothing downstream checks. The smoothing has to reduce to the empirical
distribution when there is data and to the back-off when there is none, since one
expression serves both and a mistake there is silent. And a plan has to survive the
round trip to JSON and back, because that is how it reaches a generation run.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wyckoff_transformer.system_prior import (
    DEFAULT_SG_KAPPA,
    SystemSpaceGroupPrior,
    SystemDraws,
    _decode_space_groups,
    _split_systems,
)

# Li Na K O S Fe Mn, then the service tokens the element tokeniser carries.
SYMBOLS = ("Li", "Na", "K", "O", "S", "Fe", "Mn", "MASK", "STOP", "PAD")
LI, NA, K, O, S, FE, MN = range(7)

#: A corpus small enough to check by hand: Li-O is the common system, Li-O-Mn the
#: next, and Na-S is the one a Li query must never reach.
ROWS = (
    ((LI, O), 12), ((LI, O), 12), ((LI, O), 12), ((LI, O), 225),
    ((LI, O, MN), 2), ((LI, O, MN), 12),
    ((LI, O, FE), 62),
    ((LI, S), 14),
    ((NA, S), 225), ((NA, S), 225),
    ((FE, O), 62), ((FE, O), 12),
)


def build_prior(rows=ROWS, held_out=None, **kwargs):
    return SystemSpaceGroupPrior.from_rows(
        systems=[row[0] for row in rows],
        space_groups=[row[1] for row in rows],
        element_symbols=SYMBOLS,
        held_out_systems=held_out,
        **kwargs)


class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.prior = build_prior()

    def _symbols(self, feasible):
        return {self.prior.system_symbols(index) for index in feasible.indices}

    def test_required_is_a_floor_and_allowed_a_ceiling(self):
        feasible = self.prior.feasible_systems(required="Li", allowed="Li-O-Mn")
        self.assertEqual(self._symbols(feasible), {("Li", "O"), ("Li", "O", "Mn")})

    def test_an_element_outside_the_palette_excludes_the_system(self):
        # Li-O-Fe is observed and contains Li, but Fe is not on the palette.
        feasible = self.prior.feasible_systems(required="Li", allowed="Li-O-Mn")
        self.assertNotIn(("Li", "O", "Fe"), self._symbols(feasible))

    def test_no_required_elements_keeps_every_subset_of_the_palette(self):
        feasible = self.prior.feasible_systems(allowed="Na-S")
        self.assertEqual(self._symbols(feasible), {("Na", "S")})

    def test_the_default_palette_is_the_whole_vocabulary(self):
        self.assertEqual(len(self.prior.feasible_systems()), 6)

    def test_weights_follow_the_row_counts(self):
        feasible = self.prior.feasible_systems(required="Li", allowed="Li-O-Mn")
        by_system = dict(zip(
            (self.prior.system_symbols(index) for index in feasible.indices), feasible.weights))
        # Li-O has four rows against Li-O-Mn's two.
        self.assertAlmostEqual(by_system[("Li", "O")], 4 / 6)
        self.assertAlmostEqual(by_system[("Li", "O", "Mn")], 2 / 6)

    def test_temperature_flattens_the_weights(self):
        feasible = self.prior.feasible_systems(
            required="Li", allowed="Li-O-Mn", system_temperature=1e6)
        self.assertTrue(np.allclose(feasible.weights, 0.5, atol=1e-3))

    def test_arity_bounds_narrow_the_set(self):
        self.assertEqual(
            self._symbols(self.prior.feasible_systems(
                required="Li", allowed="Li-O-Mn-Fe", max_arity=2)),
            {("Li", "O")})
        self.assertEqual(
            self._symbols(self.prior.feasible_systems(
                required="Li", allowed="Li-O-Mn-Fe", min_arity=3)),
            {("Li", "O", "Mn"), ("Li", "O", "Fe")})

    def test_acceptance_rate_is_the_share_of_the_corpus(self):
        feasible = self.prior.feasible_systems(required="Li", allowed="Li-O-Mn")
        self.assertAlmostEqual(feasible.acceptance_rate, 6 / len(ROWS))

    def test_a_required_element_outside_the_palette_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.prior.feasible_systems(required="Fe", allowed="Li-O")
        self.assertIn("allowed", str(caught.exception))

    def test_an_unknown_element_is_refused(self):
        with self.assertRaises(KeyError):
            self.prior.feasible_systems(required="Xe", allowed="Li-O-Xe")

    def test_service_tokens_are_not_elements(self):
        with self.assertRaises(ValueError):
            self.prior.feasible_systems(required="STOP")
        # ... and are silently dropped from a palette, where they are harmless.
        self.assertEqual(len(self.prior.feasible_systems(allowed="Li-O-STOP")), 1)

    def test_tokens_are_accepted_as_well_as_symbols(self):
        by_symbol = self.prior.feasible_systems(required="Li", allowed="Li-O-Mn")
        by_token = self.prior.feasible_systems(required=[LI], allowed=[LI, O, MN])
        self.assertEqual(list(by_symbol.indices), list(by_token.indices))


class TestWideVocabulary(unittest.TestCase):
    """The bitmask is 64 bits per lane, so a vocabulary wider than that is the case
    where an off-by-one in the packing shows up as a wrong answer rather than a crash."""

    def setUp(self):
        self.symbols = tuple(f"E{index}" for index in range(130))
        rows = [((1, 63, 64), 12), ((1, 64, 129), 2), ((63, 65), 225), ((1, 63, 64, 129), 14)]
        self.prior = SystemSpaceGroupPrior.from_rows(
            systems=[row[0] for row in rows],
            space_groups=[row[1] for row in rows],
            element_symbols=self.symbols)

    def test_subset_tests_cross_the_lane_boundary(self):
        feasible = self.prior.feasible_systems(required=[63], allowed=[1, 63, 64])
        self.assertEqual({self.prior.system_symbols(index) for index in feasible.indices},
                         {("E1", "E63", "E64")})

    def test_the_high_lane_is_not_ignored(self):
        # E129 lives in the second lane; a palette without it must exclude both systems
        # that use it.
        feasible = self.prior.feasible_systems(allowed=[1, 63, 64, 65])
        self.assertEqual(len(feasible), 2)


class TestSpaceGroupDistribution(unittest.TestCase):
    def setUp(self):
        self.prior = build_prior()

    def test_it_is_a_distribution(self):
        probabilities = self.prior.space_group_probabilities((LI, O))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue((probabilities > 0).all())

    def test_no_shrinkage_is_the_empirical_table(self):
        probabilities = self.prior.space_group_probabilities((LI, O), kappa=0.0)
        index = {int(sg): position for position, sg in enumerate(self.prior.space_groups)}
        self.assertAlmostEqual(probabilities[index[12]], 0.75)
        self.assertAlmostEqual(probabilities[index[225]], 0.25)
        self.assertAlmostEqual(probabilities[index[62]], 0.0)

    def test_heavy_shrinkage_reaches_the_back_off(self):
        self.assertTrue(np.allclose(
            self.prior.space_group_probabilities((LI, O), kappa=1e9),
            self.prior.back_off_probabilities((LI, O)),
            atol=1e-6))

    def test_an_unobserved_system_is_exactly_the_back_off(self):
        # Li-Fe is not in the corpus; the same expression has to serve it, with no
        # separate code path and no zero division.
        self.assertIsNone(self.prior.index_of((LI, FE)))
        self.assertTrue(np.allclose(
            self.prior.space_group_probabilities((LI, FE)),
            self.prior.back_off_probabilities((LI, FE))))

    def test_shrinkage_keeps_the_observed_mode(self):
        probabilities = self.prior.space_group_probabilities((LI, O), kappa=DEFAULT_SG_KAPPA)
        index = {int(sg): position for position, sg in enumerate(self.prior.space_groups)}
        self.assertEqual(int(np.argmax(probabilities)), index[12])

    def test_the_back_off_prefers_what_the_elements_prefer(self):
        # Fe appears only in 62 and 12, so an unseen Fe system should favour them over
        # 14, which only Li-S ever used.
        probabilities = self.prior.back_off_probabilities((FE, MN))
        index = {int(sg): position for position, sg in enumerate(self.prior.space_groups)}
        self.assertGreater(probabilities[index[62]], probabilities[index[14]])

    def test_kappa_zero_on_an_unobserved_system_is_refused(self):
        with self.assertRaises(ValueError):
            self.prior.space_group_probabilities((LI, FE), kappa=0.0)

    def test_an_empty_system_has_no_distribution(self):
        with self.assertRaises(ValueError):
            self.prior.back_off_probabilities(())

    def test_temperature_flattens(self):
        sharp = self.prior.space_group_probabilities((LI, O))
        flat = self.prior.space_group_probabilities((LI, O), temperature=1e6)
        self.assertLess(float(flat.max()) - float(flat.min()),
                        float(sharp.max()) - float(sharp.min()))
        self.assertAlmostEqual(float(flat.sum()), 1.0)

    def test_temperature_cannot_resurrect_an_impossible_space_group(self):
        # Flattening reweights what the distribution admits; with no shrinkage a space
        # group the system has never been seen in stays at zero, which is the
        # difference between a temperature and a smoothing.
        flat = self.prior.space_group_probabilities((LI, O), kappa=0.0, temperature=1e6)
        index = {int(sg): position for position, sg in enumerate(self.prior.space_groups)}
        self.assertEqual(float(flat[index[62]]), 0.0)


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.prior = build_prior()

    def _draws(self, **kwargs):
        kwargs.setdefault("rng", 0)
        kwargs.setdefault("novel_fraction", 0.0)
        return self.prior.sample(64, required="Li", allowed="Li-O-Mn-Fe", **kwargs)

    def test_every_row_satisfies_the_constraint(self):
        draws = self._draws()
        allowed = {LI, O, MN, FE}
        for row in range(len(draws)):
            tokens = set(draws.element_tokens[row])
            self.assertIn(LI, tokens)
            self.assertTrue(tokens <= allowed, tokens)

    def test_it_draws_the_number_asked_for(self):
        self.assertEqual(len(self._draws()), 64)

    def test_space_groups_come_from_the_vocabulary(self):
        draws = self._draws()
        self.assertTrue(set(draws.space_groups.tolist()) <= set(
            int(sg) for sg in self.prior.space_groups))

    def test_no_novel_rows_when_none_are_asked_for(self):
        draws = self._draws()
        self.assertFalse(draws.is_novel.any())
        for tokens in draws.element_tokens:
            self.assertIsNotNone(self.prior.index_of(tokens))

    def test_novel_rows_are_systems_the_data_does_not_have(self):
        draws = self._draws(novel_fraction=1.0)
        self.assertTrue(draws.is_novel.all())
        for tokens in draws.element_tokens:
            self.assertIsNone(self.prior.index_of(tokens))
            self.assertIn(LI, tokens)
            self.assertTrue(set(tokens) <= {LI, O, MN, FE})

    def test_novel_rows_respect_arity_bounds(self):
        draws = self._draws(novel_fraction=1.0, min_arity=3, max_arity=3)
        self.assertTrue(all(len(tokens) == 3 for tokens in draws.element_tokens))

    def test_the_mixture_produces_both_kinds(self):
        draws = self.prior.sample(
            400, required="Li", allowed="Li-O-Mn-Fe", novel_fraction=0.5, rng=1)
        self.assertGreater(int(draws.is_novel.sum()), 100)
        self.assertLess(int(draws.is_novel.sum()), 300)

    def test_the_seed_makes_it_reproducible(self):
        first = self._draws(rng=7)
        second = self._draws(rng=7)
        self.assertEqual(first.element_tokens, second.element_tokens)
        self.assertTrue(np.array_equal(first.space_groups, second.space_groups))

    def test_different_seeds_differ(self):
        self.assertNotEqual(self._draws(rng=1).element_tokens, self._draws(rng=2).element_tokens)

    def test_the_novelty_rate_measured_at_build_time_is_the_default(self):
        prior = build_prior(held_out=[(LI, O), (LI, FE), (LI, MN), (NA, S)])
        self.assertAlmostEqual(prior.metadata["held_out_novelty_rate"], 0.5)
        draws = prior.sample(400, required="Li", allowed="Li-O-Mn-Fe", rng=3)
        self.assertAlmostEqual(draws.query["novel_fraction"], 0.5)
        self.assertGreater(int(draws.is_novel.sum()), 100)

    def test_no_feasible_system_and_no_novelty_is_an_error_that_says_what_to_do(self):
        with self.assertRaises(ValueError) as caught:
            self.prior.sample(10, required="K", allowed="K-O", novel_fraction=0.0)
        self.assertIn("novel_fraction", str(caught.exception))

    def test_an_empty_observed_set_falls_back_to_the_novel_proposal(self):
        draws = self.prior.sample(16, required="K", allowed="K-O-S", novel_fraction=0.5, rng=0)
        self.assertTrue(draws.is_novel.all())
        for tokens in draws.element_tokens:
            self.assertIn(K, tokens)

    def test_a_palette_with_nothing_new_left_tops_up_from_the_counts(self):
        # Every subset of Na-S that contains Na-S is observed, so the novel proposal
        # cannot deliver and the batch is completed from the empirical component
        # rather than looping forever.
        draws = self.prior.sample(8, required="Na-S", allowed="Na-S", novel_fraction=1.0, rng=0)
        self.assertEqual(len(draws), 8)
        self.assertFalse(draws.is_novel.any())

    def test_zero_structures_is_refused(self):
        with self.assertRaises(ValueError):
            self.prior.sample(0)

    def test_an_out_of_range_novel_fraction_is_refused(self):
        with self.assertRaises(ValueError):
            self.prior.sample(4, novel_fraction=1.5)

    def test_the_query_is_recorded_with_the_draws(self):
        query = self._draws().query
        self.assertEqual(query["required"], "Li")
        self.assertEqual(query["allowed"], "Li-O-Fe-Mn")
        self.assertEqual(query["n_structures"], 64)

    def test_the_empirical_component_follows_the_weights(self):
        draws = self.prior.sample(
            4000, required="Li", allowed="Li-O-Mn", novel_fraction=0.0, rng=11)
        share = sum(1 for tokens in draws.element_tokens if tokens == (LI, O)) / len(draws)
        self.assertAlmostEqual(share, 4 / 6, delta=0.05)


class TestDraws(unittest.TestCase):
    def setUp(self):
        self.prior = build_prior()
        self.draws = self.prior.sample(
            32, required="Li", allowed="Li-O-Mn-Fe", novel_fraction=0.25, rng=5)

    def test_grouping_conserves_the_batch(self):
        cells = self.draws.grouped()
        self.assertEqual(sum(cell.count for cell in cells), len(self.draws))
        self.assertEqual(len({(cell.element_tokens, cell.space_group) for cell in cells}),
                         len(cells))

    def test_the_manifest_carries_its_vocabulary(self):
        manifest = json.loads(json.dumps(self.draws.manifest()))
        self.assertEqual(manifest["element_symbols"], list(SYMBOLS))
        restored = SystemDraws.from_manifest(manifest)
        self.assertEqual(restored.element_symbols, SYMBOLS)

    def test_a_plan_read_against_the_wrong_vocabulary_is_refused(self):
        manifest = self.draws.manifest()
        with self.assertRaises(ValueError) as caught:
            SystemDraws.from_manifest(manifest, SYMBOLS[:-1])
        self.assertIn("vocabulary", str(caught.exception))

    def test_a_plan_without_a_vocabulary_needs_one(self):
        manifest = self.draws.manifest()
        del manifest["element_symbols"]
        with self.assertRaises(ValueError):
            SystemDraws.from_manifest(manifest)
        self.assertEqual(len(SystemDraws.from_manifest(manifest, SYMBOLS)), len(self.draws))

    def test_the_manifest_round_trips(self):
        manifest = json.loads(json.dumps(self.draws.manifest()))
        restored = SystemDraws.from_manifest(manifest, SYMBOLS)
        self.assertEqual(len(restored), len(self.draws))
        self.assertEqual(sorted(restored.systems()), sorted(self.draws.systems()))
        self.assertEqual(sorted(restored.space_groups.tolist()),
                         sorted(self.draws.space_groups.tolist()))
        self.assertEqual(int(restored.is_novel.sum()), int(self.draws.is_novel.sum()))

    def test_the_element_mask_is_the_row_system_and_stop(self):
        mask = self.draws.element_mask(len(SYMBOLS))
        self.assertEqual(tuple(mask.shape), (len(self.draws), len(SYMBOLS)))
        stop = SYMBOLS.index("STOP")
        for row in range(len(self.draws)):
            expected = set(self.draws.element_tokens[row]) | {stop}
            self.assertEqual(set(np.nonzero(mask[row].numpy())[0].tolist()), expected)

    def test_the_mask_is_the_system_not_the_palette(self):
        # The palette has four elements; no row may admit all four unless it drew
        # all four, which is the whole point of sampling a system per structure.
        mask = self.draws.element_mask(len(SYMBOLS))
        for row in range(len(self.draws)):
            self.assertEqual(int(mask[row].sum()), len(self.draws.element_tokens[row]) + 1)

    def test_the_conditioning_block_is_the_chemical_system_representation(self):
        try:
            from wyckoff_transformer.chemical_system import chemical_system_vector
        except ImportError:  # pragma: no cover - the mode is not installed
            self.skipTest("chemical_system conditioning is not available")
        block = self.draws.conditioning_block(len(SYMBOLS))
        self.assertEqual(tuple(block.shape), (len(self.draws), len(SYMBOLS)))
        for row in range(len(self.draws)):
            self.assertTrue(np.allclose(
                block[row].numpy(),
                chemical_system_vector(self.draws.element_tokens[row], len(SYMBOLS)).numpy()))

    def test_start_tensor_one_hot(self):
        tokeniser = _FakeSpaceGroupEncoder({12: (1.0, 0.0), 14: (0.0, 1.0),
                                            2: (1.0, 1.0), 225: (0.0, 0.0), 62: (1.0, 0.5)})
        start = self.draws.start_tensor(tokeniser, "one_hot")
        self.assertEqual(tuple(start.shape), (len(self.draws), 2))

    def test_start_tensor_categorial(self):
        tokeniser = {12: 0, 14: 1, 2: 2, 225: 3, 62: 4}
        start = self.draws.start_tensor(tokeniser, "categorial")
        self.assertEqual(tuple(start.shape), (len(self.draws),))
        self.assertEqual(
            start.tolist(), [tokeniser[int(sg)] for sg in self.draws.space_groups])

    def test_a_space_group_the_model_does_not_know_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.draws.start_tensor({12: 0}, "categorial")
        self.assertIn("vocabulary", str(caught.exception))

    def test_an_unknown_start_type_is_refused(self):
        with self.assertRaises(ValueError):
            self.draws.start_tensor({}, "embedding")

    def test_the_summary_names_the_top_systems(self):
        self.assertIn("Li-O", self.draws.summary())


class _FakeSpaceGroupEncoder(dict):
    """Enough of `SpaceGroupEncoder` for `start_tensor`: a mapping plus an encoder."""

    def encode_spacegroups(self, space_groups, **tensor_args):
        import torch

        return torch.tensor([self[int(sg)] for sg in space_groups], **tensor_args)


class TestArtifact(unittest.TestCase):
    def test_save_and_load_preserve_the_tables_and_the_sampling(self):
        prior = build_prior(held_out=[(LI, O), (LI, FE)])
        with tempfile.TemporaryDirectory() as directory:
            path = prior.save(Path(directory) / "prior.npz")
            restored = SystemSpaceGroupPrior.load(path)
        self.assertEqual(restored.element_symbols, prior.element_symbols)
        self.assertEqual(restored.metadata, prior.metadata)
        self.assertTrue(np.array_equal(restored.system_counts, prior.system_counts))
        self.assertTrue(np.allclose(
            restored.space_group_probabilities((LI, O)),
            prior.space_group_probabilities((LI, O))))
        self.assertEqual(
            restored.sample(16, required="Li", rng=2).element_tokens,
            prior.sample(16, required="Li", rng=2).element_tokens)


class TestFromRows(unittest.TestCase):
    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            SystemSpaceGroupPrior.from_rows([(LI, O)], [12, 14], SYMBOLS)

    def test_an_empty_corpus_is_refused(self):
        with self.assertRaises(ValueError):
            SystemSpaceGroupPrior.from_rows([], [], SYMBOLS)

    def test_an_empty_system_is_refused(self):
        with self.assertRaises(ValueError):
            SystemSpaceGroupPrior.from_rows([()], [12], SYMBOLS)

    def test_a_token_outside_the_vocabulary_is_refused(self):
        with self.assertRaises(ValueError):
            SystemSpaceGroupPrior.from_rows([(len(SYMBOLS),)], [12], SYMBOLS)

    def test_repeated_elements_are_one_system(self):
        prior = SystemSpaceGroupPrior.from_rows(
            [(LI, O, O), (O, LI)], [12, 12], SYMBOLS)
        self.assertEqual(prior.metadata["n_systems"], 1)

    def test_weights_replace_row_counts(self):
        prior = SystemSpaceGroupPrior.from_rows(
            [(LI, O), (NA, S)], [12, 225], SYMBOLS, weights=[3.0, 1.0])
        feasible = prior.feasible_systems()
        self.assertAlmostEqual(float(feasible.weights.max()), 0.75)


class TestCacheHelpers(unittest.TestCase):
    def test_ragged_composition_tokens_become_systems(self):
        systems = _split_systems(np.array([0, 3, 3, 5, 1]), np.array([2, 1, 2]))
        self.assertEqual(systems, [(0, 3), (3,), (1, 5)])

    def test_categorial_start_tokens_decode_by_index(self):
        class _Tokeniser:
            to_token = [12, 14, 225]
        self.assertEqual(
            _decode_space_groups(np.array([2, 0, 1]), _Tokeniser()).tolist(), [225, 12, 14])

    def test_encoded_start_tokens_decode_bytewise(self):
        class _Encoder:
            np_dict = {12: np.array([1, 0, 1]), 14: np.array([0, 1, 1])}
        encoded = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.int64)
        self.assertEqual(_decode_space_groups(encoded, _Encoder()).tolist(), [14, 12])

    def test_an_unknown_encoding_is_refused(self):
        class _Encoder:
            np_dict = {12: np.array([1, 0, 1])}
        with self.assertRaises(ValueError):
            _decode_space_groups(np.array([[0, 0, 0]], dtype=np.int64), _Encoder())


class TestCacheReading(unittest.TestCase):
    """The reader walks the safetensors structure metadata by hand, to pull two
    columns out of a 6.5 GB file rather than materialising all of it. That is worth a
    test against a real cache file, since the layout is not this module's to define."""

    def setUp(self):
        import torch

        from wyckoff_transformer.tokenization import save_tensor_cache

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.safetensors"
        save_tensor_cache({
            "train": {
                "composition_tokens": [torch.tensor([0, 3]), torch.tensor([1, 3, 6])],
                "spacegroup_number": torch.tensor([[1, 0], [0, 1]], dtype=torch.int16),
                "energy_above_hull": torch.tensor([[0.0], [1.0]]),
            },
        }, self.path)

    def _open(self):
        import json as json_module

        from safetensors import safe_open

        handle = safe_open(str(self.path), framework="pt", device="cpu")
        structure = json_module.loads(
            handle.metadata()["wyckoff_transformer_tensor_cache_structure"])
        return handle, structure

    def test_a_dense_field_comes_back_as_it_was_written(self):
        from wyckoff_transformer.system_prior import _read_cache_node

        handle, structure = self._open()
        with handle:
            values = _read_cache_node(handle, structure, "train", "energy_above_hull")
        self.assertEqual(values.reshape(-1).tolist(), [0.0, 1.0])

    def test_a_ragged_field_comes_back_as_data_and_lengths(self):
        from wyckoff_transformer.system_prior import _read_cache_node

        handle, structure = self._open()
        with handle:
            data, lengths = _read_cache_node(handle, structure, "train", "composition_tokens")
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(_split_systems(data, lengths), [(0, 3), (1, 3, 6)])

    def test_a_missing_field_says_what_is_there(self):
        from wyckoff_transformer.system_prior import _read_cache_node

        handle, structure = self._open()
        with handle, self.assertRaises(KeyError) as caught:
            _read_cache_node(handle, structure, "train", "max_force")
        self.assertIn("composition_tokens", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
