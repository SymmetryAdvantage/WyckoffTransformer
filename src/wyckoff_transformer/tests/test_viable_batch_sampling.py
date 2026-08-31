"""Tests for length-indexed sampling of batches viable at a given known_seq_len."""
import random
import unittest
from collections import Counter
from unittest.mock import MagicMock

import torch

from wyckoff_transformer.cascade.dataset import (
    AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass,
    SAMPLE_WITHOUT_REPLACEMENT_RATIO)
from wyckoff_transformer.trainer import WyckoffTrainer


def _make_dataset(lengths, max_sequence_length=8, batch_size=None):
    """A dataset whose pure sequence lengths are exactly `lengths`."""
    n = len(lengths)
    data = {
        "field1": torch.arange(n * max_sequence_length, dtype=torch.int64).reshape(
            n, max_sequence_length) % 3,
        "spacegroup": torch.zeros(n, dtype=torch.int64),
        "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64),
    }
    return AugmentedCascadeDataset(
        data=data,
        cascade_order=("field1",),
        masks={"field1": 99},
        pads={"field1": 100},
        stops={"field1": 101},
        # Must span the service token ids below: the multiclass target is a bincount over classes.
        num_classes={"field1": 102},
        start_field="spacegroup",
        augmented_fields=None,
        batch_size=batch_size,
    )


class TestViableCount(unittest.TestCase):
    def test_matches_brute_force(self):
        lengths = [0, 1, 1, 2, 3, 3, 3, 7]
        dataset = _make_dataset(lengths)
        for k in range(dataset.max_sequence_length + 1):
            expected = sum(1 for length in lengths if length >= k)
            self.assertEqual(dataset.viable_count(k), expected, f"at known_seq_len={k}")

    def test_beyond_the_longest_sequence_is_zero(self):
        dataset = _make_dataset([0, 1, 2])
        self.assertEqual(dataset.viable_count(3), 0)


class TestSampleViableBatch(unittest.TestCase):
    def test_only_returns_viable_examples(self):
        lengths = [0, 1, 2, 3, 4, 5, 6, 7] * 4
        dataset = _make_dataset(lengths)
        for k in range(8):
            indices = dataset.sample_viable_batch(k, batch_size=16)
            self.assertGreater(len(indices), 0)
            self.assertTrue(
                bool((dataset.pure_sequences_lengths[indices] >= k).all()),
                f"non-viable example sampled at known_seq_len={k}")

    def test_batch_is_as_full_as_the_data_allows(self):
        lengths = [7] * 50 + [0] * 50
        dataset = _make_dataset(lengths)
        # Plenty of viable examples: the batch is full.
        self.assertEqual(len(dataset.sample_viable_batch(0, batch_size=32)), 32)
        self.assertEqual(len(dataset.sample_viable_batch(7, batch_size=32)), 32)
        # Fewer viable examples than the batch: take all of them, and no more.
        self.assertEqual(len(dataset.sample_viable_batch(7, batch_size=80)), 50)

    def test_the_whole_viable_set_is_returned_exactly(self):
        """Not a bootstrap resample of it, which would miss ~37% of the examples."""
        lengths = [7] * 30 + [0] * 70
        dataset = _make_dataset(lengths)
        indices = dataset.sample_viable_batch(7, batch_size=100)
        self.assertEqual(len(indices), 30)
        self.assertEqual(len(set(indices.tolist())), 30)
        self.assertEqual(
            set(indices.tolist()),
            {i for i, length in enumerate(lengths) if length >= 7})

    def test_a_partial_batch_is_drawn_without_replacement(self):
        lengths = [7] * 100
        dataset = _make_dataset(lengths)
        # 40 * 16 >= 100, so this takes the permuted path.
        self.assertGreaterEqual(40 * SAMPLE_WITHOUT_REPLACEMENT_RATIO, 100)
        for _ in range(20):
            indices = dataset.sample_viable_batch(7, batch_size=40).tolist()
            self.assertEqual(len(indices), 40)
            self.assertEqual(len(set(indices)), 40, "duplicate example in a partial batch")

    def test_no_viable_example_raises_instead_of_looping(self):
        dataset = _make_dataset([0, 1, 2])
        with self.assertRaises(ValueError):
            dataset.sample_viable_batch(5, batch_size=2)

    def test_loader_delegates_to_the_length_index(self):
        lengths = [0, 1, 2, 3, 4, 5, 6, 7] * 4
        dataset = _make_dataset(lengths, batch_size=8)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        indices = loader.get_next_viable_batch(5)
        self.assertTrue(bool((dataset.pure_sequences_lengths[indices] >= 5).all()))


class TestSampleKnownSeqLen(unittest.TestCase):
    """Training draws known_seq_len from the distribution evaluation weights it by."""

    def test_never_draws_a_length_with_no_viable_example(self):
        # Padded width 8, nothing longer than 3.
        dataset = _make_dataset([0, 1, 2, 3] * 5, max_sequence_length=8)
        for _ in range(500):
            k = dataset.sample_known_seq_len()
            self.assertGreater(dataset.viable_count(k), 0)

    def test_stays_within_the_padded_width(self):
        dataset = _make_dataset([7] * 20, max_sequence_length=8)
        for _ in range(500):
            self.assertIn(dataset.sample_known_seq_len(), range(8))

    def test_frequencies_track_the_viable_counts(self):
        lengths = [0] * 100 + [1] * 50 + [2] * 25 + [5] * 5
        dataset = _make_dataset(lengths, max_sequence_length=8)
        random.seed(20260828)
        draws = 200000
        counts = Counter(dataset.sample_known_seq_len() for _ in range(draws))
        total = sum(dataset.viable_count(k) for k in range(dataset.max_sequence_length))
        for k in range(dataset.max_sequence_length):
            expected = dataset.viable_count(k) / total
            self.assertAlmostEqual(
                counts[k] / draws, expected, delta=0.01,
                msg=f"known_seq_len={k} drawn at the wrong rate")


class TestTrainStepScale(unittest.TestCase):
    """With known_seq_len sampled by viability, every step must be on one loss scale."""

    def _make_trainer(self, dataset, loader):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.NextToken
        trainer.multiclass_next_token_with_order_permutation = True
        trainer.condition_feature = None
        trainer.cascade_target_count = 1
        trainer.clip_grad_norm = 1e9
        trainer.train_dataset = dataset
        trainer.train_loader = loader
        # A per-example loss of exactly 1, so the assertion is about the scale, not the criterion.
        trainer.criterion = lambda prediction, target: prediction.new_tensor(
            float(prediction.size(0)))
        trainer.model = lambda start, masked, padding, known_cascade_len, cond=None: torch.zeros(
            start.size(0), 3)
        return trainer

    def test_every_known_seq_len_carries_the_same_weight(self):
        lengths = [0, 1, 2, 3, 4, 5, 6, 7] * 25
        dataset = _make_dataset(lengths, batch_size=16)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        trainer = self._make_trainer(dataset, loader)
        for k in range(8):
            loss, n_samples = trainer.get_loss(
                dataset, k, 0, loader=loader, rescale_to_viable=False, return_n_samples=True)
            loss = loss / n_samples
            # The stub criterion sums 1 per example, so a per-example mean is exactly 1 at every
            # known_seq_len -- including those where the viable set is smaller than a batch.
            self.assertAlmostEqual(
                float(loss), 1.0, places=4,
                msg=f"training step at known_seq_len={k} is off the per-example scale")

    def test_evaluation_still_weights_by_viability(self):
        """The metric must keep summing over known_seq_len weighted by how much data reaches it."""
        lengths = [0, 1, 2, 3, 4, 5, 6, 7] * 25
        dataset = _make_dataset(lengths, batch_size=16)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        trainer = self._make_trainer(dataset, loader)
        for k in range(8):
            self.assertAlmostEqual(
                float(trainer.get_loss(dataset, k, 0, loader=loader)),
                float(dataset.viable_count(k)), places=4)


class TestViabilityRescale(unittest.TestCase):
    """A sampled batch must carry the same weight as a pass over every viable example."""

    def _make_trainer(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.NextToken
        trainer.multiclass_next_token_with_order_permutation = True
        trainer.condition_feature = None
        # Loss counting examples, so the assertions are about the rescale, not the criterion.
        trainer.criterion = lambda prediction, target: prediction.new_tensor(
            float(prediction.size(0)))
        trainer.model = lambda start, masked, padding, known_cascade_len, cond=None: torch.zeros(
            start.size(0), 3)
        return trainer

    def test_batched_loss_matches_a_full_pass(self):
        lengths = [0, 1, 2, 3, 4, 5, 6, 7] * 25
        dataset = _make_dataset(lengths, batch_size=16)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        trainer = self._make_trainer()
        for k in range(8):
            full = trainer.get_loss(dataset, k, 0, no_batch=True)
            batched = trainer.get_loss(dataset, k, 0, loader=loader, no_batch=False)
            # The stub criterion counts examples, so a correctly rescaled batch reports the
            # number of viable examples in the split no matter how many were sampled.
            self.assertAlmostEqual(
                float(full), float(batched), places=4,
                msg=f"batched loss is off scale at known_seq_len={k}")
            self.assertAlmostEqual(float(full), float(dataset.viable_count(k)), places=4)

    def test_batchless_mode_is_unscaled(self):
        """With no loader the batch already is every viable example, so the factor is 1."""
        dataset = _make_dataset([3] * 10 + [0] * 10)
        trainer = self._make_trainer()
        self.assertAlmostEqual(float(trainer.get_loss(dataset, 3, 0, no_batch=True)), 10.0)


class TestEvaluateSkipsUnreachableLengths(unittest.TestCase):
    """A split can be shorter than the padded width, which leaves some known_seq_len empty."""

    def _make_trainer(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.NextToken
        trainer.multiclass_next_token_with_order_permutation = True
        trainer.condition_feature = None
        trainer.cascade_len = 1
        trainer.cascade_target_count = 1
        trainer.evaluation_samples = 1
        trainer.device = torch.device("cpu")
        trainer.optimizer = None
        trainer.criterion = lambda prediction, target: prediction.new_tensor(
            float(prediction.size(0)))
        trainer.model = MagicMock()
        trainer.model.eval = MagicMock()
        trainer.model.side_effect = lambda start, masked, padding, known_cascade_len, cond=None: \
            torch.zeros(start.size(0), 3)
        return trainer

    def test_evaluate_handles_lengths_no_example_reaches(self):
        # Padded width 8, but nothing is longer than 3: known_seq_len 4..7 have no viable example.
        dataset = _make_dataset([0, 1, 2, 3] * 5, max_sequence_length=8, batch_size=4)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        trainer = self._make_trainer()
        # Both the sampled and the exhaustive path must skip them rather than form an empty batch.
        batched = trainer.evaluate(dataset, loader=loader)
        exhaustive = trainer.evaluate(dataset, loader=None)
        self.assertTrue(torch.isfinite(batched).all())
        self.assertTrue(torch.isfinite(exhaustive).all())


if __name__ == "__main__":
    unittest.main()
