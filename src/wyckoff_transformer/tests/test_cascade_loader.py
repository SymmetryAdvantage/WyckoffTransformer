import unittest
import torch
from wyckoff_transformer.cascade.dataset import AugmentedCascadeLoader


class _DummyDataset:
    def __init__(self, num_examples: int):
        self.num_examples = num_examples
        self.augmented_storage_device = torch.device("cpu")
        self.pin_memory = False


class TestCascadeLoader(unittest.TestCase):
    def test_batch_size_equal_to_num_examples_does_not_recurse(self):
        """When batch_size == num_examples, get_next_batch must return full batches across epochs without RecursionError."""
        dataset = _DummyDataset(num_examples=10)
        loader = AugmentedCascadeLoader(dataset, batch_size=10, fix_batch_size=True)
        self.assertEqual(loader.batches_per_epoch, 1)
        batch0 = loader.get_next_batch()
        self.assertEqual(len(batch0), 10)
        batch1 = loader.get_next_batch()
        self.assertEqual(len(batch1), 10)
        self.assertFalse(torch.equal(batch0, batch1))

    def test_all_batches_cover_entire_permutation_in_epoch(self):
        """An epoch consisting of multiple batches must cover all examples without prematurely dropping the final batch."""
        torch.manual_seed(42)
        dataset = _DummyDataset(num_examples=32)
        loader = AugmentedCascadeLoader(dataset, batch_size=8, fix_batch_size=True)
        initial_order = loader.this_shuffle_order.clone()
        batches = [loader.get_next_batch() for _ in range(loader.batches_per_epoch)]
        all_returned = torch.cat(batches)
        self.assertEqual(len(all_returned), 32)
        torch.testing.assert_close(all_returned, initial_order, rtol=0, atol=0)

    def test_partial_batch_dropped_when_fix_batch_size_is_true(self):
        """When fix_batch_size=True, a trailing partial batch is dropped and a fresh full batch from next epoch is returned."""
        torch.manual_seed(42)
        dataset = _DummyDataset(num_examples=15)
        loader = AugmentedCascadeLoader(dataset, batch_size=10, fix_batch_size=True)
        self.assertEqual(loader.batches_per_epoch, 1)
        initial_order = loader.this_shuffle_order.clone()
        batch0 = loader.get_next_batch()
        self.assertEqual(len(batch0), 10)
        torch.testing.assert_close(batch0, initial_order[:10], rtol=0, atol=0)
        batch1 = loader.get_next_batch()
        self.assertEqual(len(batch1), 10)

    def test_partial_batch_kept_when_fix_batch_size_is_false(self):
        """When fix_batch_size=False, trailing partial batch is returned with remaining size."""
        torch.manual_seed(42)
        dataset = _DummyDataset(num_examples=15)
        loader = AugmentedCascadeLoader(dataset, batch_size=10, fix_batch_size=False)
        self.assertEqual(loader.batches_per_epoch, 2)
        batch0 = loader.get_next_batch()
        self.assertEqual(len(batch0), 10)
        batch1 = loader.get_next_batch()
        self.assertEqual(len(batch1), 5)
        batch2 = loader.get_next_batch()
        self.assertEqual(len(batch2), 10)


class TestAugmentedStorageBounds(unittest.TestCase):
    def test_int16_tensor_against_int64_storage_does_not_overflow(self):
        """Comparing an int16 tensor max to torch.iinfo(int64).max must not overflow."""
        from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset
        data = {
            "elements": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "field_a": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "start": torch.tensor([1, 1], dtype=torch.int64),
            "pure_sequence_length": torch.tensor([2, 2], dtype=torch.int64),
            # Augmented variants stored in int16 with normal values
            "field_a_augmented": [[torch.tensor([[1, 2], [80, 2]], dtype=torch.int16)],
                                  [torch.tensor([[1, 2], [80, 2]], dtype=torch.int16)]],
            "field_a_variants": torch.tensor([1, 1], dtype=torch.int64),
        }
        dataset = AugmentedCascadeDataset(
            data=data,
            cascade_order=["elements", "field_a"],
            masks={"elements": 0, "field_a": 0},
            pads={"elements": 0, "field_a": 0},
            stops={"elements": 0, "field_a": 0},
            num_classes={"elements": 100, "field_a": 100},
            start_field="start",
            augmented_fields=["field_a"],
            batch_size=2,
            augmented_storage_dtype=None,  # defaults to dtype = int64
        )
        self.assertEqual(dataset.num_examples, 2)

