import torch
import unittest
from unittest.mock import MagicMock
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset

class TestCalibrateEmptyTail(unittest.TestCase):
    def test_calibrate_empty_tail(self):
        # Mock model
        model = MagicMock(spec=torch.nn.Module)
        param = torch.nn.Parameter(torch.ones(1))
        model.parameters.side_effect = lambda: iter([param])
        # Mock model return value: a logit tensor [batch_size, num_classes]
        # For NextToken, batch_size depends on the dataset split
        model.side_effect = lambda start, data, padding, cascade_idx, cond=None: torch.randn(start.size(0), 10)

        # Mock dataset
        dataset = MagicMock(spec=AugmentedCascadeDataset)
        dataset.cascade_order = ("field1",)
        dataset.max_sequence_length = 2

        # Mock get_masked_multiclass_cascade_data to return enough samples to ALWAYS be >= threshold
        batch_size = 110
        start_tokens = torch.zeros(batch_size, dtype=torch.long)
        masked_data = [torch.zeros(batch_size, 1, dtype=torch.long)]
        target = torch.zeros(batch_size, dtype=torch.long)
        chosen_indices = torch.ones(batch_size, dtype=torch.bool)

        dataset.get_masked_multiclass_cascade_data.return_value = (
            start_tokens, masked_data, target, chosen_indices)
        
        generator = WyckoffGenerator(
            model=model,
            cascade_order=("field1",),
            cascade_is_target={"field1": True},
            token_engineers={},
            masks={"field1": 0},
            max_sequence_len=2
        )
        
        # This should NOT raise ValueError: torch.cat(): expected a non-empty list of Tensors
        # With threshold 100, and target.size(0) = 110, it will always use specific calibrators
        # and tail_predictions will be empty.
        try:
            generator.calibrate(dataset, calibration_element_count_threshold=100)
        except ValueError as e:
            if "torch.cat(): expected a non-empty list of Tensors" in str(e):
                self.fail("WyckoffGenerator.calibrate failed with empty tail_predictions")
            raise e

    def test_a_length_no_structure_reaches_is_skipped(self):
        """The padded width can exceed every structure of the split being calibrated on."""
        model = MagicMock(spec=torch.nn.Module)
        param = torch.nn.Parameter(torch.ones(1))
        model.parameters.side_effect = lambda: iter([param])

        def forward(start, data, padding, cascade_idx, cond=None):
            if start.size(0) == 0:
                raise RuntimeError("cannot reshape tensor of 0 elements")
            return torch.randn(start.size(0), 10)
        model.side_effect = forward

        def batch(size):
            return (torch.zeros(size, dtype=torch.long), [torch.zeros(size, 1, dtype=torch.long)],
                    torch.zeros(size, dtype=torch.long), torch.ones(size, dtype=torch.bool))

        dataset = MagicMock(spec=AugmentedCascadeDataset)
        dataset.cascade_order = ("field1",)
        dataset.max_sequence_length = 3
        # Lengths 0 and 1 have examples; nothing in the split is long enough for length 2.
        dataset.get_masked_multiclass_cascade_data.side_effect = (
            lambda known_seq_len, *args, **kwargs: batch(0 if known_seq_len == 2 else 110))

        generator = WyckoffGenerator(
            model=model, cascade_order=("field1",), cascade_is_target={"field1": True},
            token_engineers={}, masks={"field1": 0}, max_sequence_len=3)
        generator.calibrate(dataset, calibration_element_count_threshold=100)
        self.assertEqual(len(generator.calibrators[0]), 2)
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
