import torch
import unittest
from unittest.mock import MagicMock
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, TargetClass

class TestCalibrateEmptyTail(unittest.TestCase):
    def test_calibrate_empty_tail(self):
        # Mock model
        model = MagicMock(spec=torch.nn.Module)
        param = torch.nn.Parameter(torch.ones(1))
        model.parameters.side_effect = lambda: iter([param])
        # Mock model return value: a logit tensor [batch_size, num_classes]
        # For NextToken, batch_size depends on the dataset split
        model.side_effect = lambda start, data, padding, cascade_idx: torch.randn(start.size(0), 10)
        
        # Mock dataset
        dataset = MagicMock(spec=AugmentedCascadeDataset)
        dataset.cascade_order = ("field1",)
        dataset.max_sequence_length = 2
        
        # Mock get_masked_multiclass_cascade_data to return enough samples to ALWAYS be >= threshold
        # start_tokens: [batch_size]
        # masked_data: list of [batch_size, seq_len]
        # target: [batch_size]
        batch_size = 110
        start_tokens = torch.zeros(batch_size, dtype=torch.long)
        masked_data = [torch.zeros(batch_size, 1, dtype=torch.long)]
        target = torch.zeros(batch_size, dtype=torch.long)
        
        dataset.get_masked_multiclass_cascade_data.return_value = (start_tokens, masked_data, target)
        
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

if __name__ == "__main__":
    unittest.main()
