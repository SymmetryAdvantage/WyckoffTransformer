import unittest
import torch
from ..dataset import AugmentedCascadeDataset

class TestCascadeData(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.seq_len = 5
        self.cascade_size = 2
        raw_data = torch.arange(1, self.batch_size * self.seq_len * self.cascade_size + 1, dtype=torch.int64).reshape(
            self.batch_size, self.seq_len, self.cascade_size)
        self.dataset = AugmentedCascadeDataset(
            data={
                "f0": raw_data[:, :, 0],
                "f1": raw_data[:, :, 1],
                "start": torch.zeros(self.batch_size, dtype=torch.int64),
            },
            cascade_order=("f0", "f1"),
            masks={"f0": 0, "f1": 0},
            pads={"f0": 255, "f1": 255},
            stops={"f0": 254, "f1": 254},
            num_classes={"f0": 256, "f1": 256},
            start_field="start",
            augmented_fields=None,
            batch_size=None,
        )
    
    def test_get_masked_cascade_data_predict_first_cascade(self):
        start_tokens, res, target = self.dataset.get_masked_cascade_data(known_seq_len=1, known_cascade_len=0)
        self.assertEqual(start_tokens.shape, torch.Size([3]))

        expected_res0 = torch.tensor([
            [1, 0],
            [11, 0],
            [21, 0],
        ], dtype=torch.int64)
        expected_res1 = torch.tensor([
            [2, 0],
            [12, 0],
            [22, 0],
        ], dtype=torch.int64)
        expected_target = torch.tensor([3, 13, 23], dtype=torch.int64)

        self.assertTrue(torch.equal(res[0], expected_res0))
        self.assertTrue(torch.equal(res[1], expected_res1))
        self.assertTrue(torch.equal(target, expected_target))

    def test_get_masked_cascade_data_predict_second_cascade(self):
        _, res, target = self.dataset.get_masked_cascade_data(known_seq_len=1, known_cascade_len=1)

        expected_res0 = torch.tensor([
            [1, 3],
            [11, 13],
            [21, 23],
        ], dtype=torch.int64)
        expected_res1 = torch.tensor([
            [2, 0],
            [12, 0],
            [22, 0],
        ], dtype=torch.int64)
        expected_target = torch.tensor([4, 14, 24], dtype=torch.int64)

        self.assertTrue(torch.equal(res[0], expected_res0))
        self.assertTrue(torch.equal(res[1], expected_res1))
        self.assertTrue(torch.equal(target, expected_target))