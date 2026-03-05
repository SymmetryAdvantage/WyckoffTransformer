import unittest
import torch
from ..model import  get_masked_cascade_data

class TestCascadeData(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.seq_len = 5
        self.cascade_size = 2
        self.data = torch.arange(1, self.batch_size * self.seq_len * self.cascade_size + 1, dtype=torch.uint8).reshape(
            self.batch_size, self.seq_len, self.cascade_size)
        self.mask_indices = torch.zeros(self.cascade_size)
    
    def test_data(self):
        self.assert_((get_masked_cascade_data(self.data, self.mask_indices, 0, 0) == 0).all())

        res = get_masked_cascade_data(self.data, self.mask_indices, 1, 0)
        truth = torch.tensor([[[ 1.,  2.],
         [ 0.,  0.]],
        [[11., 12.],
         [ 0.,  0.]],
        [[21., 22.],
         [ 0.,  0.]]])
        self.assert_(torch.allclose(res, truth))

        res = get_masked_cascade_data(self.data, self.mask_indices, 1, 1)
        truth = torch.tensor([[[ 1.,  2.],
         [ 3.,  0.]],

        [[11., 12.],
         [13.,  0.]],

        [[21., 22.],
         [23.,  0.]]])
        self.assert_(torch.allclose(res, truth))