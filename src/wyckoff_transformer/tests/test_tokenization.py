import unittest
import torch
from ..tokenization import argsort_multiple, SpaceGroupEncoder


class TestArgsortMultiple(unittest.TestCase):
    @torch.no_grad()
    def test_argsort_multiple_single_tensor_respects_dim(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        expected_output = torch.tensor([[0, 0, 0], [1, 1, 1]])
        output = argsort_multiple(tensor1, dim=0)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_two_tensors_lexicographic_no_collision(self):
        # Crafted so radix=max(second_key) would collide, but radix=max+1 stays injective.
        tensor1 = torch.tensor([[1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.uint8)
        tensor2 = torch.tensor([[0, 4, 0, 4, 0, 4, 0, 4]], dtype=torch.uint8)
        expected_output = torch.tensor([[1, 0, 3, 2, 5, 4, 7, 6]])
        output = argsort_multiple(tensor1, tensor2, dim=1)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_two_tensors_respects_dim_zero(self):
        tensor1 = torch.tensor([[2, 1, 0], [1, 1, 1]], dtype=torch.uint8)
        tensor2 = torch.tensor([[0, 2, 3], [5, 1, 0]], dtype=torch.uint8)
        expected_output = torch.tensor([[1, 1, 0], [0, 0, 1]])
        output = argsort_multiple(tensor1, tensor2, dim=0)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_rejects_non_uint8_two_tensor_input(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        tensor2 = torch.tensor([[9, 8, 7], [6, 5, 4]])
        with self.assertRaises(NotImplementedError):
            argsort_multiple(tensor1, tensor2, dim=1)

    @torch.no_grad()
    def test_argsort_multiple_invalid_arity(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]], dtype=torch.uint8)
        tensor2 = torch.tensor([[9, 8, 7], [6, 5, 4]], dtype=torch.uint8)
        tensor3 = torch.tensor([[3, 2, 1], [4, 5, 6]], dtype=torch.uint8)
        with self.assertRaises(NotImplementedError):
            argsort_multiple(tensor1, tensor2, tensor3, dim=1)

class TestSpaceGroupEncoder(unittest.TestCase):
    def test_space_group_encoder(self):
        """
        By design, all space groups must have uniqe encodings
        """
        all_group_encoder = SpaceGroupEncoder.from_sg_set(range(1, 231))
        all_groups = set()
        for group_number in range(1, 231):
            group = tuple(all_group_encoder[group_number])
            if group in all_groups:
                raise ValueError(f"Duplicate group: {group}")
            all_groups.add(group)        


if __name__ == '__main__':
    unittest.main()
