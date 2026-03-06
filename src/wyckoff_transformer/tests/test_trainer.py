import unittest
from types import SimpleNamespace

import torch

from ..trainer import WyckoffTrainer

# Intentionally tests a private helper to validate serialized distribution semantics.
# pylint: disable=protected-access


class TestBuildStartTokenDistribution(unittest.TestCase):
    def test_build_start_token_distribution_categorial(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.train_dataset = SimpleNamespace(
            start_tokens=torch.tensor([0, 2, 2], dtype=torch.int64)
        )
        trainer.val_dataset = SimpleNamespace(
            start_tokens=torch.tensor([1, 2], dtype=torch.int64)
        )
        trainer.model = SimpleNamespace(
            start_type="categorial",
            start_embedding=SimpleNamespace(num_embeddings=5),
        )
        trainer.start_name = "spacegroup_number"
        trainer.max_sequence_length = 13

        distribution = trainer._build_start_token_distribution()

        self.assertEqual(distribution["start_name"], "spacegroup_number")
        self.assertEqual(distribution["start_type"], "categorial")
        self.assertEqual(distribution["max_sequence_length"], 13)
        self.assertEqual(distribution["counts"], [1, 1, 3, 0, 0])

    def test_build_start_token_distribution_one_hot(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.train_dataset = SimpleNamespace(
            start_tokens=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        trainer.val_dataset = SimpleNamespace(
            start_tokens=torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        trainer.model = SimpleNamespace(start_type="one_hot")
        trainer.start_name = "spacegroup_number"
        trainer.max_sequence_length = 7

        distribution = trainer._build_start_token_distribution()

        self.assertEqual(distribution["start_name"], "spacegroup_number")
        self.assertEqual(distribution["start_type"], "one_hot")
        self.assertEqual(distribution["max_sequence_length"], 7)

        vector_count_map = {
            tuple(vector): count
            for vector, count in zip(distribution["vectors"], distribution["counts"])
        }
        self.assertEqual(vector_count_map[(1.0, 0.0, 0.0)], 2)
        self.assertEqual(vector_count_map[(0.0, 1.0, 0.0)], 2)
        self.assertEqual(vector_count_map[(0.0, 0.0, 1.0)], 1)


if __name__ == "__main__":
    unittest.main()
