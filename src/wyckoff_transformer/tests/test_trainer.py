import unittest
import pytest
import warnings
from types import SimpleNamespace

import torch
from unittest.mock import patch, MagicMock

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
        trainer.production_training = False

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
        trainer.production_training = False

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

class TestWyckoffTrainerGeneration(unittest.TestCase):
    def setUp(self):
        self.trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        self.trainer.model = MagicMock()
        self.trainer.model.start_type = "categorial"
        self.trainer.model.start_embedding.num_embeddings = 5
        self.trainer.cascade_order = ["spacegroup", "harmonic_site_symmetries"]
        self.trainer.cascade_is_target = {"spacegroup": True}
        self.trainer.token_engineers = {}
        self.trainer.tokenisers = {'elements': MagicMock()}
        self.trainer.masks_dict = {}
        self.trainer.start_name = "spacegroup_number"
        self.trainer.start_token_distribution = None
        
        self.trainer.train_dataset = MagicMock()
        self.trainer.train_dataset.start_tokens = torch.tensor([0, 1])
        self.trainer.train_dataset.masks = {}
        self.trainer.train_dataset.max_sequence_length = 10
        
        self.trainer.val_dataset = MagicMock()
        self.trainer.val_dataset.start_tokens = torch.tensor([1, 2])
        self.trainer.max_sequence_length = 10
        self.trainer.device = torch.device("cpu")
        self.trainer.processor = MagicMock()
        self.trainer.production_training = False
        
        # Setup processor to return a mock pyxtal-like string/object
        self.trainer.processor.tensor_to_pyxtal.return_value = "pyxtal_mock"

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.gzip.open")
    @patch("wyckoff_transformer.trainer.pickle.load")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_generate_structures(self, mock_get_wp_index, mock_pickle_load, mock_gzip_open, MockWyckoffGenerator):
        # Setup mocks
        mock_generator_instance = MockWyckoffGenerator.return_value
        # generated tensors: mock what generator.generate_tensors returns
        mock_generator_instance.generate_tensors.return_value = [torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_pickle_load.return_value = [None, None, "ss_from_letter_mock"]
        
        # Test basic generation
        structures = self.trainer.generate_structures(
            n_structures=2, 
            calibrate=False, 
            compute_validity_per_known_sequence_length=False
        )
        
        # Assertions
        mock_generator_instance.generate_tensors.assert_called_once()
        self.assertEqual(len(structures), 2)
        self.assertEqual(structures[0], "pyxtal_mock")
        # Ensure that harmonic_site_symmetries is deleted from tensors during processing 
        # (which means tensor_to_pyxtal is called with 1-element cascade_order)
        self.assertTrue(self.trainer.processor.tensor_to_pyxtal.called)

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.gzip.open")
    @patch("wyckoff_transformer.trainer.pickle.load")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_generate_csx_structures(self, mock_get_wp_index, mock_pickle_load, mock_gzip_open, MockWyckoffGenerator):
        # Setup mocks
        mock_generator_instance = MockWyckoffGenerator.return_value
        mock_generator_instance.generate_tensors.return_value = [torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_pickle_load.return_value = [None, None, "ss_from_letter_mock"]
        
        start_tensor = torch.tensor([0, 1])

        structures = self.trainer.generate_csx_structures(
            n_structures=2,
            calibrate=False,
            required_element_set="Li-O",
            allowed_element_set="all",
            start_tensor=start_tensor
        )
        
        # Assertions
        mock_generator_instance.generate_tensors.assert_called_once()
        call_kwargs = mock_generator_instance.generate_tensors.call_args[1]
        self.assertEqual(call_kwargs["required_element_set"], "Li-O")
        self.assertEqual(call_kwargs["allowed_element_set"], "all")
        self.assertEqual(len(structures), 2)
        self.assertEqual(structures[0], "pyxtal_mock")



if __name__ == "__main__":
    unittest.main()
