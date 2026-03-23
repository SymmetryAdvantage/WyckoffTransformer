import unittest
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


class TestTrainedModelIOI8TYCX(unittest.TestCase):
    def setUp(self):
        from omegaconf import OmegaConf
        from pathlib import Path
        self.run_path = Path(__file__).resolve().parent / "fixtures" / "ioi8tycx"
        if not self.run_path.exists():
            self.skipTest("Run ioi8tycx not found")
            
        config = OmegaConf.load(self.run_path / "config.yaml")

        self.trainer = WyckoffTrainer.from_config(
            config,
            device=torch.device("cpu"),
            use_cached_tensors=False,
            run_path=self.run_path,
            load_datasets=True
        )
        self.trainer.model.load_state_dict(
            torch.load(self.run_path / "best_model_params.pt", map_location="cpu", weights_only=True)
        )

    def test_formal_format(self):
        n_structures = 1000
        generated_wp = self.trainer.generate_structures(
            n_structures=n_structures,
            calibrate=False,
            compute_validity_per_known_sequence_length=False
        )
        
        valid_wp = [wp for wp in generated_wp if wp is not None]
        formal_validity = len(valid_wp) / n_structures
        
        self.assertGreater(formal_validity, 0.0)
        
        import json
        from pathlib import Path
        metrics_file = Path(__file__).resolve().parent / "fixtures" / "ioi8tycx_reference_metrics.json"
        with open(metrics_file, "r") as f:
            ref_metrics = json.load(f)
            
        from wyckoff_transformer.evaluation.cdvae_metrics import timed_smact_validity_from_record
        from wyckoff_transformer.evaluation.statistical_evaluator import StatisticalEvaluator
        import pandas as pd
        
        smact_valid_count = sum(1 for wp in valid_wp if timed_smact_validity_from_record(wp))
        smact_validity = smact_valid_count / len(valid_wp) if len(valid_wp) > 0 else 0
        
        df = pd.DataFrame(valid_wp)
        df['spacegroup_number'] = df['group']
        p1_percent = (df['spacegroup_number'] == 1).mean()
        
        import gzip
        import pickle
        from omegaconf import OmegaConf
        config = OmegaConf.load(self.run_path / "config.yaml")
        data_cache_path = Path(__file__).resolve().parents[3] / "cache" / config.dataset / "data.pkl.gz"
        with gzip.open(data_cache_path, "rb") as f:
            datasets_pd = pickle.load(f)
            
        if 'structure' not in datasets_pd['test']:
            class MockStructure:
                def __len__(self): return 1
                @property
                def density(self): return 1.0
            datasets_pd['test']['structure'] = [MockStructure()] * len(datasets_pd['test'])
            
        evaluator = StatisticalEvaluator(datasets_pd['test'])
        sg_chi2 = evaluator.get_sg_chi2(df)
        elements_emd = evaluator.get_num_elements_emd(df)
        sites_emd = evaluator.get_num_sites_emd(df)
        
        # Test that metrics are similar to reference within acceptable variance (larger sample -> smaller delta needed)
        self.assertAlmostEqual(formal_validity, ref_metrics["formal_validity"], delta=0.1)
        self.assertAlmostEqual(smact_validity, ref_metrics["smact_validity"], delta=0.1)
        self.assertAlmostEqual(p1_percent, ref_metrics["p1_percent"], delta=0.05)
        self.assertAlmostEqual(sg_chi2, ref_metrics["sg_chi2"], delta=0.2)
        self.assertAlmostEqual(elements_emd, ref_metrics["elements_emd"], delta=0.3)
        self.assertAlmostEqual(sites_emd, ref_metrics["sites_emd"], delta=0.3)


if __name__ == "__main__":
    unittest.main()
