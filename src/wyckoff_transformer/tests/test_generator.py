import unittest
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

from wyckoff_transformer.trainer import WyckoffTrainer
from wyckoff_transformer.generator import WyckoffGenerator

@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded, use of fork().*:DeprecationWarning")
class TestGeneratorIOI8TYCX(unittest.TestCase):
    def setUp(self):
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
        self.generator = WyckoffGenerator(
            self.trainer.model, 
            self.trainer.cascade_order, 
            self.trainer.cascade_is_target, 
            self.trainer.token_engineers,
            self.trainer.masks_dict, 
            self.trainer.max_sequence_length
        )

    @pytest.mark.filterwarnings("ignore:No Pauling electronegativity for .*")
    def test_generate_tensors(self):
        n_structures = 10
        start_tensor = self.trainer._sample_start_tokens_from_distribution(n_structures)
        tensors = self.generator.generate_tensors(start_tensor, compute_validity=False)
        
        # Tensors is a list of tensors for each cascade field
        self.assertEqual(len(tensors), len(self.trainer.cascade_order))
        for t in tensors:
            self.assertEqual(t.size(0), n_structures)
            self.assertEqual(t.size(1), self.trainer.max_sequence_length)

if __name__ == "__main__":
    unittest.main()
