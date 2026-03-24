import unittest
import pytest
import warnings
import torch
import json
import gzip
import pickle
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from ..trainer import WyckoffTrainer
from ..evaluation.cdvae_metrics import timed_smact_validity_from_record
from ..evaluation.statistical_evaluator import StatisticalEvaluator

@pytest.mark.needs_cache
class TestTrainedModelIOI8TYCX(unittest.TestCase):
    def setUp(self):
        self.run_path = Path(__file__).resolve().parent / "fixtures" / "ioi8tycx"
        if not self.run_path.exists():
            self.skipTest("Run ioi8tycx not found")
            
        config = OmegaConf.load(self.run_path / "config.yaml")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="This process.*is multi-threaded, use of fork\\(\\) may lead to deadlocks in the child.")
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

    @pytest.mark.filterwarnings("ignore:No Pauling electronegativity for .*")
    @pytest.mark.filterwarnings("ignore:This process.*is multi-threaded, use of fork\\(\\) may lead to deadlocks in the child\\.:DeprecationWarning")
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
        
        metrics_file = Path(__file__).resolve().parent / "fixtures" / "ioi8tycx_reference_metrics.json"
        with open(metrics_file, "r") as f:
            ref_metrics = json.load(f)
            
        smact_valid_count = sum(1 for wp in valid_wp if timed_smact_validity_from_record(wp))
        smact_validity = smact_valid_count / len(valid_wp) if len(valid_wp) > 0 else 0
        
        df = pd.DataFrame(valid_wp)
        df['spacegroup_number'] = df['group']
        p1_percent = (df['spacegroup_number'] == 1).mean()
        
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
