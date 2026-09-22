import json
import unittest
from pathlib import Path
import tempfile
import torch
import pandas as pd

import sys
scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from slice_dataset_by_ehull import slice_safetensors, slice_dataframe_cache
from wyckoff_transformer.dataset_cache import load_cache, save_cache
from wyckoff_transformer.tokenization import load_tensor_cache, save_tensor_cache


class TestSliceDatasetByEhull(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_slice_safetensors(self):
        source_path = self.tmp_path / "source.safetensors"
        target_path = self.tmp_path / "target.safetensors"

        # Mock tensors
        tensors = {
            "train": {
                "energy_above_hull": torch.tensor([0.02, 0.15, 0.08, 0.25, 0.00]),
                "spacegroup": torch.tensor([1, 2, 3, 4, 5]),
                "augmented_list": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3]), torch.tensor([4]), torch.tensor([5])],
            },
            "val": {
                "energy_above_hull": torch.tensor([0.05, 0.12]),
                "spacegroup": torch.tensor([10, 20]),
                "augmented_list": [torch.tensor([10]), torch.tensor([20])],
            }
        }
        save_tensor_cache(tensors, source_path)

        masks = slice_safetensors(source_path, target_path, ehull_cutoff=0.1)

        self.assertTrue(target_path.exists())
        sliced = load_tensor_cache(target_path)

        # Train: indices 0, 2, 4 <= 0.1
        self.assertEqual(len(sliced["train"]["energy_above_hull"]), 3)
        torch.testing.assert_close(sliced["train"]["spacegroup"], torch.tensor([1, 3, 5]))
        self.assertEqual(len(sliced["train"]["augmented_list"]), 3)
        torch.testing.assert_close(sliced["train"]["augmented_list"][0], torch.tensor([1]))
        torch.testing.assert_close(sliced["train"]["augmented_list"][1], torch.tensor([3]))
        torch.testing.assert_close(sliced["train"]["augmented_list"][2], torch.tensor([5]))

        # Val: index 0 <= 0.1
        self.assertEqual(len(sliced["val"]["energy_above_hull"]), 1)
        torch.testing.assert_close(sliced["val"]["spacegroup"], torch.tensor([10]))
        torch.testing.assert_close(sliced["val"]["augmented_list"][0], torch.tensor([10]))

    def test_slice_dataframe_cache(self):
        source_path = self.tmp_path / "source"
        target_path = self.tmp_path / "target"

        df_train = pd.DataFrame({
            "energy_above_hull": [0.01, 0.20, 0.05],
            "formula": ["NaCl", "Fe2O3", "TiO2"],
        }, index=["id1", "id2", "id3"])

        df_val = pd.DataFrame({
            "energy_above_hull": [0.08, 0.11],
            "formula": ["SiO2", "Al2O3"],
        }, index=["id4", "id5"])

        save_cache({"train": df_train, "val": df_val}, source_path)

        filtered = slice_dataframe_cache(source_path, target_path, ehull_cutoff=0.1)

        self.assertEqual(load_cache(target_path).keys(), {"train", "val"})
        self.assertEqual(len(filtered["train"]), 2)
        self.assertEqual(list(filtered["train"].index), ["id1", "id3"])
        self.assertEqual(len(filtered["val"]), 1)
        self.assertEqual(list(filtered["val"].index), ["id4"])
