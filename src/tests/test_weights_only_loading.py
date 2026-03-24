from pathlib import Path
import unittest

import pytest
import torch


def _find_cdvae_checkpoint() -> Path:
    cdvae_property_models = pytest.importorskip("cdvae_property_models")
    prop_models = Path(cdvae_property_models.__file__).resolve().parent / "prop_models"
    for dataset in ("mp20", "carbon", "perovskite"):
        matches = sorted((prop_models / dataset).glob("*.ckpt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No CDVAE checkpoint found under {prop_models}")


class TestCdvaeCheckpointWeightsOnlyBehavior(unittest.TestCase):
    def test_default_torch_load_fails_with_weights_only_error(self) -> None:
        ckpt = _find_cdvae_checkpoint()

        with self.assertRaises(Exception) as ctx:
            torch.load(ckpt, map_location='cpu')

        self.assertIn(type(ctx.exception).__name__, ["UnpicklingError", "RuntimeError"])

        self.assertIn("Weights only load failed", str(ctx.exception))
        self.assertIn("EarlyStopping", str(ctx.exception))

    def test_loads_when_weights_only_is_false(self) -> None:
        ckpt = _find_cdvae_checkpoint()

        loaded = torch.load(ckpt, weights_only=False, map_location='cpu')

        self.assertIsInstance(loaded, dict)
        self.assertIn("state_dict", loaded)


if __name__ == "__main__":
    unittest.main()
