from pathlib import Path
import unittest

import torch


def _find_cdvae_checkpoint() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    prop_models = repo_root / "src" / "wyckoff_transformer" / "cdvae_evals" / "prop_models"
    for dataset in ("mp20", "carbon", "perovskite"):
        matches = sorted((prop_models / dataset).glob("*.ckpt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No CDVAE checkpoint found under {prop_models}")


class TestCdvaeCheckpointWeightsOnlyBehavior(unittest.TestCase):
    def test_default_torch_load_fails_with_weights_only_error(self) -> None:
        ckpt = _find_cdvae_checkpoint()

        with self.assertRaises(Exception) as ctx:
            torch.load(ckpt)

        self.assertEqual(type(ctx.exception).__name__, "UnpicklingError")
        self.assertIn("Weights only load failed", str(ctx.exception))
        self.assertIn("EarlyStopping", str(ctx.exception))

    def test_loads_when_weights_only_is_false(self) -> None:
        ckpt = _find_cdvae_checkpoint()

        loaded = torch.load(ckpt, weights_only=False)

        self.assertIsInstance(loaded, dict)
        self.assertIn("state_dict", loaded)


if __name__ == "__main__":
    unittest.main()
