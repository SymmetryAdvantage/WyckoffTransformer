from pathlib import Path
import unittest

from ..evaluation.cdvae_metrics import load_model


def _bundled_prop_model_dirs() -> list[Path]:
    prop_models_root = (
        Path(__file__).resolve().parents[1] / "cdvae_evals" / "prop_models"
    )
    if not prop_models_root.exists():
        raise FileNotFoundError(f"Bundled prop_models directory not found: {prop_models_root}")

    model_dirs = sorted(path for path in prop_models_root.iterdir() if path.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No bundled property model directories found in {prop_models_root}")

    return model_dirs


class TestCdvaeBundledPropertyModels(unittest.TestCase):
    def test_load_model_succeeds_for_all_bundled_property_models(self) -> None:
        for model_dir in _bundled_prop_model_dirs():
            with self.subTest(model_dir=model_dir.name):
                model = load_model(model_dir)

                self.assertIsNotNone(model)
                self.assertTrue(hasattr(model, "scaler"))
                self.assertTrue(hasattr(model, "lattice_scaler"))


if __name__ == "__main__":
    unittest.main()