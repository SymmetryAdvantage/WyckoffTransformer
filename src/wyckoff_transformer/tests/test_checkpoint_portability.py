"""A checkpoint moves between a compiled GPU trainer and an uncompiled CPU generator.

Training compiles the model and runs on CUDA; generation has neither reason to
compile nor a GPU to load onto. Both differences rename or misplace the tensors,
so `load_model_weights` has to reconcile them.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from wyckoff_transformer.trainer import _COMPILE_PREFIX, load_model_weights


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def _save(state_dict, directory: Path) -> Path:
    path = directory / "best_model_params.pt"
    torch.save(state_dict, path)
    return path


class TestLoadModelWeights(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = _tiny_model()

    def assert_same_weights(self, loaded: nn.Module):
        expected = self.source.state_dict()
        actual = {k.removeprefix(_COMPILE_PREFIX): v for k, v in loaded.state_dict().items()}
        self.assertEqual(sorted(actual), sorted(expected))
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value)

    def test_plain_checkpoint_into_plain_model(self):
        path = _save(self.source.state_dict(), self.tmp_path)
        destination = _tiny_model()
        load_model_weights(destination, path)
        self.assert_same_weights(destination)

    def test_compiled_checkpoint_into_plain_model(self):
        """The case that blocked CPU generation: trained with compile, sampled without."""
        compiled = {_COMPILE_PREFIX + k: v for k, v in self.source.state_dict().items()}
        path = _save(compiled, self.tmp_path)
        destination = _tiny_model()
        load_model_weights(destination, path)
        self.assert_same_weights(destination)

    def test_plain_checkpoint_into_compiled_model(self):
        path = _save(self.source.state_dict(), self.tmp_path)
        destination = torch.compile(_tiny_model())
        load_model_weights(destination, path)
        self.assert_same_weights(destination)

    def test_compiled_checkpoint_into_compiled_model(self):
        compiled = {_COMPILE_PREFIX + k: v for k, v in self.source.state_dict().items()}
        path = _save(compiled, self.tmp_path)
        destination = torch.compile(_tiny_model())
        load_model_weights(destination, path)
        self.assert_same_weights(destination)

    def test_cuda_checkpoint_loads_on_cpu(self):
        """`torch.load` refuses a CUDA-tagged storage on a CPU-only host without map_location."""
        if not torch.cuda.is_available():
            self.skipTest("needs a GPU to write a CUDA-tagged checkpoint")
        cuda_state = {k: v.cuda() for k, v in self.source.state_dict().items()}
        path = _save(cuda_state, self.tmp_path)
        destination = _tiny_model()
        load_model_weights(destination, path, device="cpu")
        self.assert_same_weights(destination)
        for value in destination.state_dict().values():
            self.assertEqual(value.device.type, "cpu")


class TestGenerationOnlySchedule(unittest.TestCase):
    """A run trained under a step-indexed schedule can be loaded without its dataset.

    `warmup_stable_decay` sizes itself from `epochs * batches_per_epoch`, so it needs a
    training set that dataset-free generation deliberately does not load. Nothing steps
    the schedule while sampling, so it is skipped rather than treated as a misconfiguration.
    """

    def setUp(self):
        from omegaconf import OmegaConf

        self.run_path = Path(__file__).resolve().parent / "fixtures" / "ioi8tycx"
        if not self.run_path.exists():
            self.skipTest("Run ioi8tycx not found")
        self.config = OmegaConf.load(self.run_path / "config.yaml")
        self.config.optimisation.scheduler = OmegaConf.create({
            "module": "wyckoff_transformer.schedules",
            "name": "warmup_stable_decay",
            "config": {"warmup_fraction": 0.002, "decay_fraction": 0.1,
                       "final_lr_fraction": 0.0, "decay_shape": "1-sqrt"},
        })

    def test_step_indexed_schedule_is_skipped_without_a_training_set(self):
        from wyckoff_transformer.trainer import WyckoffTrainer

        trainer = WyckoffTrainer.from_config(
            self.config,
            device=torch.device("cpu"),
            use_cached_tensors=False,
            run_path=self.run_path,
            load_datasets=False,
        )
        self.assertIsNone(trainer.scheduler)
        self.assertFalse(trainer.scheduler_steps_per_batch)

    def test_the_schedule_is_still_sized_when_the_training_set_is_loaded(self):
        """Sampling is the only exemption; a run that loads its data still gets a schedule."""
        from wyckoff_transformer.trainer import WyckoffTrainer

        trainer = WyckoffTrainer.from_config(
            self.config,
            device=torch.device("cpu"),
            use_cached_tensors=False,
            run_path=self.run_path,
            load_datasets=True,
        )
        self.assertIsNotNone(trainer.scheduler)
        self.assertTrue(trainer.scheduler_steps_per_batch)
        self.assertGreater(trainer.scheduler_total_steps, 0)


if __name__ == "__main__":
    unittest.main()
