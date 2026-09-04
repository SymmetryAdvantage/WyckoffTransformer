"""A crashed run picks up where it left off, and lands where it would have without the crash.

Before this, the only thing a run wrote was `best_model_params.pt` -- weights, and only on
epochs that improved. The optimiser moments, the schedule's step counter, the RNG, the loader's
shuffle position and the early-stopping bookkeeping all lived in locals of `train()`, so a
crash cost the whole run. The tests here hold `last_checkpoint.pt` to the standard that makes
`--resume` worth having: continuing from it must produce the same weights as never having
stopped, not merely weights of the same quality.
"""
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import schedulefree
import torch
from omegaconf import OmegaConf
from torch import nn

from wyckoff_transformer.cascade.dataset import (
    AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass)
from wyckoff_transformer.schedules import warmup_stable_decay
from wyckoff_transformer.trainer import (
    CHECKPOINT_FILENAME, CHECKPOINT_FORMAT_VERSION, WyckoffTrainer, atomic_torch_save,
    check_resume_config, train_from_config)

MAX_SEQ = 6
N_CLASSES = 8
SEED = 7


class _Crash(RuntimeError):
    """Stands in for whatever actually kills a run: OOM, a pre-empted node, a power cut."""


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, N_CLASSES)

    def forward(self, start_tokens, masked_data, padding_mask, known_cascade_len, cond=None):
        return self.head(start_tokens.float().unsqueeze(-1))


def _make_dataset(batch_size):
    lengths = [1, 1, 2, 2, 3, 3, 4, 5] * 4
    n = len(lengths)
    rows = []
    for length in lengths:
        row = torch.randint(0, 5, (MAX_SEQ,))
        row[length] = 7
        row[length + 1:] = 5
        rows.append(row)
    data = {"field1": torch.stack(rows).to(torch.int64),
            "spacegroup": torch.arange(n, dtype=torch.int64) % 3,
            "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64)}
    return AugmentedCascadeDataset(
        data=data, cascade_order=("field1",), masks={"field1": 6}, pads={"field1": 5},
        stops={"field1": 7}, num_classes={"field1": N_CLASSES}, start_field="spacegroup",
        augmented_fields=None, batch_size=batch_size)


def _make_trainer(run_path: Path, epochs: int, resume: bool = False,
                  optimiser: str = "adamw", scheduled: bool = True,
                  checkpoint_period: int = 1, validation_period: int = 1) -> WyckoffTrainer:
    """A trainer whose whole state is small enough to compare exactly, seeded reproducibly.

    Seeding here rather than in the tests is what makes two separately built trainers start
    from the same weights, the same shuffle order and the same RNG stream -- without which
    "the resumed run matches the uninterrupted one" would not be a statement about resuming.
    """
    torch.manual_seed(SEED)
    random.seed(SEED)
    trainer = WyckoffTrainer.__new__(WyckoffTrainer)
    trainer.target = TargetClass.NextToken
    trainer.multiclass_next_token_with_order_permutation = True
    trainer.condition_feature = None
    trainer.cascade_len = 1
    trainer.cascade_target_count = 1
    trainer.cascade_order = ("field1",)
    trainer.evaluation_samples = 1
    trainer.device = torch.device("cpu")
    trainer.clip_grad_norm = None
    trainer.criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer.model = _TinyModel()
    trainer.train_dataset = _make_dataset(batch_size=8)
    trainer.val_dataset = _make_dataset(batch_size=8)
    trainer.test_dataset = None
    trainer.train_loader = AugmentedCascadeLoader.from_dataset(trainer.train_dataset)
    trainer.val_loader = AugmentedCascadeLoader.from_dataset(trainer.val_dataset)
    trainer.test_loader = None
    trainer.max_sequence_length = trainer.train_dataset.max_sequence_length
    if optimiser == "adamw":
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.05)
    elif optimiser == "schedule_free":
        trainer.optimizer = schedulefree.AdamWScheduleFree(
            trainer.model.parameters(), lr=0.05, warmup_steps=2)
    else:
        raise ValueError(optimiser)
    if scheduled:
        total_steps = epochs * trainer.train_loader.batches_per_epoch
        trainer.scheduler = warmup_stable_decay(trainer.optimizer, total_steps=total_steps)
        trainer.scheduler_steps_per_batch = True
        trainer.scheduler_total_steps = total_steps
    else:
        trainer.scheduler = None
        trainer.scheduler_steps_per_batch = False
        trainer.scheduler_total_steps = None
    trainer.epochs = epochs
    trainer.validation_period = validation_period
    trainer.checkpoint_period = checkpoint_period
    trainer.early_stopping_patience_epochs = epochs * 10
    trainer.production_training = False
    trainer.run_path = run_path
    trainer.resume = resume
    # Bypasses the dataset scan; the file it writes plays no part in resuming.
    trainer.start_token_distribution = {"start_name": "spacegroup", "start_type": "categorial",
                                        "max_sequence_length": MAX_SEQ, "counts": [1, 1, 1]}
    return trainer


def _crash_after(trainer: WyckoffTrainer, completed_epochs: int) -> None:
    """Make `train_epoch` raise once `completed_epochs` of them have finished."""
    original = trainer.train_epoch
    remaining = [completed_epochs]

    def train_epoch():
        if remaining[0] <= 0:
            raise _Crash("the node went away")
        remaining[0] -= 1
        original()

    trainer.train_epoch = train_epoch


def _run(trainer: WyckoffTrainer):
    with patch("wyckoff_transformer.trainer.wandb") as wandb_mock:
        wandb_mock.run.id = "testrunid"
        trainer.train()


def _weights(trainer: WyckoffTrainer):
    return {key: value.clone() for key, value in trainer.model.state_dict().items()}


class _RunDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def run_dir(self, name: str) -> Path:
        path = self.tmp_path / name
        path.mkdir()
        return path


class TestResumeReproducesAnUninterruptedRun(_RunDirTestCase):
    """The property the whole feature rests on, for each optimiser we train with."""

    def _uninterrupted_then_resumed(self, epochs=6, crash_after=3, **kwargs):
        uninterrupted = _make_trainer(self.run_dir("uninterrupted"), epochs=epochs, **kwargs)
        _run(uninterrupted)

        crashed_path = self.run_dir("crashed")
        crashed = _make_trainer(crashed_path, epochs=epochs, **kwargs)
        _crash_after(crashed, crash_after)
        with self.assertRaises(_Crash):
            _run(crashed)
        self.assertTrue((crashed_path / CHECKPOINT_FILENAME).exists())

        resumed = _make_trainer(crashed_path, epochs=epochs, resume=True, **kwargs)
        _run(resumed)
        return uninterrupted, resumed

    def assert_same_weights(self, expected: WyckoffTrainer, actual: WyckoffTrainer):
        expected_weights, actual_weights = _weights(expected), _weights(actual)
        self.assertEqual(sorted(expected_weights), sorted(actual_weights))
        for key, value in expected_weights.items():
            # Exactly, not approximately: the same steps in the same order on the same data.
            # Anything else means some piece of the state was re-derived rather than restored.
            torch.testing.assert_close(actual_weights[key], value, rtol=0, atol=0)

    def test_adamw_with_a_step_indexed_schedule(self):
        """The moments and the schedule's step counter both have to survive the gap."""
        uninterrupted, resumed = self._uninterrupted_then_resumed()
        self.assert_same_weights(uninterrupted, resumed)
        self.assertEqual(resumed.scheduler.last_epoch, uninterrupted.scheduler.last_epoch)
        self.assertEqual(resumed.optimizer.param_groups[0]["lr"],
                         uninterrupted.optimizer.param_groups[0]["lr"])

    def test_schedule_free(self):
        """`z` lives in the optimiser state and never touches `best_model_params.pt`."""
        uninterrupted, resumed = self._uninterrupted_then_resumed(optimiser="schedule_free")
        self.assert_same_weights(uninterrupted, resumed)

    def test_without_a_schedule(self):
        uninterrupted, resumed = self._uninterrupted_then_resumed(scheduled=False)
        self.assert_same_weights(uninterrupted, resumed)

    def test_a_checkpoint_cadence_coarser_than_the_crash(self):
        """Resuming rewinds to the last checkpoint, which is behind where the crash happened."""
        uninterrupted, resumed = self._uninterrupted_then_resumed(
            epochs=8, crash_after=5, checkpoint_period=2, validation_period=2)
        # The last checkpoint was at epoch 4, so epoch 5 is repeated rather than skipped: the
        # same eight epochs of training happen, and the weights land in the same place.
        self.assert_same_weights(uninterrupted, resumed)

    def test_the_best_weights_and_their_epoch_survive(self):
        uninterrupted, resumed = self._uninterrupted_then_resumed()
        best = torch.load(resumed.run_path / "best_model_params.pt", weights_only=True)
        reference = torch.load(
            uninterrupted.run_path / "best_model_params.pt", weights_only=True)
        for key, value in reference.items():
            torch.testing.assert_close(best[key], value, rtol=0, atol=0)


class TestCheckpointContents(_RunDirTestCase):
    def test_round_trip_restores_optimiser_schedule_and_bookkeeping(self):
        run_path = self.run_dir("run")
        trainer = _make_trainer(run_path, epochs=4)
        _run(trainer)
        expected_optimizer = trainer.optimizer.state_dict()

        restored = _make_trainer(run_path, epochs=4, resume=True)
        state = restored.load_training_checkpoint()

        self.assertEqual(state["epoch"], 4)
        self.assertGreaterEqual(state["best_val_epoch"], 0)
        self.assertLess(state["best_val_loss"], float("inf"))
        self.assertEqual(restored.scheduler.state_dict()["last_epoch"],
                         trainer.scheduler.state_dict()["last_epoch"])
        for group_index, group in enumerate(expected_optimizer["param_groups"]):
            self.assertEqual(restored.optimizer.state_dict()["param_groups"][group_index]["lr"],
                             group["lr"])
        for key, value in expected_optimizer["state"].items():
            for name, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    torch.testing.assert_close(
                        restored.optimizer.state_dict()["state"][key][name], tensor,
                        rtol=0, atol=0)

    def test_it_loads_without_executing_pickled_objects(self):
        """`weights_only=True` is the reason the RNG snapshot leaves numpy out."""
        run_path = self.run_dir("run")
        _run(_make_trainer(run_path, epochs=2))
        loaded = torch.load(run_path / CHECKPOINT_FILENAME, weights_only=True)
        self.assertEqual(loaded["format_version"], CHECKPOINT_FORMAT_VERSION)
        self.assertIn("python", loaded["rng"])
        self.assertNotIn("numpy", loaded["rng"])

    def test_a_completed_run_resumes_with_nothing_left_to_train(self):
        """So a crash in the generation that follows training does not cost a training run."""
        run_path = self.run_dir("run")
        _run(_make_trainer(run_path, epochs=3))
        self.assertEqual(
            torch.load(run_path / CHECKPOINT_FILENAME, weights_only=True)["epoch"], 3)

        resumed = _make_trainer(run_path, epochs=3, resume=True)
        _crash_after(resumed, 0)  # any training at all now raises
        _run(resumed)

    def test_early_stopping_records_the_run_as_finished(self):
        run_path = self.run_dir("run")
        trainer = _make_trainer(run_path, epochs=20, scheduled=False)
        trainer.early_stopping_patience_epochs = 0
        _run(trainer)
        self.assertEqual(
            torch.load(run_path / CHECKPOINT_FILENAME, weights_only=True)["epoch"], 20)


class TestCheckpointRefusals(_RunDirTestCase):
    def _write_run(self, epochs=4) -> Path:
        run_path = self.run_dir("run")
        _run(_make_trainer(run_path, epochs=epochs))
        return run_path

    def test_a_different_epoch_count_is_refused(self):
        """It would put the schedule and the patience budget on a horizon neither run has."""
        run_path = self._write_run()
        resumed = _make_trainer(run_path, epochs=5, resume=True)
        with self.assertRaisesRegex(ValueError, "4-epoch run"):
            resumed.load_training_checkpoint()

    def test_a_different_step_budget_is_refused(self):
        run_path = self._write_run()
        resumed = _make_trainer(run_path, epochs=4, resume=True)
        resumed.scheduler_total_steps += 1
        with self.assertRaisesRegex(ValueError, "optimiser steps"):
            resumed.load_training_checkpoint()

    def test_an_unreadable_format_version_is_refused(self):
        run_path = self._write_run()
        path = run_path / CHECKPOINT_FILENAME
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
        torch.save(checkpoint, path)
        resumed = _make_trainer(run_path, epochs=4, resume=True)
        with self.assertRaisesRegex(ValueError, "cannot be resumed"):
            resumed.load_training_checkpoint()

    def test_a_dataset_of_a_different_size_is_refused(self):
        """The shuffle order is indexed by example, so it cannot be reused across a reshape."""
        run_path = self._write_run()
        resumed = _make_trainer(run_path, epochs=4, resume=True)
        resumed.train_loader.num_examples += 1
        with self.assertRaisesRegex(ValueError, "dataset changed under the run"):
            resumed.load_training_checkpoint()


class TestAtomicSave(_RunDirTestCase):
    def test_a_crash_mid_write_leaves_the_previous_checkpoint_intact(self):
        """The failure the whole feature exists for must not take the checkpoint with it."""
        path = self.tmp_path / CHECKPOINT_FILENAME
        atomic_torch_save({"epoch": torch.tensor(1)}, path)

        def die(*args, **kwargs):
            raise _Crash("out of disk")

        with patch("wyckoff_transformer.trainer.torch.save", side_effect=die), \
             self.assertRaises(_Crash):
            atomic_torch_save({"epoch": torch.tensor(2)}, path)
        self.assertEqual(torch.load(path, weights_only=True)["epoch"].item(), 1)


class TestLoaderStateRoundTrip(unittest.TestCase):
    """The shuffle position, which `get_next_batch` reads and the Scalar-target path uses.

    Restoring the RNG alone is not enough there: it would put the loader at the start of a
    *new* permutation rather than partway through the one the run was drawing from.
    """

    @staticmethod
    def _batches(loader, count):
        return [loader.get_next_batch().clone() for _ in range(count)]

    def test_a_restored_loader_continues_the_same_permutation(self):
        torch.manual_seed(SEED)
        dataset = _make_dataset(batch_size=8)
        reference = AugmentedCascadeLoader.from_dataset(dataset)
        self._batches(reference, 2)
        state, rng = reference.state_dict(), torch.get_rng_state()
        # Long enough to run past the end of the shuffle order, where the loader draws a fresh
        # permutation: the restored RNG has to take over exactly there.
        expected = self._batches(reference, 6)

        restored = AugmentedCascadeLoader.from_dataset(dataset)
        self.assertFalse(torch.equal(restored.this_shuffle_order, state["this_shuffle_order"]),
                         "a fresh loader should start on a different order, or this proves nothing")
        restored.load_state_dict(state)
        torch.set_rng_state(rng)
        for expected_batch, actual_batch in zip(expected, self._batches(restored, 6)):
            torch.testing.assert_close(actual_batch, expected_batch, rtol=0, atol=0)

    def test_a_dataset_of_a_different_size_is_refused(self):
        torch.manual_seed(SEED)
        dataset = _make_dataset(batch_size=8)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        state = loader.state_dict()
        loader.num_examples += 1
        with self.assertRaisesRegex(ValueError, "dataset changed under the run"):
            loader.load_state_dict(state)


class TestResumeConfigCheck(_RunDirTestCase):
    BASE = {"dataset": "mp_20",
            "model": {"WyckoffTrainer_args": {"train_batch_size": 100}},
            "optimisation": {"optimiser": {"config": {"lr": 0.1}}, "epochs": 10}}

    def _saved(self, config=None) -> Path:
        path = self.tmp_path / "config.yaml"
        OmegaConf.save(OmegaConf.create(config or self.BASE), path)
        return path

    def test_the_same_config_passes(self):
        check_resume_config(OmegaConf.create(self.BASE), self._saved())

    def test_a_changed_value_is_reported_by_key(self):
        changed = OmegaConf.create(self.BASE)
        changed.optimisation.optimiser.config.lr = 0.2
        with self.assertRaises(ValueError) as caught:
            check_resume_config(changed, self._saved())
        self.assertIn("optimisation.optimiser.config.lr", str(caught.exception))
        self.assertIn("0.1", str(caught.exception))

    def test_an_added_key_is_reported(self):
        changed = OmegaConf.create(self.BASE)
        changed.model.WyckoffTrainer_args.compile_model = True
        with self.assertRaises(ValueError) as caught:
            check_resume_config(changed, self._saved())
        self.assertIn("model.WyckoffTrainer_args.compile_model", str(caught.exception))

    def test_a_missing_saved_config_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            check_resume_config(OmegaConf.create(self.BASE), self.tmp_path / "absent.yaml")


class TestTrainFromConfigResumeBranch(_RunDirTestCase):
    """What `--resume` checks before it loads a dataset or touches the GPU."""

    RUN_ID = "abcd1234"

    def setUp(self):
        super().setUp()
        self.runs = self.tmp_path / "runs"
        self.this_run = self.runs / self.RUN_ID
        self.config = OmegaConf.create({"dataset": "mp_20", "optimisation": {"epochs": 4}})

    def _call(self, resume: bool):
        with patch("wyckoff_transformer.trainer.wandb") as wandb_mock:
            wandb_mock.run.id = self.RUN_ID
            train_from_config(self.config, torch.device("cpu"), run_path=self.runs, resume=resume)

    def test_resuming_a_run_with_no_checkpoint_is_refused(self):
        """The one case a resume cannot rescue, and it must say so rather than start over."""
        self.this_run.mkdir(parents=True)
        OmegaConf.save(self.config, self.this_run / "config.yaml")
        with self.assertRaisesRegex(FileNotFoundError, "no checkpoint"):
            self._call(resume=True)

    def test_resuming_under_a_different_config_is_refused_before_training(self):
        self.this_run.mkdir(parents=True)
        (self.this_run / CHECKPOINT_FILENAME).touch()
        OmegaConf.save(
            OmegaConf.create({"dataset": "mp_20", "optimisation": {"epochs": 8}}),
            self.this_run / "config.yaml")
        with self.assertRaisesRegex(ValueError, "optimisation.epochs"):
            self._call(resume=True)

    def test_without_resume_an_existing_run_directory_is_still_refused(self):
        """Overwriting a previous run's directory stays an error; --resume is the way in."""
        self.this_run.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self._call(resume=False)


if __name__ == "__main__":
    unittest.main()
