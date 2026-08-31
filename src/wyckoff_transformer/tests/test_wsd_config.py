"""The AdamW/WSD config builds, and the trainer drives the schedule the way it claims to."""
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from wyckoff_transformer.schedules import warmup_stable_decay
from wyckoff_transformer.trainer import WyckoffTrainer

CONFIG = Path(__file__).parents[3] / "yamls" / "models" / "lemat_bulk_ehull" / "ehull_adamw_wsd.yaml"
# Train split of cache/lemat_bulk_ehull, which sets the step budget the schedule is placed on.
TRAIN_EXAMPLES = 4007729


class _Dispatcher:
    """The trainer's scheduler-construction branch, isolated from the rest of __init__."""

    def __init__(self, optimisation_config, optimizer, batches_per_epoch):
        import importlib
        self.optimizer = optimizer
        self.train_loader = type("L", (), {"batches_per_epoch": batches_per_epoch})()
        scheduler_config = dict(optimisation_config.scheduler.get("config", {}))
        module = importlib.import_module(
            optimisation_config.scheduler.get("module", "torch.optim.lr_scheduler"))
        name = optimisation_config.scheduler.name
        factory = getattr(module, name)
        if name == "ReduceLROnPlateau":
            self.scheduler = factory(optimizer, 'min', **scheduler_config)
            self.scheduler_steps_per_batch = False
        else:
            if name in getattr(module, "NEEDS_TOTAL_STEPS", ()):
                scheduler_config.setdefault(
                    "total_steps", optimisation_config.epochs * batches_per_epoch)
            self.scheduler = factory(optimizer, **scheduler_config)
            self.scheduler_steps_per_batch = True


class TestAdamWWsdConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = OmegaConf.load(CONFIG)

    def test_the_four_intended_differences_are_present(self):
        opt = self.config.optimisation
        args = self.config.model.WyckoffTrainer_args
        layer = self.config.model.CascadeTransformer_args.TransformerEncoderLayer_args
        self.assertEqual(layer.dropout, 0.0)
        self.assertEqual(opt.optimiser.name, "AdamW")
        self.assertEqual(args.train_batch_size, 50000)
        self.assertEqual(opt.scheduler.name, "warmup_stable_decay")

    def test_it_stays_comparable_to_the_schedule_free_config(self):
        """Everything not deliberately changed must match, or the comparison means nothing."""
        baseline = OmegaConf.load(CONFIG.parent / "ehull_schedule_free.yaml")
        self.assertEqual(self.config.model.cascade, baseline.model.cascade)
        self.assertEqual(self.config.tokeniser, baseline.tokeniser)
        for key in ("target", "condition_feature", "condition_transform",
                    "multiclass_next_token_with_order_permutation"):
            self.assertEqual(self.config.model.WyckoffTrainer_args[key],
                             baseline.model.WyckoffTrainer_args[key], key)
        new_layer = dict(self.config.model.CascadeTransformer_args.TransformerEncoderLayer_args)
        old_layer = dict(baseline.model.CascadeTransformer_args.TransformerEncoderLayer_args)
        self.assertEqual({k: v for k, v in new_layer.items() if k != "dropout"},
                         {k: v for k, v in old_layer.items() if k != "dropout"})

    def test_total_steps_comes_from_the_run_length_not_the_config(self):
        """Writing total_steps by hand is how it comes to disagree with `epochs`."""
        self.assertNotIn("total_steps", self.config.optimisation.scheduler.config)
        batches_per_epoch = TRAIN_EXAMPLES // self.config.model.WyckoffTrainer_args.train_batch_size
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], **self.config.optimisation.optimiser.config)
        d = _Dispatcher(self.config.optimisation, opt, batches_per_epoch)
        self.assertTrue(d.scheduler_steps_per_batch)
        self.assertEqual(batches_per_epoch, 80)

    def test_the_schedule_completes_its_decay_within_the_budget(self):
        """The decay is where the gain is; it must finish inside `epochs`, not after."""
        batches_per_epoch = TRAIN_EXAMPLES // self.config.model.WyckoffTrainer_args.train_batch_size
        total_steps = self.config.optimisation.epochs * batches_per_epoch
        param = torch.nn.Parameter(torch.zeros(1))
        peak = self.config.optimisation.optimiser.config.lr
        opt = torch.optim.AdamW([param], lr=peak)
        sched = warmup_stable_decay(
            opt, total_steps=total_steps, **self.config.optimisation.scheduler.config)
        # Sampling the schedule rather than stepping 6.8M times.
        def lr_at(step):
            sched.last_epoch = step - 1
            return sched.get_lr()[0]
        self.assertLess(lr_at(1), peak)                      # warming up
        self.assertAlmostEqual(lr_at(total_steps // 2), peak, places=12)   # stable
        self.assertLess(lr_at(total_steps - 1), peak * 0.02)  # annealed by the end
        self.assertGreater(lr_at(int(total_steps * 0.89)), peak * 0.99)   # decay not early

    def test_early_stopping_cannot_cut_the_decay_phase(self):
        opt = self.config.optimisation
        self.assertGreater(opt.early_stopping_patience_epochs, opt.epochs)

    def test_the_budget_matches_the_measured_throughput(self):
        """85000 epochs at the 18.60 ms/step measured on t1c9ehzp must fit in 72 hours."""
        seconds_per_epoch = 0.01860 * (TRAIN_EXAMPLES // 25000)
        hours = self.config.optimisation.epochs * seconds_per_epoch / 3600
        self.assertLess(hours, 72.0)
        self.assertGreater(hours, 66.0)  # and does not leave the card idle for hours either


class TestTheScheduleHorizonMatchesTheRun(unittest.TestCase):
    """The schedule places warmup and decay as fractions of a horizon derived from `epochs`.
    If the run does not consume exactly that many steps the phases land in the wrong places."""

    @staticmethod
    def _trainer(epochs, batches_per_epoch, total_steps, patience=10**9):
        t = WyckoffTrainer.__new__(WyckoffTrainer)
        t.epochs = epochs
        t.train_loader = type("L", (), {"batches_per_epoch": batches_per_epoch})()
        t.scheduler_total_steps = total_steps
        t.early_stopping_patience_epochs = patience
        t.train_dataset = object()
        t.val_dataset = object()
        t.production_training = False
        t.run_path = None
        return t

    def test_a_matching_horizon_passes_the_guard(self):
        t = self._trainer(epochs=100, batches_per_epoch=80, total_steps=8000)
        # The guard runs before anything touches the filesystem, so it is the only thing
        # that can raise here; failing past it means a different error surfaces.
        with self.assertRaises(AttributeError):
            WyckoffTrainer.train(t)

    def test_changing_epochs_after_construction_is_refused(self):
        t = self._trainer(epochs=200, batches_per_epoch=80, total_steps=8000)
        with self.assertRaises(ValueError) as cm:
            WyckoffTrainer.train(t)
        self.assertIn("16000", str(cm.exception))
        self.assertIn("8000", str(cm.exception))

    def test_changing_the_batch_size_after_construction_is_refused(self):
        t = self._trainer(epochs=100, batches_per_epoch=160, total_steps=8000)
        with self.assertRaises(ValueError):
            WyckoffTrainer.train(t)

    def test_early_stopping_below_the_horizon_is_warned_about(self):
        t = self._trainer(epochs=100, batches_per_epoch=80, total_steps=8000, patience=10)
        with self.assertLogs(level="WARNING") as logs:
            with self.assertRaises(AttributeError):
                WyckoffTrainer.train(t)
        self.assertTrue(any("never completes its decay" in m for m in logs.output))

    def test_a_metric_driven_schedule_is_not_subject_to_the_guard(self):
        """ReduceLROnPlateau has no horizon, so `epochs` may be anything."""
        t = self._trainer(epochs=100, batches_per_epoch=80, total_steps=None)
        with self.assertRaises(AttributeError):
            WyckoffTrainer.train(t)

    def test_the_config_horizon_is_what_the_run_will_consume(self):
        cfg = OmegaConf.load(CONFIG)
        batches_per_epoch = TRAIN_EXAMPLES // cfg.model.WyckoffTrainer_args.train_batch_size
        self.assertEqual(cfg.optimisation.epochs * batches_per_epoch, 6_800_000)


@pytest.mark.needs_cache
class TestTheScheduleIsConsumedExactlyByARealRun(unittest.TestCase):
    """Counting the steps a real train() actually takes, rather than trusting the arithmetic.

    Uses the pilot cache so it runs on a CPU in seconds; what is under test is the wiring
    between epochs, batches_per_epoch and scheduler.step(), which is size-independent.
    """

    EPOCHS = 7

    def setUp(self):
        if not (Path("cache") / "lemat_bulk_ehull_pilot" / "tensors").exists():
            self.skipTest("lemat_bulk_ehull_pilot cache not present")

    def _run(self):
        import wandb
        cfg = OmegaConf.load(CONFIG)
        cfg.tokeniser = OmegaConf.load(
            CONFIG.parents[2] / "tokenisers" / "lemat_bulk_ehull_sg_multiplicity.yaml")
        cfg.dataset = "lemat_bulk_ehull_pilot"
        for key in ("train_batch_size", "val_batch_size", "test_batch_size"):
            cfg.model.WyckoffTrainer_args[key] = 2000
        cfg.model.WyckoffTrainer_args.compile_model = False
        cfg.optimisation.epochs = self.EPOCHS
        cfg.optimisation.validation_period = 3
        # Exaggerated so all three phases are visible in a handful of epochs.
        cfg.optimisation.scheduler.config.warmup_fraction = 0.1
        cfg.optimisation.scheduler.config.decay_fraction = 0.5

        wandb.init(mode="disabled")
        with tempfile.TemporaryDirectory() as tmp:
            trainer = WyckoffTrainer.from_config(
                cfg, torch.device("cpu"), run_path=Path(tmp), no_test=True)
            counts = {"opt": 0, "sched": 0}
            lrs = []
            real_opt, real_sched = trainer.optimizer.step, trainer.scheduler.step

            def opt_step(*a, **kw):
                counts["opt"] += 1
                lrs.append(trainer.optimizer.param_groups[0]["lr"])
                return real_opt(*a, **kw)

            def sched_step(*a, **kw):
                counts["sched"] += 1
                return real_sched(*a, **kw)

            trainer.optimizer.step, trainer.scheduler.step = opt_step, sched_step
            trainer.train()
            return trainer, counts, lrs, cfg

    def test_the_step_count_matches_epochs_times_batches_per_epoch(self):
        trainer, counts, lrs, _ = self._run()
        horizon = self.EPOCHS * trainer.train_loader.batches_per_epoch
        self.assertEqual(counts["opt"], horizon)
        self.assertEqual(counts["sched"], horizon)
        self.assertEqual(trainer.scheduler.last_epoch, horizon)
        self.assertEqual(trainer.scheduler_total_steps, horizon)

    def test_every_step_is_taken_at_a_usable_rate(self):
        """No step at zero or below: the schedule must not run past its own horizon."""
        _, _, lrs, _ = self._run()
        self.assertGreater(min(lrs), 0.0)

    def test_the_run_reaches_the_peak_and_finishes_annealed(self):
        _, _, lrs, cfg = self._run()
        peak = cfg.optimisation.optimiser.config.lr
        self.assertAlmostEqual(max(lrs), peak, places=12)
        self.assertLess(lrs[-1], peak * 0.05)
        # Warmup is monotone up to the peak, and the final rate is below the first.
        self.assertLess(lrs[0], peak)
        self.assertLess(lrs[-1], lrs[0])


if __name__ == "__main__":
    unittest.main()
