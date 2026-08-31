"""The AdamW/WSD config builds, and the trainer drives the schedule the way it claims to."""
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from wyckoff_transformer.schedules import warmup_stable_decay

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


if __name__ == "__main__":
    unittest.main()
