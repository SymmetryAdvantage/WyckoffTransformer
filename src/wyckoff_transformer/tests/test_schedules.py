"""The step-indexed schedules, and the trainer's dispatch between them and ReduceLROnPlateau."""
import unittest

import torch

from wyckoff_transformer.schedules import NEEDS_TOTAL_STEPS, warmup_stable_decay


def _lrs(total_steps, peak=1.0, **kwargs):
    """The learning rate actually in force at each of `total_steps` steps."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=peak)
    sched = warmup_stable_decay(opt, total_steps=total_steps, **kwargs)
    seen = []
    for _ in range(total_steps):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return seen


class TestWarmupStableDecayShape(unittest.TestCase):
    def test_the_three_phases_have_the_configured_lengths(self):
        lrs = _lrs(1000, warmup_fraction=0.1, decay_fraction=0.2)
        self.assertEqual(len(lrs), 1000)
        # Warmup: strictly increasing over the first 100 steps.
        self.assertTrue(all(b > a for a, b in zip(lrs[:100], lrs[1:100])))
        # Stable: exactly at the peak from 100 to 800.
        self.assertTrue(all(lr == 1.0 for lr in lrs[100:800]))
        # Decay: strictly decreasing over the last 200.
        self.assertTrue(all(b < a for a, b in zip(lrs[800:], lrs[801:])))

    def test_the_first_step_is_not_wasted_at_zero(self):
        lrs = _lrs(1000, warmup_fraction=0.1)
        self.assertGreater(lrs[0], 0.0)

    def test_it_reaches_the_peak_and_ends_near_zero(self):
        lrs = _lrs(1000, warmup_fraction=0.1, decay_fraction=0.2)
        self.assertAlmostEqual(max(lrs), 1.0, places=12)
        self.assertLess(lrs[-1], 0.01)

    def test_the_floor_is_respected(self):
        lrs = _lrs(1000, warmup_fraction=0.1, decay_fraction=0.2, final_lr_fraction=0.1)
        self.assertGreaterEqual(min(lrs[100:]), 0.1 - 1e-12)

    def test_no_warmup_starts_at_the_peak(self):
        lrs = _lrs(100, warmup_fraction=0.0, decay_fraction=0.0)
        self.assertTrue(all(lr == 1.0 for lr in lrs))

    def test_overrunning_the_horizon_holds_the_floor_rather_than_going_negative(self):
        """A run allowed past total_steps must not reverse its updates."""
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = warmup_stable_decay(opt, total_steps=100, decay_fraction=0.2)
        for _ in range(300):
            opt.step()
            sched.step()
            self.assertGreaterEqual(opt.param_groups[0]["lr"], 0.0)

    def test_decay_shapes_differ_but_all_land_at_zero(self):
        curves = {s: _lrs(1000, warmup_fraction=0.0, decay_fraction=0.5, decay_shape=s)
                  for s in ("linear", "1-sqrt", "cosine")}
        for shape, lrs in curves.items():
            self.assertLess(lrs[-1], 0.01, shape)
        # 1-sqrt drops below linear immediately and stays there; cosine starts above it.
        # (At the exact decay midpoint cosine and linear cross, both at 0.5, so compare early.)
        quarter = 625
        self.assertLess(curves["1-sqrt"][quarter], curves["linear"][quarter])
        self.assertGreater(curves["cosine"][quarter], curves["linear"][quarter])

    def test_bad_configurations_are_rejected(self):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        for kwargs in ({"warmup_fraction": 0.7, "decay_fraction": 0.7},
                       {"warmup_fraction": -0.1},
                       {"decay_fraction": 1.0},
                       {"final_lr_fraction": 1.5},
                       {"decay_shape": "exponential"}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    warmup_stable_decay(opt, total_steps=100, **kwargs)
        with self.assertRaises(ValueError):
            warmup_stable_decay(opt, total_steps=0)

    def test_the_registry_names_the_schedule_that_needs_a_horizon(self):
        self.assertIn("warmup_stable_decay", NEEDS_TOTAL_STEPS)


if __name__ == "__main__":
    unittest.main()
