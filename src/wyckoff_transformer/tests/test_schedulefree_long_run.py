"""How schedule-free's averaged iterate behaves over runs far longer than we have used before.

`SGDScheduleFree` keeps `x`, the iterate it reports and checkpoints, as a weighted mean of the
raw iterates `z`:  x <- x + ckp1*(z - x),  ckp1 = weight / weight_sum,
weight = (k+1)**r * lr_max**weight_lr_power.  With the default r=0 and a constant lr the weight
is constant, so ckp1 = 1/(k+1) and `x` is the *uniform* mean of every `z` since step 1 -- step 1
and step ten-million carry equal weight.

Two things follow that only appear at high step counts, and both are exercised here on a
stochastic quadratic small enough to run millions of steps on a CPU:

1. Algorithmic. Uniform averaging assumes the iterate settles. If `z` keeps moving, `x` lags it
   by roughly v*k/(r+2) -- unbounded in k -- and the reported loss rises even though the
   optimiser is doing its job. This is a property of the method, not of the implementation, and
   it reproduces in exact arithmetic.
2. Numerical. The update is applied in the parameter's dtype. Once ckp1*|z-x| falls below the
   float32 spacing of |x| it rounds away. That is harmless while (z-x) changes sign, since the
   lost increments cancel, but it becomes a one-way bias as soon as `z` is going somewhere.

Every run in this project before LeMat-Bulk stopped at ~1.5e6 steps, where neither bites.

These are diagnostics for that investigation rather than tests of our own code -- they
characterise the optimiser library -- so the whole module is marked `slow` and is deselected by
default. Run it with `pytest -m slow` when revisiting the averaging behaviour.
"""
import unittest

import pytest
import schedulefree
import torch

pytestmark = pytest.mark.slow

# The geometry that decides whether the averaging update survives float32 is the *relative* gap
# ||z - x|| / ||x||, not the absolute scale: measured at ~0.02 on the real model.
THETA_NORM = 170.0
RELATIVE_GAP = 0.02
# Measured weight-norm growth of the real run, per optimiser step.
DRIFT_PER_STEP = 2.5e-6


def _ckp1(opt) -> float:
    g = opt.param_groups[0]
    weight = ((g["k"] + 1) ** g["r"]) * (g["lr_max"] ** g["weight_lr_power"])
    return weight / (g["weight_sum"] + weight)


class TestAveragingWindow(unittest.TestCase):
    """What ckp1 does as k grows -- the shape of the averaging window we rely on."""

    def _opt_after(self, steps, r=0.0, n=8):
        p = torch.nn.Parameter(torch.ones(n))
        opt = schedulefree.SGDScheduleFree([p], lr=1.0, warmup_steps=0, r=r)
        opt.train()
        p.grad = torch.zeros(n)
        with torch.no_grad():
            for _ in range(steps):
                p.grad.copy_(p)
                opt.step()
        return opt

    def test_default_r_gives_a_uniform_average_over_all_history(self):
        for steps in (10, 100, 1000):
            opt = self._opt_after(steps)
            self.assertAlmostEqual(_ckp1(opt), 1.0 / (steps + 1), places=9,
                                   msg=f"ckp1 is not 1/(k+1) after {steps} steps")

    def test_larger_r_weights_recent_iterates_more(self):
        """weight ~ k**r makes the window ~k/(r+1) instead of k -- shorter, but still unbounded."""
        steps = 2000
        for r in (0.0, 2.0, 8.0):
            opt = self._opt_after(steps, r=r)
            self.assertAlmostEqual(_ckp1(opt) * steps, r + 1.0, delta=0.05,
                                   msg=f"ckp1 is not ~{r + 1:.0f}/k at r={r}")

    def test_the_window_never_stops_growing(self):
        """No setting of r bounds the window, which is what lets x fall behind a moving z."""
        for r in (0.0, 8.0):
            small, large = self._opt_after(500, r=r), self._opt_after(2000, r=r)
            self.assertLess(_ckp1(large), _ckp1(small) / 3,
                            f"window did not keep growing at r={r}")


class TestFloat32TruncatesTheAveragingUpdate(unittest.TestCase):
    """x <- x + ckp1*(z - x) at the dtype of x, with realistic magnitudes."""

    def _realised_fraction(self, ckp1, dtype, n=100_000, seed=0):
        gen = torch.Generator().manual_seed(seed)
        x = torch.full((n,), THETA_NORM / n**0.5, dtype=torch.float64)
        d = torch.randn(n, generator=gen, dtype=torch.float64)
        d *= RELATIVE_GAP * x.norm() / d.norm()
        xd, zd = x.to(dtype), (x + d).to(dtype)
        moved = xd.clone().lerp_(zd, weight=ckp1)
        return float((moved.to(torch.float64) - xd.to(torch.float64)).norm()
                     / (ckp1 * d).norm())

    def test_short_horizons_are_unaffected(self):
        # ckp1 = 1e-5 is about 1e5 steps -- longer than most runs in this project's history.
        self.assertAlmostEqual(self._realised_fraction(1e-5, torch.float32), 1.0, delta=0.05)

    def test_long_horizons_lose_most_of_the_update(self):
        # ckp1 = 1e-7 is about 1e7 steps, which the LeMat-Bulk runs reached.
        self.assertLess(self._realised_fraction(1e-7, torch.float32), 0.75,
                        "float32 is not truncating; the geometry constants may be stale")

    def test_float64_keeps_the_update_at_every_horizon(self):
        for ckp1 in (1e-5, 1e-7, 1e-9):
            self.assertAlmostEqual(self._realised_fraction(ckp1, torch.float64), 1.0, delta=1e-6,
                                   msg=f"float64 lost the update at ckp1={ckp1:g}")


def _run_quadratic(dtype, steps, n=256, lr=0.1, sigma=1.0, seed=0, r=0.0, drift=0.0):
    """Minimise 0.5*||p - theta*||^2 with noisy gradients; theta* may drift.

    Returns (final loss of the averaged iterate x, final ||x - z||).
    """
    gen = torch.Generator().manual_seed(seed)
    base = torch.full((n,), THETA_NORM / n**0.5, dtype=torch.float64)
    direction = base / base.norm()
    p = torch.nn.Parameter(base.clone().to(dtype))
    opt = schedulefree.SGDScheduleFree([p], lr=lr, warmup_steps=0, r=r)
    opt.train()
    p.grad = torch.zeros(n, dtype=dtype)
    star = base
    with torch.no_grad():
        for k in range(1, steps + 1):
            if drift:
                star = base + direction * (drift * k)
            noise = torch.randn(n, generator=gen, dtype=torch.float64) * sigma
            p.grad.copy_(((p.detach().to(torch.float64) - star) + noise).to(dtype))
            opt.step()
        opt.eval()
        x = p.detach().to(torch.float64)
        z = opt.state[p]["z"].to(torch.float64)
    return 0.5 * float(((x - star) ** 2).sum()), float((x - z).norm())


class TestLongRunDynamics(unittest.TestCase):
    """Millions of real optimiser steps, which is where both effects live."""

    STEPS = 2_000_000

    def test_a_settled_iterate_is_averaged_correctly_in_float32(self):
        """With no drift, (z-x) changes sign and the lost increments cancel."""
        loss64, _ = _run_quadratic(torch.float64, self.STEPS)
        loss32, _ = _run_quadratic(torch.float32, self.STEPS)
        self.assertLess(loss64, 1e-3, "float64 did not converge; the setup is wrong")
        self.assertAlmostEqual(loss32 / loss64, 1.0, delta=0.15,
                               msg="float32 diverged from float64 without any drift")

    def test_a_drifting_iterate_makes_the_average_go_stale(self):
        """The algorithmic failure: exact arithmetic, and the reported loss still climbs."""
        early, _ = _run_quadratic(torch.float64, self.STEPS // 10, drift=DRIFT_PER_STEP)
        late, _ = _run_quadratic(torch.float64, self.STEPS, drift=DRIFT_PER_STEP)
        self.assertGreater(late, 5 * early,
                           "a uniform average of a drifting iterate should fall further behind")

    def test_float32_compounds_the_staleness(self):
        """The numerical failure: once z has a direction, the truncation stops cancelling."""
        loss64, _ = _run_quadratic(torch.float64, self.STEPS, drift=DRIFT_PER_STEP)
        loss32, _ = _run_quadratic(torch.float32, self.STEPS, drift=DRIFT_PER_STEP)
        self.assertGreater(loss32 / loss64, 1.1,
                           "float32 did not add to the drift-induced staleness")

    def test_larger_r_reduces_the_staleness(self):
        """The lag goes as v*k/(r+2), so a shorter window trails a drifting optimum less."""
        loss_r0, _ = _run_quadratic(torch.float64, self.STEPS, drift=DRIFT_PER_STEP, r=0.0)
        loss_r8, _ = _run_quadratic(torch.float64, self.STEPS, drift=DRIFT_PER_STEP, r=8.0)
        self.assertLess(loss_r8, loss_r0 / 2,
                        "raising r did not shorten the effective averaging window")


if __name__ == "__main__":
    unittest.main()
