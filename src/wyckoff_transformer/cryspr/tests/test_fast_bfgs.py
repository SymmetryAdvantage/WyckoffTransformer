"""Equivalent BFGS steps and NEP89-only optimizer selection."""

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase._4.optimize.bfgs import BFGSMethod
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS

from wyckoff_transformer.cryspr.fast_bfgs import PositiveDefiniteBFGS
from wyckoff_transformer.cryspr.generator import prerelax
from wyckoff_transformer.cryspr.nep89 import Nep89WithFallback, ScreenedMorse


def _step(optimizer_class, hessian, gradient):
    atoms = Atoms("H3", positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    opt = optimizer_class(atoms, logfile=None)
    pos = atoms.get_positions().ravel()
    opt.state = BFGSMethod(hessian.copy())
    opt.pos0 = pos.copy()  # The Hessian update sees no displacement.
    opt.forces0 = np.zeros_like(pos)
    return opt.prepare_step(pos, gradient)


def test_positive_definite_step_matches_ase():
    rng = np.random.default_rng(89)
    matrix = rng.normal(size=(9, 9))
    hessian = matrix @ matrix.T + np.eye(9)
    gradient = rng.normal(size=9)
    actual_step, actual_lengths = _step(PositiveDefiniteBFGS, hessian, gradient)
    expected_step, expected_lengths = _step(BFGS, hessian, gradient)
    np.testing.assert_allclose(actual_step, expected_step, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual_lengths, expected_lengths, rtol=1e-10, atol=1e-12)


def test_indefinite_step_uses_ase_rule():
    hessian = np.diag([-2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    gradient = np.arange(1.0, 10.0)
    actual_step, actual_lengths = _step(PositiveDefiniteBFGS, hessian, gradient)
    expected_step, expected_lengths = _step(BFGS, hessian, gradient)
    np.testing.assert_allclose(actual_step, expected_step, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_lengths, expected_lengths, rtol=0, atol=1e-12)


def test_optimizer_log_keeps_bfgs_label():
    atoms = Atoms("Ar2", positions=[(0, 0, 0), (4, 0, 0)])
    atoms.calc = LennardJones()
    log = StringIO()
    PositiveDefiniteBFGS(atoms, logfile=log).run(fmax=1e-3, steps=1)
    assert "BFGS:" in log.getvalue()


def test_only_covered_nep89_uses_fast_solver(tmp_path: Path):
    model = tmp_path / "nep.txt"
    model.write_text("nep4_zbl 1 Si\n", encoding="utf-8")
    calculator = Nep89WithFallback(model=model, elements=frozenset({"Si"}))
    silicon = Atoms("Si", cell=np.eye(3) * 5, pbc=True)
    polonium = Atoms("Po", cell=np.eye(3) * 5, pbc=True)

    with patch("wyckoff_transformer.cryspr.generator.stepwise_relax", side_effect=lambda **kw: kw["atoms_in"]) as relax:
        prerelax(silicon, calculator, tmp_path)
        assert relax.call_args.kwargs["optimizer"] is PositiveDefiniteBFGS
        prerelax(polonium, calculator, tmp_path)
        assert relax.call_args.kwargs["optimizer"] is BFGS
        prerelax(silicon, ScreenedMorse(), tmp_path)
        assert relax.call_args.kwargs["optimizer"] is BFGS
