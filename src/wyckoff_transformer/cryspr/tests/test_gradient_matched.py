"""Tests for the gradient-matched relaxation-energy estimate.

Run with ``CUDA_VISIBLE_DEVICES=""``: nothing here needs a GPU, and a busy GPU 0 is
enough to OOM pytest otherwise.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from wyckoff_transformer.cryspr.gradient_matched import (
    GeneralizedCoordinates,
    GradientMatch,
    GradientMatchedCalculator,
    calculator_evaluator,
    corrected_evaluator,
    curvature_along,
    lemat_stress_to_ase,
    newton_estimate,
    relax_estimate,
    steepest_descent_bound,
)

SYMPREC = 1e-5


def _rattled_cu() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.65, cubic=True) * (2, 1, 1)
    atoms.rattle(0.03, seed=3)
    return atoms


def _targets(n_atoms: int, force_scale: float, stress_scale: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    forces = rng.normal(scale=force_scale, size=(n_atoms, 3))
    forces -= forces.mean(axis=0)
    stress = rng.normal(scale=stress_scale, size=(3, 3))
    return forces, 0.5 * (stress + stress.T)


class HarmonicToy(Calculator):
    """E = k/2 sum |u_i - mean u|^2 + K V0/2 |E_GL|^2, a known quadratic surface about x0."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, reference: Atoms, k: float, bulk_modulus: float):
        super().__init__()
        zeros = np.zeros((len(reference), 3))
        self.frame = GradientMatch(reference, zeros, np.zeros((3, 3)), zeros, np.zeros((3, 3)))
        self.k = k
        self.bulk_modulus = bulk_modulus

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = self.atoms.cell.array
        u = self.frame.displacement(self.atoms.positions, cell)
        u_rel = u - u.mean(axis=0)
        strain = self.frame.green_lagrange(cell)
        deformation = self.frame.inv_cell0 @ cell
        v0, v = self.frame.volume0, abs(np.linalg.det(cell))
        energy = 0.5 * self.k * np.sum(u_rel ** 2) + 0.5 * self.bulk_modulus * v0 * np.sum(strain ** 2)
        forces = -self.k * u_rel @ np.linalg.inv(deformation).T
        stress = (self.bulk_modulus * v0 / v) * deformation.T @ strain @ deformation
        self.results = {"energy": energy, "free_energy": energy, "forces": forces,
                        "stress": full_3x3_to_voigt_6_stress(stress)}


def _p1_reference() -> Atoms:
    rng = np.random.default_rng(11)
    cell = np.diag([4.1, 4.4, 4.8]) + rng.normal(scale=0.2, size=(3, 3))
    return Atoms("Cu4", scaled_positions=rng.random((4, 3)), cell=cell, pbc=True)


def test_lemat_stress_is_vasp_kbar():
    np.testing.assert_allclose(lemat_stress_to_ase(np.eye(3) * 1602.1766208), -np.eye(3))


def test_gradient_equals_targets_at_reference():
    atoms = _rattled_cu()
    forces, stress = _targets(len(atoms), 0.1, 2e-3)
    calc = GradientMatchedCalculator.from_reference(EMT(), atoms, forces, stress)
    probe = atoms.copy()
    probe.calc = calc
    np.testing.assert_allclose(probe.get_forces(), forces, atol=1e-12)
    np.testing.assert_allclose(probe.get_stress(voigt=False), stress, atol=1e-12)


@pytest.mark.parametrize("strained", [False, True])
def test_forces_and_stress_are_derivatives_of_energy(strained):
    atoms = _rattled_cu()
    forces, stress = _targets(len(atoms), 0.2, 5e-3, seed=1)
    calc = GradientMatchedCalculator.from_reference(EMT(), atoms, forces, stress)
    probe = atoms.copy()
    probe.positions += np.random.default_rng(2).normal(scale=0.05, size=probe.positions.shape)
    if strained:
        deformation = np.eye(3) + np.array([[0.02, 0.01, -0.004], [0.003, -0.015, 0.006], [0.0, 0.008, 0.01]])
        probe.set_cell(probe.cell.array @ deformation, scale_atoms=True)
    probe.calc = calc
    np.testing.assert_allclose(probe.get_forces(), calc.calculate_numerical_forces(probe, d=1e-5), atol=1e-7)
    np.testing.assert_allclose(probe.get_stress(), calc.calculate_numerical_stress(probe, d=1e-6), atol=1e-8)


def test_translation_and_rotation_invariance_of_the_correction():
    atoms = _rattled_cu()
    forces, stress = _targets(len(atoms), 0.2, 5e-3, seed=4)
    match = GradientMatch(atoms, np.zeros_like(forces), np.zeros((3, 3)), forces, stress)
    moved = atoms.positions + 0.05 * np.random.default_rng(5).normal(size=atoms.positions.shape)
    cell = atoms.cell.array @ (np.eye(3) + 0.01 * np.random.default_rng(6).normal(size=(3, 3)))
    e_ref = match.terms(moved, cell)[0]
    e_translated = match.terms(moved + np.array([0.3, -0.2, 0.1]), cell)[0]
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    e_rotated = match.terms(moved @ rotation.T, cell @ rotation.T)[0]
    assert e_translated == pytest.approx(e_ref, abs=1e-12)
    assert e_rotated == pytest.approx(e_ref, abs=1e-12)


def test_zero_dft_gradient_gives_zero_drop():
    """A row with zero forces and stress is stationary on E_corr, whatever EMT says."""
    atoms = _rattled_cu()
    zeros = np.zeros((len(atoms), 3))
    calc = GradientMatchedCalculator.from_reference(EMT(), atoms, zeros, np.zeros((3, 3)))
    relaxed = relax_estimate(atoms, calc, symprec=SYMPREC, fmax=1e-4)
    assert relaxed.drop == pytest.approx(0.0, abs=1e-10)
    coords = GeneralizedCoordinates(atoms, symprec=SYMPREC)
    evaluator = corrected_evaluator(calculator_evaluator(EMT()), calc.match)
    newton = newton_estimate(coords, evaluator, step=1e-4)
    assert newton.gradient_norm == pytest.approx(0.0, abs=1e-10)
    assert newton.drop == pytest.approx(0.0, abs=1e-12)
    assert steepest_descent_bound(newton.gradient_norm, 1.0) == pytest.approx(0.0, abs=1e-20)


def test_targets_equal_to_base_reduce_to_plain_relaxation():
    atoms = _rattled_cu()
    base = EMT()
    probe = atoms.copy()
    probe.calc = base
    calc = GradientMatchedCalculator.from_reference(base, atoms, probe.get_forces(),
                                                    probe.get_stress(voigt=False))
    np.testing.assert_allclose(calc.match.delta_forces, 0.0, atol=1e-12)
    np.testing.assert_allclose(calc.match.delta_stress, 0.0, atol=1e-12)
    corrected = relax_estimate(atoms, calc, symprec=SYMPREC, fmax=1e-4)

    plain = atoms.copy()
    plain.calc = EMT()
    plain.set_constraint(FixSymmetry(plain, symprec=SYMPREC))
    e0 = plain.get_potential_energy()
    BFGS(FrechetCellFilter(plain), maxstep=0.05, logfile=None).run(fmax=1e-4, steps=500)
    assert corrected.drop == pytest.approx(e0 - plain.get_potential_energy(), abs=1e-6)
    assert corrected.drop > 1e-3


def test_harmonic_toy_has_the_analytic_drop():
    reference = _p1_reference()
    k, bulk_modulus = 5.0, 1.0
    forces, stress = _targets(len(reference), 0.05, 1e-3, seed=7)
    toy = HarmonicToy(reference, k, bulk_modulus)
    calc = GradientMatchedCalculator.from_reference(toy, reference, forces, stress)
    v0 = reference.get_volume()
    expected_pos = np.sum(forces ** 2) / (2 * k)
    expected_cell = v0 * np.sum(stress ** 2) / (2 * bulk_modulus)

    relaxed = relax_estimate(reference, calc, symprec=SYMPREC, fmax=1e-6, steps=2000)
    assert relaxed.converged
    assert relaxed.drop == pytest.approx(expected_pos + expected_cell, rel=1e-4)

    coords = GeneralizedCoordinates(reference, symprec=SYMPREC)
    assert coords.n_operations == 1
    assert coords.n_position_modes == 3 * len(reference) - 3
    evaluator = corrected_evaluator(calculator_evaluator(toy), calc.match)
    newton = newton_estimate(coords, evaluator, step=1e-4, soft_curvature=1e-6)
    assert newton.drop_positions == pytest.approx(expected_pos, rel=1e-6)
    # E_GL = eps + eps^2/2 in these coordinates, so the cell block is exact to O(sigma/K).
    assert newton.drop_cell == pytest.approx(expected_cell, rel=1e-2)
    assert newton.drop == pytest.approx(expected_pos + expected_cell, rel=1e-2)

    direction = coords.basis @ newton.reduced_gradient
    curvature = curvature_along(coords, evaluator, direction, step=1e-4)
    bound = steepest_descent_bound(newton.gradient_norm, curvature)
    assert 0 < bound <= newton.drop * (1 + 1e-6)
    one_sided = curvature_along(coords, evaluator, direction, step=1e-5, one_sided=True,
                                gradient_at_reference=coords.gradients([np.zeros(coords.dim)], evaluator)[0])
    assert one_sided == pytest.approx(curvature, rel=1e-3)


def test_negative_curvature_aborts_instead_of_running_away():
    reference = _p1_reference()
    forces, stress = _targets(len(reference), 0.05, 1e-4, seed=8)
    toy = HarmonicToy(reference, k=-1.0, bulk_modulus=1.0)
    calc = GradientMatchedCalculator.from_reference(toy, reference, forces, stress)
    relaxed = relax_estimate(reference, calc, symprec=SYMPREC, fmax=1e-4, steps=2000)
    assert relaxed.aborted
    assert not relaxed.converged
    assert np.isnan(relaxed.drop)
    coords = GeneralizedCoordinates(reference, symprec=SYMPREC)
    newton = newton_estimate(coords, corrected_evaluator(calculator_evaluator(toy), calc.match), step=1e-4)
    assert newton.n_negative == coords.n_position_modes
    assert newton.soft_share > 0.99


def test_symmetric_subspace_of_cubic_crystals():
    rocksalt = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    coords = GeneralizedCoordinates(rocksalt, symprec=1e-3)
    assert coords.n_position_modes == 0
    assert coords.basis_cell.shape[1] == 1  # isotropic strain only
    wurtzite = bulk("ZnO", "wurtzite", a=3.25, c=5.2, u=0.38)
    coords = GeneralizedCoordinates(wurtzite, symprec=1e-3)
    assert coords.n_position_modes == 1  # the internal u parameter
    assert coords.basis_cell.shape[1] == 2  # a and c
