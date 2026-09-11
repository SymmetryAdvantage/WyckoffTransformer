"""Tests for the trajectory checks: with the "DFT" and the "MLIP" the same, all must pass.

Run with ``CUDA_VISIBLE_DEVICES=""``.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from wyckoff_transformer.cryspr import gradient_matched as gm
from wyckoff_transformer.cryspr import trajectory_validation as tv

SYMPREC = 1e-5


def _start() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.65, cubic=True) * (2, 1, 1)
    atoms.rattle(0.04, seed=5)
    strain = np.array([[0.02, 0.005, 0.0], [0.005, -0.01, 0.0], [0.0, 0.0, 0.015]])
    atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
    return atoms


@pytest.fixture(scope="module")
def emt_task():
    return tv.synthetic_task(_start(), EMT(), SYMPREC, relax1_steps=6, fmax=1e-4, max_steps=400)


def _frames(task):
    return [tv.frames_of(calc, task["energy_key"]) for calc in task["calcs"]]


def _rotation(theta=0.7):
    axis = np.array([1.0, 2.0, -0.5]); axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def test_task_round_trips_through_frames(emt_task):
    frames = _frames(emt_task)
    assert [len(f) for f in frames] == [len(c["steps"]) for c in emt_task["calcs"]]
    first = frames[0][0]
    probe = _start()
    probe.calc = EMT()
    np.testing.assert_allclose(first.stress, probe.get_stress(voigt=False), atol=1e-12)
    np.testing.assert_allclose(first.forces, probe.get_forces(), atol=1e-12)


def test_relative_coordinates_invert_atoms_at_and_remove_rotation():
    atoms = _start()
    coords = gm.GeneralizedCoordinates(atoms, SYMPREC)
    q = np.random.default_rng(3).normal(scale=0.02, size=coords.dim)
    q[:3 * len(atoms)] -= np.tile(q[:3 * len(atoms)].reshape(-1, 3).mean(0), len(atoms))
    moved = coords.atoms_at(q)
    q_back, rotation = tv.relative_coordinates(coords, moved)
    np.testing.assert_allclose(q_back, q, atol=1e-10)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)

    turn = _rotation()
    rotated = Atoms(moved.symbols, positions=moved.positions @ turn, cell=moved.cell.array @ turn, pbc=True)
    q_rot, rotation = tv.relative_coordinates(coords, rotated)
    np.testing.assert_allclose(q_rot, q, atol=1e-10)
    np.testing.assert_allclose(rotation, turn, atol=1e-10)
    moved.calc, rotated.calc = EMT(), EMT()
    g = coords.gradient(q, moved.get_forces(), moved.get_stress(voigt=False), moved.get_volume())
    g_rot = tv.gradient_in(coords, q_rot, rotation, rotated.get_forces(), rotated.get_stress(voigt=False),
                           rotated.get_volume())
    np.testing.assert_allclose(g_rot, g, atol=1e-9)


def test_t0_t1_pass_when_the_mlip_is_the_dft(emt_task):
    tangent_ratios = []
    for frames in _frames(emt_task):
        for r in tv.analyse_pairs(frames, SYMPREC, gm.calculator_evaluator(EMT()), fd_step=1e-4):
            assert r["dq_outside_subspace"] < 1e-9
            # FrechetCellFilter moves the cell through a general deformation gradient, so
            # float noise in the stress accumulates ~1e-4 deg of genuine rotation over a
            # relaxation; relative_coordinates removes it either way.
            assert r["rotation_deg"] < 1e-2
            assert r["dE_trapezoid_positions"] + r["dE_trapezoid_cell"] == pytest.approx(r["dE_trapezoid"], abs=1e-12)
            assert r["residual_e_fr_energy"] == r["residual"] == r["residual_e_wo_entrp"]
            assert r["dV"] == pytest.approx(r["volume1"] - r["volume0"], abs=1e-12)
            assert 0.0 <= r["iso_fraction"] <= 1.0 or np.isnan(r["iso_fraction"])
            if abs(r["dE"]) > 1e-5:
                assert abs(r["residual"]) <= 0.05 * abs(r["dE"])
            if r["abs_dq"] > 1e-4:
                assert r["kappa_mlip_secant"] == pytest.approx(r["kappa_dft"], rel=1e-8, abs=1e-8)
                assert r["mlip_g_along"] == pytest.approx(r["g_along"], rel=1e-8, abs=1e-10)
                # With the DFT gradient equal to the MLIP's, the match is zero.
                assert r["kappa_matched_tangent"] == pytest.approx(r["kappa_mlip_tangent"], rel=1e-8, abs=1e-8)
                tangent_ratios.append(r["kappa_mlip_tangent"] / r["kappa_dft"])
    assert tangent_ratios and 0.8 < np.median(tangent_ratios) < 1.25


def test_t2_relax_estimate_equals_the_remaining_drop(emt_task):
    frames = _frames(emt_task)[-1]
    evaluator = gm.calculator_evaluator(EMT())
    kwargs = {"fmax": 1e-5, "steps": 3000}
    final, *_ = tv.estimate_at(frames[-1], EMT(), evaluator, SYMPREC, fd_step=1e-4, relax_kwargs=kwargs)
    assert final["relax"] == pytest.approx(0.0, abs=1e-6)
    for k in (0, len(frames) // 2, len(frames) - 2):
        estimate, *_ = tv.estimate_at(frames[k], EMT(), evaluator, SYMPREC, fd_step=1e-4, relax_kwargs=kwargs)
        assert estimate["relax"] == pytest.approx(frames[k].energy - frames[-1].energy + final["relax"], abs=1e-6)
    near, *_ = tv.estimate_at(frames[-2], EMT(), evaluator, SYMPREC, fd_step=1e-4, relax=False)
    drop = frames[-2].energy - frames[-1].energy
    if drop > 1e-6:
        assert near["newton"] == pytest.approx(drop, rel=0.2)
        assert near["sd_bound"] <= near["newton"] * (1 + 1e-6)


def test_t3_boundary_is_continuous_for_a_restart(emt_task):
    boundary, = tv.calc_boundaries(_frames(emt_task), SYMPREC)
    assert boundary["same_species"]
    assert boundary["max_abs_q"] < 1e-10
    assert boundary["dE_per_atom"] == 0.0
    assert boundary["d_hydrostatic_kbar"] == 0.0 and boundary["d_deviatoric_kbar"] == 0.0
    assert boundary["elements"] == ["Cu"]
    assert boundary["dE_e_fr_energy_per_atom"] == 0.0
    frames = _frames(emt_task)[0]
    v_first, v_last = frames[0].atoms.get_volume(), frames[-1].atoms.get_volume()
    assert boundary["calc_volume_change"] == pytest.approx(v_last / v_first - 1, abs=1e-12)
    assert boundary["effective_cutoff_ratio"] == pytest.approx((v_first / v_last) ** (2 / 3), abs=1e-12)


def _quadratic_frames(n_frames=14, seed=0):
    """Fixed-cell frames on E = 1/2 sum_i (u_i - mean u)^T K (u_i - mean u): exactly quadratic in q."""
    rng = np.random.default_rng(seed)
    cell = np.diag([4.1, 4.4, 4.8]) + rng.normal(scale=0.2, size=(3, 3))
    reference = Atoms("Cu4", scaled_positions=rng.random((4, 3)), cell=cell, pbc=True)
    stiffness = np.diag([3.0, 5.0, 8.0]) + 0.5
    frames = []
    for i in range(n_frames):
        u = rng.normal(scale=0.03, size=(4, 3)) * (0.8 ** i)
        rel = u - u.mean(0)
        atoms = reference.copy()
        atoms.positions = reference.positions + u
        frames.append(tv.Frame(atoms, float(0.5 * np.einsum("ia,ab,ib->", rel, stiffness, rel)),
                               -rel @ stiffness.T, np.zeros((3, 3)), i))
    return frames, stiffness


def test_t4_secant_fit_recovers_an_exact_quadratic():
    frames, stiffness = _quadratic_frames()
    final = frames[-1]
    record = tv.dft_secant_hessian(frames, SYMPREC)
    assert record["explored_dim"] == 3 * len(final.atoms) - 3
    assert record["gradient_fraction_explored"] == pytest.approx(1.0, abs=1e-9)
    assert record["delta_fit_explored"] == pytest.approx(final.energy, rel=1e-6)
    assert record["fit_negative_explored"] == 0
    assert record["delta_bfgs_explored"] > 0

    coords = gm.GeneralizedCoordinates(final.atoms, SYMPREC)
    n = len(final.atoms)
    full = np.zeros((coords.dim, coords.dim))
    full[:3 * n, :3 * n] = np.kron(np.eye(n) - 1.0 / n, stiffness)
    exact = coords.basis.T @ full @ coords.basis
    record = tv.dft_secant_hessian(frames, SYMPREC, mlip_hessian=exact)
    assert record["delta_mlip_explored"] == pytest.approx(final.energy, rel=1e-8)
    flipped = tv.dft_secant_hessian(frames, SYMPREC, mlip_hessian=-exact)
    modes = flipped["mlip_negative_modes"]
    assert len(modes) == 3 * n - 3
    assert all(m["overlap_explored"] > 0.99 and m["fit_curvature"] > 0 for m in modes)
