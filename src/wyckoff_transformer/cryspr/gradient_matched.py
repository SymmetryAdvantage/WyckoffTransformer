"""How much energy a DFT row would release if its relaxation had converged fully.

A DFT-relaxed structure is not at an MLIP's minimum, so relaxing it with the MLIP
measures the MLIP-vs-DFT disagreement as much as the residual forces. This module
removes the first-order part of that disagreement: the MLIP energy surface is shifted
by a term linear in the displacement and the strain, chosen per atom and per
component so that at the reference geometry x0 its forces and stress equal DFT's
exactly::

    E_corr(x) = E_MLIP(x) - dF . u(x) + V0 * dsigma : E_GL(x)
    dF     = F_DFT(x0) - F_MLIP(x0)          (N x 3, net force removed)
    dsigma = sigma_DFT(x0) - sigma_MLIP(x0)  (3 x 3, symmetric, ASE sign)

``u`` is each atom's displacement beyond the homogeneous deformation of the cell,
measured in the reference frame, and ``E_GL`` is the Green-Lagrange strain of the
cell against x0 -- both are invariant under rigid rotation, and the linear term is
invariant under rigid translation because ``dF`` sums to zero. What is left of the
MLIP is its curvature, so the estimate goes to zero as the DFT gradient does,
whatever the MLIP's own forces at x0 are; the remaining error is in curvature and is
multiplicative rather than additive.

Three estimates of the energy released come out, all in eV for the whole cell:

``relax``
    Relax on ``E_corr`` with the space group fixed (as VASP's ISYM relaxations are)
    and take ``E_corr(x0) - E_corr(x*)``. Picks up anharmonicity, in the MLIP's
    version of it.
``newton``
    ``1/2 g^T H^-1 g`` in the symmetry-allowed, translation-free subspace, with the
    Hessian by central finite differences of the corrected generalized gradient.
    Eigenvalues below ``soft_curvature`` are clamped to it and counted, because
    that is where the MLIP's relative curvature error is largest.
``steepest_descent``
    ``1/2 |g|^4 / (g^T H g)`` -- a lower bound on the harmonic drop by
    Cauchy-Schwarz, which needs the curvature along one direction only. A DFT single
    point displaced along ``g`` would supply it without any MLIP; the curvature source
    is therefore a parameter (:func:`curvature_along`).

Generalized coordinates are ``q = (u, e * L)``: ``u`` in angstrom and the strain ``e``
in a Frobenius-orthonormal basis of symmetric tensors, scaled by ``L = V0**(1/3)`` so
that both halves are lengths. At ``q``, ``cell = cell0 (I + eps)`` and
``positions = (positions0 + u)(I + eps)`` (ASE row-vector convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import prep_symmetry, symmetrize_rank1, symmetrize_rank2
from ase.stress import full_3x3_to_voigt_6_stress

#: LeMat-Bulk ``stress_tensor`` is VASP's: kBar, positive when the cell wants to
#: expand -- the opposite sign to ASE. Established against conservative ORB on 359
#: structures: slope -0.98 on both diagonal and off-diagonal components.
KBAR_PER_EV_PER_A3 = 1602.1766208

#: ``(energy, forces (N, 3), stress (3, 3))`` for each structure, ASE units and sign.
Evaluation = tuple[float, np.ndarray, np.ndarray]
BatchEvaluator = Callable[[Sequence[Atoms]], list[Evaluation]]


def lemat_stress_to_ase(stress_kbar) -> np.ndarray:
    """LeMat-Bulk / VASP stress in kBar to an ASE-sign 3x3 stress in eV/A^3."""
    return -np.asarray(stress_kbar, dtype=float).reshape(3, 3) / KBAR_PER_EV_PER_A3


def calculator_evaluator(calculator: Calculator) -> BatchEvaluator:
    """Evaluate structures one by one with any ASE calculator."""
    def evaluate(structures: Sequence[Atoms]) -> list[Evaluation]:
        results = []
        for structure in structures:
            atoms = structure.copy()
            atoms.set_constraint()
            atoms.calc = calculator
            results.append((float(atoms.get_potential_energy()),
                            atoms.get_forces().copy(),
                            atoms.get_stress(voigt=False).copy()))
        return results
    return evaluate


class GradientMatch:
    """The linear correction that pins an energy surface's gradient to targets at x0."""

    def __init__(self, reference: Atoms, base_forces, base_stress, target_forces, target_stress):
        self.cell0 = reference.cell.array.copy()
        self.inv_cell0 = np.linalg.inv(self.cell0)
        self.scaled0 = reference.positions @ self.inv_cell0
        self.volume0 = abs(np.linalg.det(self.cell0))
        delta_forces = np.asarray(target_forces, float) - np.asarray(base_forces, float)
        self.delta_forces = delta_forces - delta_forces.mean(axis=0)
        delta_stress = (np.asarray(target_stress, float).reshape(3, 3)
                        - np.asarray(base_stress, float).reshape(3, 3))
        self.delta_stress = 0.5 * (delta_stress + delta_stress.T)

    def displacement(self, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Per-atom displacement beyond the cell's deformation, in the reference frame."""
        scaled = positions @ np.linalg.inv(cell) - self.scaled0
        scaled -= np.round(scaled)
        return scaled @ self.cell0

    def green_lagrange(self, cell: np.ndarray) -> np.ndarray:
        deformation = self.inv_cell0 @ cell
        return 0.5 * (deformation @ deformation.T - np.eye(3))

    def terms(self, positions: np.ndarray, cell: np.ndarray):
        """Energy, force and stress added to the base surface at ``(positions, cell)``."""
        deformation = self.inv_cell0 @ cell
        strain = 0.5 * (deformation @ deformation.T - np.eye(3))
        volume = abs(np.linalg.det(cell))
        u = self.displacement(positions, cell)
        energy = -np.sum(self.delta_forces * u) + self.volume0 * np.sum(self.delta_stress * strain)
        forces = self.delta_forces @ np.linalg.inv(deformation).T
        stress = (self.volume0 / volume) * deformation.T @ self.delta_stress @ deformation
        return energy, forces, stress


class GradientMatchedCalculator(Calculator):
    """A base calculator plus a :class:`GradientMatch`."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, base: Calculator, match: GradientMatch, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.match = match

    @classmethod
    def from_reference(cls, base: Calculator, reference: Atoms, target_forces, target_stress):
        (_, forces, stress), = calculator_evaluator(base)([reference])
        return cls(base, GradientMatch(reference, forces, stress, target_forces, target_stress))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        (energy, forces, stress), = calculator_evaluator(self.base)([self.atoms])
        d_energy, d_forces, d_stress = self.match.terms(self.atoms.positions, self.atoms.cell.array)
        self.results = {
            "energy": energy + d_energy,
            "free_energy": energy + d_energy,
            "forces": forces + d_forces,
            "stress": full_3x3_to_voigt_6_stress(stress + d_stress),
        }


def corrected_evaluator(base: BatchEvaluator, match: GradientMatch) -> BatchEvaluator:
    """A batch evaluator for ``E_corr``, e.g. around a batched MLIP."""
    def evaluate(structures: Sequence[Atoms]) -> list[Evaluation]:
        out = []
        for structure, (energy, forces, stress) in zip(structures, base(structures)):
            d_energy, d_forces, d_stress = match.terms(structure.positions, structure.cell.array)
            out.append((energy + d_energy, forces + d_forces, stress + d_stress))
        return out
    return evaluate


def _strain_basis() -> np.ndarray:
    """Six symmetric 3x3 tensors, orthonormal under the Frobenius product."""
    basis = []
    for a in range(3):
        t = np.zeros((3, 3)); t[a, a] = 1.0; basis.append(t)
    for a, b in ((1, 2), (0, 2), (0, 1)):
        t = np.zeros((3, 3)); t[a, b] = t[b, a] = 1 / np.sqrt(2); basis.append(t)
    return np.stack(basis)


class GeneralizedCoordinates:
    """Positions-and-strain coordinates about a reference, and its symmetric subspace."""

    def __init__(self, reference: Atoms, symprec: float):
        self.reference = reference.copy()
        self.reference.set_constraint()
        self.n_atoms = len(reference)
        self.cell0 = reference.cell.array.copy()
        self.positions0 = reference.positions.copy()
        self.length = abs(np.linalg.det(self.cell0)) ** (1 / 3)
        self.strain_basis = _strain_basis()
        self.dim = 3 * self.n_atoms + 6

        rotations, translations, symm_map = prep_symmetry(self.reference, symprec=symprec)
        self.n_operations = len(rotations)
        inv_cell0 = np.linalg.inv(self.cell0)
        n3 = 3 * self.n_atoms
        p_pos = np.empty((n3, n3))
        for j in range(n3):
            e = np.zeros(n3); e[j] = 1.0
            p_pos[:, j] = symmetrize_rank1(self.cell0, inv_cell0, e.reshape(-1, 3),
                                           rotations, translations, symm_map).ravel()
        translation = np.zeros((3, n3))
        for a in range(3):
            translation[a, a::3] = 1 / np.sqrt(self.n_atoms)
        p_pos = p_pos @ (np.eye(n3) - translation.T @ translation)
        p_cell = np.empty((6, 6))
        for k, tensor in enumerate(self.strain_basis):
            sym = symmetrize_rank2(self.cell0, inv_cell0, tensor, rotations)
            p_cell[:, k] = np.einsum("kab,ab->k", self.strain_basis, sym)
        self.basis_positions = self._range(p_pos)
        self.basis_cell = self._range(p_cell)
        basis = np.zeros((self.dim, self.basis_positions.shape[1] + self.basis_cell.shape[1]))
        basis[:n3, :self.basis_positions.shape[1]] = self.basis_positions
        basis[n3:, self.basis_positions.shape[1]:] = self.basis_cell
        #: (dim, m) orthonormal columns spanning the symmetry-allowed, translation-free
        #: displacements; the first ``n_position_modes`` are atomic, the rest cell.
        self.basis = basis
        self.n_position_modes = self.basis_positions.shape[1]

    @staticmethod
    def _range(projector: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(0.5 * (projector + projector.T))
        return vectors[:, values > 0.5]

    def strain(self, q: np.ndarray) -> np.ndarray:
        return np.einsum("k,kab->ab", q[3 * self.n_atoms:] / self.length, self.strain_basis)

    def atoms_at(self, q: np.ndarray) -> Atoms:
        deformation = np.eye(3) + self.strain(q)
        atoms = self.reference.copy()
        atoms.set_cell(self.cell0 @ deformation, scale_atoms=False)
        atoms.positions = (self.positions0 + q[:3 * self.n_atoms].reshape(-1, 3)) @ deformation
        return atoms

    def gradient(self, q: np.ndarray, forces: np.ndarray, stress: np.ndarray, volume: float) -> np.ndarray:
        """dE/dq from ASE forces and stress evaluated at ``atoms_at(q)``."""
        deformation = np.eye(3) + self.strain(q)
        g_positions = -forces @ deformation.T
        m = volume * np.linalg.inv(deformation) @ stress
        g_cell = np.einsum("ab,kab->k", m, self.strain_basis) / self.length
        return np.concatenate([g_positions.ravel(), g_cell])

    def gradients(self, qs: Sequence[np.ndarray], evaluator: BatchEvaluator) -> np.ndarray:
        structures = [self.atoms_at(q) for q in qs]
        return np.stack([
            self.gradient(q, forces, stress, structure.get_volume())
            for q, structure, (_, forces, stress) in zip(qs, structures, evaluator(structures))
        ])


def _clamped_newton(hessian: np.ndarray, gradient: np.ndarray, soft_curvature: float):
    values, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    c2 = (vectors.T @ gradient) ** 2
    clamped = np.maximum(values, soft_curvature)
    contributions = 0.5 * c2 / clamped
    soft = values < soft_curvature
    total = float(contributions.sum())
    return total, values, float(contributions[soft].sum()), int(soft.sum()), int((values < 0).sum())


@dataclass
class NewtonEstimate:
    drop: float                 #: eV, positions and cell jointly
    drop_positions: float       #: eV, position block alone
    drop_cell: float            #: eV, cell block alone
    soft_share: float           #: fraction of ``drop`` from clamped modes
    n_soft: int
    n_negative: int
    min_curvature: float        #: eV/A^2
    n_modes: int
    gradient_norm: float        #: |g| in the symmetric subspace, eV/A
    hessian: np.ndarray
    reduced_gradient: np.ndarray


def newton_estimate(coords: GeneralizedCoordinates, evaluator: BatchEvaluator,
                    step: float = 1e-3, soft_curvature: float = 0.1) -> NewtonEstimate:
    """Harmonic drop ``1/2 g^T H^-1 g`` with a finite-difference Hessian of ``evaluator``."""
    basis = coords.basis
    m = basis.shape[1]
    q0 = np.zeros(coords.dim)
    qs = [q0] + [s * step * basis[:, k] for k in range(m) for s in (1.0, -1.0)]
    grads = coords.gradients(qs, evaluator)
    g = basis.T @ grads[0]
    if m == 0:
        return NewtonEstimate(0.0, 0.0, 0.0, 0.0, 0, 0, np.nan, 0, 0.0, np.zeros((0, 0)), g)
    plus, minus = grads[1::2], grads[2::2]
    hessian = basis.T @ ((plus - minus).T / (2 * step))
    hessian = 0.5 * (hessian + hessian.T)
    drop, values, soft_drop, n_soft, n_negative = _clamped_newton(hessian, g, soft_curvature)
    p = coords.n_position_modes
    drop_pos = _clamped_newton(hessian[:p, :p], g[:p], soft_curvature)[0] if p else 0.0
    drop_cell = _clamped_newton(hessian[p:, p:], g[p:], soft_curvature)[0] if m > p else 0.0
    return NewtonEstimate(drop, drop_pos, drop_cell, soft_drop / drop if drop > 0 else 0.0,
                          n_soft, n_negative, float(values.min()), m, float(np.linalg.norm(g)),
                          hessian, g)


def curvature_along(coords: GeneralizedCoordinates, evaluator: BatchEvaluator, direction: np.ndarray,
                    step: float = 1e-3, one_sided: bool = False,
                    gradient_at_reference: np.ndarray | None = None) -> float:
    """``v^T H v`` for a unit ``v`` along ``direction`` (full generalized coordinates).

    ``evaluator`` is the curvature source. With ``one_sided=True`` and the reference
    gradient supplied, it costs a single evaluation -- the shape of a DFT calibration,
    where the reference gradient is the archived forces and stress.
    """
    v = direction / np.linalg.norm(direction)
    if one_sided:
        if gradient_at_reference is None:
            raise ValueError("one_sided needs gradient_at_reference")
        g_plus, = coords.gradients([step * v], evaluator)
        return float(v @ (g_plus - gradient_at_reference) / step)
    g_plus, g_minus = coords.gradients([step * v, -step * v], evaluator)
    return float(v @ (g_plus - g_minus) / (2 * step))


def steepest_descent_bound(gradient_norm: float, curvature: float) -> float:
    """``1/2 |g|^2 / (v^T H v)`` along ``v = g/|g|``: a lower bound on ``1/2 g^T H^-1 g``."""
    if gradient_norm == 0.0:
        return 0.0
    return 0.5 * gradient_norm ** 2 / curvature if curvature > 0 else float("nan")


@dataclass
class RelaxEstimate:
    drop: float                 #: eV; NaN when aborted
    converged: bool
    steps: int
    max_displacement: float     #: A, largest |u| reached along the path
    max_strain: float           #: largest |E_GL| component reached along the path
    trust_exceeded: bool
    aborted: bool               #: left the abort region: E_corr has no nearby minimum
    final: Atoms


def relax_estimate(reference: Atoms, calculator: GradientMatchedCalculator, symprec: float,
                   fmax: float = 1e-3, steps: int = 500, maxstep: float = 0.05,
                   trust_displacement: float = 0.15, trust_strain: float = 0.02,
                   abort_displacement: float = 1.0, abort_strain: float = 0.2) -> RelaxEstimate:
    """Relax on ``E_corr`` with the space group fixed; the drop is ``E(x0) - E(x*)``.

    Where the MLIP's curvature is negative along a direction the DFT gradient has a
    component on, the linear term makes ``E_corr`` unbounded below and the relaxation
    runs away. It is stopped once any atom moves ``abort_displacement`` or the strain
    reaches ``abort_strain``, and the drop is reported as NaN: the estimate does not
    exist there, which is itself the finding.
    """
    atoms = reference.copy()
    atoms.set_constraint()
    atoms.calc = calculator
    atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
    e0 = atoms.get_potential_energy()
    match = calculator.match
    extremes = {"u": 0.0, "strain": 0.0}

    def track():
        u = match.displacement(atoms.positions, atoms.cell.array)
        extremes["u"] = max(extremes["u"], float(np.linalg.norm(u, axis=1).max()))
        extremes["strain"] = max(extremes["strain"],
                                 float(np.abs(match.green_lagrange(atoms.cell.array)).max()))

    optimizer = BFGS(FrechetCellFilter(atoms), maxstep=maxstep, logfile=None)
    converged, aborted = False, False
    for converged in optimizer.irun(fmax=fmax, steps=steps):
        track()
        if extremes["u"] > abort_displacement or extremes["strain"] > abort_strain:
            aborted, converged = True, False
            break
    e1 = atoms.get_potential_energy()
    return RelaxEstimate(
        drop=float("nan") if aborted else float(e0 - e1), converged=bool(converged),
        steps=int(optimizer.nsteps), max_displacement=extremes["u"], max_strain=extremes["strain"],
        trust_exceeded=extremes["u"] > trust_displacement or extremes["strain"] > trust_strain,
        aborted=aborted, final=atoms)
