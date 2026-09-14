"""Check an MLIP's curvature and the gradient-matched estimator against DFT relaxations.

A VASP relaxation trajectory records, per ionic step, the geometry, the energy, the
forces and the stress. Consecutive steps therefore measure the DFT energy surface
along exactly the directions DFT moved, which is what is needed to test whether an
MLIP's curvature -- the only thing :mod:`gradient_matched` takes from it -- is right.

Everything is done in the generalized coordinates of
:class:`~wyckoff_transformer.cryspr.gradient_matched.GeneralizedCoordinates` about a
reference step, projected on its symmetry-allowed, translation-free subspace.
Coordinates of another step are read off its geometry (:func:`relative_coordinates`,
with any rigid rotation of the cell removed by polar decomposition), and its forces and
stress are turned into the generalized gradient in the same frame.

Within one VASP calculation only: the plane-wave basis is reset between ``relax1`` and
``relax2``, and the energy is discontinuous there (:func:`calc_boundaries` measures that
jump instead).

Checks, one function each:

* T0 :func:`analyse_pairs` -- energy change against the trapezoid prediction
  ``+1/2 (g_k + g_k+1) . dq`` (``g = dE/dq``, i.e. minus the generalized force); forces
  and stress inconsistent with the energy show up as a residual.
* T1 :func:`analyse_pairs` -- curvature along the DFT step, ``(g_k+1 - g_k) . dq/|dq|^2``,
  against the MLIP's along the same ``dq``: its secant between the same two geometries,
  its tangent at ``q_k`` (central difference, plain MLIP), and the tangent of the
  gradient-matched surface, which differs only by the ``V0 dsigma : E_GL`` term.
* T2 :func:`estimate_at` -- the estimator at an intermediate step, to be compared with
  the DFT drop to the final step.
* T3 :func:`calc_boundaries` -- last step of one calculation against the first of the
  next.
* T4 :func:`dft_secant_hessian` -- a DFT-only Hessian from the trajectory (BFGS, and a
  symmetric least-squares secant fit), restricted to the subspace the trajectory
  explored.

:func:`synthetic_task` writes a trajectory in the task format from any ASE calculator;
with the same calculator as the "MLIP", every check must pass trivially.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from wyckoff_transformer.cryspr import gradient_matched as gm

KBAR = gm.KBAR_PER_EV_PER_A3


@dataclass
class Frame:
    """One ionic step: ASE units, ASE stress sign."""

    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: np.ndarray          # 3x3, eV/A^3, ASE sign
    step_index: int
    n_electronic: int | None = None
    #: every energy the step reports (``e_fr_energy``, ``e_wo_entrp``, ``e_0_energy``):
    #: which one the forces are the derivative of depends on ISMEAR.
    energies: dict | None = None

    @property
    def stress_kbar_vasp(self) -> np.ndarray:
        return -self.stress * KBAR


def load_task(path: Path | str) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def frames_of(calc: dict, energy_key: str) -> list[Frame]:
    """The steps of one calculation that carry an energy, forces and a 3x3 stress."""
    frames = []
    for index, step in enumerate(calc["steps"]):
        forces, stress, energy = step.get("forces"), step.get("stress"), step.get(energy_key)
        if energy is None or not forces or not stress or np.asarray(stress).size != 9:
            continue
        atoms = Atoms(symbols=list(step["species"]), scaled_positions=np.asarray(step["frac"], float),
                      cell=np.asarray(step["lattice"], float), pbc=True)
        energies = {key: float(step[key]) for key in ("e_fr_energy", "e_wo_entrp", "e_0_energy")
                    if step.get(key) is not None}
        frames.append(Frame(atoms, float(energy), np.asarray(forces, float),
                            gm.lemat_stress_to_ase(stress), index, step.get("n_electronic"), energies))
    return frames


def relative_coordinates(coords: gm.GeneralizedCoordinates, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """``(q, Q)``: ``atoms`` equals ``coords.atoms_at(q)`` rigidly rotated by ``Q``.

    With row-vector cells, ``cell = cell0 G`` and ``G = S Q`` (``S`` symmetric positive,
    ``Q`` orthogonal), so the strain is ``S - I``. Positions follow from fractional
    coordinates, minimum image against the reference.
    """
    deformation = np.linalg.inv(coords.cell0) @ atoms.cell.array
    values, vectors = np.linalg.eigh(deformation @ deformation.T)
    stretch = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    rotation = np.linalg.inv(stretch) @ deformation
    strain = stretch - np.eye(3)
    q_cell = coords.length * np.einsum("kab,ab->k", coords.strain_basis, strain)
    scaled = atoms.get_scaled_positions(wrap=False) - coords.reference.get_scaled_positions(wrap=False)
    scaled -= np.round(scaled)
    u = scaled @ coords.cell0
    return np.concatenate([u.ravel(), q_cell]), rotation


def gradient_in(coords: gm.GeneralizedCoordinates, q: np.ndarray, rotation: np.ndarray,
                forces: np.ndarray, stress: np.ndarray, volume: float) -> np.ndarray:
    """Generalized gradient at ``q`` from forces and stress given in the rotated frame."""
    return coords.gradient(q, forces @ rotation.T, rotation @ stress @ rotation.T, volume)


def rotation_angle(rotation: np.ndarray) -> float:
    """Degrees; atan2 of sin and cos, since arccos alone loses ~1e-5 deg near identity."""
    sin = np.linalg.norm(rotation - rotation.T) / (2 * np.sqrt(2))
    cos = (np.trace(rotation) - 1) / 2
    return float(np.degrees(np.arctan2(sin, cos)))


def analyse_pairs(frames: Sequence[Frame], symprec: float, evaluator: gm.BatchEvaluator | None,
                  fd_step: float = 1e-3) -> list[dict]:
    """T0 and T1 for every consecutive pair of one calculation.

    Energies are total (eV) and gradients in eV/A; ``dE_trapezoid_positions`` +
    ``dE_trapezoid_cell`` = ``dE_trapezoid``. MLIP curvatures are NaN when ``evaluator``
    is None.
    """
    n = len(frames)
    if n < 2:
        return []
    coords = [gm.GeneralizedCoordinates(frame.atoms, symprec) for frame in frames[:-1]]
    final = frames[-1]
    pairs = []
    for k in range(n - 1):
        c, f0, f1 = coords[k], frames[k], frames[k + 1]
        basis = c.basis
        p = c.n_position_modes
        q1, rot1 = relative_coordinates(c, f1.atoms)
        q_final, _ = relative_coordinates(c, final.atoms)
        dq = basis.T @ q1
        g0 = basis.T @ c.gradient(np.zeros(c.dim), f0.forces, f0.stress, f0.atoms.get_volume())
        g1 = basis.T @ gradient_in(c, q1, rot1, f1.forces, f1.stress, f1.atoms.get_volume())
        mean_g = 0.5 * (g0 + g1)
        norm = float(np.linalg.norm(dq))
        to_final = basis.T @ q_final
        record = {
            "k": k, "step_index": f0.step_index, "steps_to_end": n - 2 - k, "n_atoms": len(f0.atoms),
            "n_modes": basis.shape[1], "n_position_modes": p,
            "dE": f1.energy - f0.energy,
            # g is dE/dq, so the trapezoid rule predicts dE = +1/2 (g_k + g_k+1) . dq.
            "dE_trapezoid": float(mean_g @ dq),
            "dE_trapezoid_positions": float(mean_g[:p] @ dq[:p]),
            "dE_trapezoid_cell": float(mean_g[p:] @ dq[p:]),
            "abs_dq": norm,
            "dq_outside_subspace": float(np.linalg.norm(q1 - basis @ dq)),
            "strain_fraction": float(np.sum(dq[p:] ** 2) / norm ** 2) if norm > 0 else np.nan,
            "rotation_deg": rotation_angle(rot1),
            "distance_to_final": float(np.linalg.norm(to_final)),
            "final_strain_fraction": (float(np.sum(to_final[p:] ** 2) / np.sum(to_final ** 2))
                                      if np.sum(to_final ** 2) > 0 else np.nan),
            "g_along": float(-g0 @ dq / norm) if norm > 0 else np.nan,
            "dg_along": float((g1 - g0) @ dq / norm) if norm > 0 else np.nan,
            "kappa_dft": float((g1 - g0) @ dq / norm ** 2) if norm > 0 else np.nan,
            "abs_g0": float(np.linalg.norm(g0)),
        }
        record["residual"] = record["dE"] - record["dE_trapezoid"]
        # A constant-basis relaxation's energy and a constant-cutoff stress differ by an
        # isotropic (Pulay) term, which shows up as a residual linear in the volume change.
        v0, v1 = f0.atoms.get_volume(), f1.atoms.get_volume()
        isotropic = c.length * np.trace(c.strain(q1)) / np.sqrt(3)
        record.update(volume0=v0, volume1=v1, dV=v1 - v0,
                      iso_fraction=float(isotropic ** 2 / np.sum(q1 ** 2)) if np.sum(q1 ** 2) > 0 else np.nan)
        for key in sorted(set(f0.energies or {}) & set(f1.energies or {})):
            record[f"dE_{key}"] = f1.energies[key] - f0.energies[key]
            record[f"residual_{key}"] = record[f"dE_{key}"] - record["dE_trapezoid"]
        pairs.append((record, c, q1, rot1, dq))

    records = [r for r, *_ in pairs]
    if evaluator is None:
        for r in records:
            r.update(kappa_mlip_secant=np.nan, kappa_mlip_tangent=np.nan, kappa_matched_tangent=np.nan,
                     mlip_g_along=np.nan)
        return records

    at_frames = evaluator([frame.atoms for frame in frames])
    probes, owners = [], []
    for i, (r, c, q1, rot1, dq) in enumerate(pairs):
        if r["abs_dq"] > 0:
            v = c.basis @ (dq / r["abs_dq"])
            probes += [c.atoms_at(fd_step * v), c.atoms_at(-fd_step * v)]
            owners.append((i, v))
    at_probes = evaluator(probes) if probes else []
    for j, (i, v) in enumerate(owners):
        r, c, q1, rot1, dq = pairs[i]
        k = r["k"]
        f0 = frames[k]
        e0, e1 = at_frames[k], at_frames[k + 1]
        basis, unit = c.basis, dq / r["abs_dq"]
        m0 = basis.T @ c.gradient(np.zeros(c.dim), e0[1], e0[2], f0.atoms.get_volume())
        m1 = basis.T @ gradient_in(c, q1, rot1, e1[1], e1[2], frames[k + 1].atoms.get_volume())
        r["kappa_mlip_secant"] = float((m1 - m0) @ unit / r["abs_dq"])
        r["mlip_g_along"] = float(-m0 @ unit)
        plus, minus = probes[2 * j], probes[2 * j + 1]
        (_, fp, sp), (_, fm, sm) = at_probes[2 * j], at_probes[2 * j + 1]
        qp, qm = fd_step * v, -fd_step * v
        gp = c.gradient(qp, fp, sp, plus.get_volume())
        gm_ = c.gradient(qm, fm, sm, minus.get_volume())
        r["kappa_mlip_tangent"] = float(v @ (gp - gm_) / (2 * fd_step))
        match = gm.GradientMatch(f0.atoms, e0[1], e0[2], f0.forces, f0.stress)
        dp, dm = match.terms(plus.positions, plus.cell.array), match.terms(minus.positions, minus.cell.array)
        gpc = c.gradient(qp, fp + dp[1], sp + dp[2], plus.get_volume())
        gmc = c.gradient(qm, fm + dm[1], sm + dm[2], minus.get_volume())
        r["kappa_matched_tangent"] = float(v @ (gpc - gmc) / (2 * fd_step))
    for r in records:
        for key in ("kappa_mlip_secant", "kappa_mlip_tangent", "kappa_matched_tangent", "mlip_g_along"):
            r.setdefault(key, np.nan)
    return records


def estimate_at(frame: Frame, base_calculator: Calculator, base_evaluator: gm.BatchEvaluator, symprec: float,
                fd_step: float = 1e-3, soft_curvature: float = 0.1, relax: bool = True,
                relax_kwargs: dict | None = None) -> tuple[dict, gm.NewtonEstimate, gm.GeneralizedCoordinates]:
    """T2: the gradient-matched estimator at one step, in eV for the whole cell."""
    corrected = gm.GradientMatchedCalculator.from_reference(base_calculator, frame.atoms, frame.forces, frame.stress)
    coords = gm.GeneralizedCoordinates(frame.atoms, symprec)
    evaluator = gm.corrected_evaluator(base_evaluator, corrected.match)
    newton = gm.newton_estimate(coords, evaluator, step=fd_step, soft_curvature=soft_curvature)
    if newton.gradient_norm > 0:
        curvature = gm.curvature_along(coords, evaluator, coords.basis @ newton.reduced_gradient, step=fd_step)
    else:
        curvature = np.nan
    record = {
        "newton": newton.drop, "newton_positions": newton.drop_positions, "newton_cell": newton.drop_cell,
        "newton_soft_share": newton.soft_share, "n_negative": newton.n_negative, "n_soft": newton.n_soft,
        "min_curvature": newton.min_curvature, "n_modes": newton.n_modes, "gradient_norm": newton.gradient_norm,
        "sd_curvature": curvature, "sd_bound": gm.steepest_descent_bound(newton.gradient_norm, curvature),
    }
    if relax:
        relaxed = gm.relax_estimate(frame.atoms, corrected, symprec=symprec, **(relax_kwargs or {}))
        record.update(relax=relaxed.drop, relax_converged=relaxed.converged, relax_aborted=relaxed.aborted,
                      relax_steps=relaxed.steps, trust_exceeded=relaxed.trust_exceeded,
                      max_displacement=relaxed.max_displacement, max_strain=relaxed.max_strain)
    return record, newton, coords


def calc_boundaries(frames_by_calc: Sequence[Sequence[Frame]], symprec: float) -> list[dict]:
    """T3: last step of each calculation against the first step of the next."""
    out = []
    for i in range(len(frames_by_calc) - 1):
        if not frames_by_calc[i] or not frames_by_calc[i + 1]:
            continue
        first, a, b = frames_by_calc[i][0], frames_by_calc[i][-1], frames_by_calc[i + 1][0]
        coords = gm.GeneralizedCoordinates(a.atoms, symprec)
        q, rot = relative_coordinates(coords, b.atoms)
        sa, sb = a.stress_kbar_vasp, b.stress_kbar_vasp
        d = sb - sa
        hydro = np.trace(d) / 3
        v_first, v_a = first.atoms.get_volume(), a.atoms.get_volume()
        energy_jumps = {f"dE_{key}_per_atom": (b.energies[key] - a.energies[key]) / len(a.atoms)
                        for key in sorted(set(a.energies or {}) & set(b.energies or {}))}
        out.append({
            "boundary": i, "n_atoms": len(a.atoms),
            "elements": sorted(set(a.atoms.get_chemical_symbols())),
            "volume_per_atom": v_a / len(a.atoms),
            # A fixed plane-wave basis set up at the calc's first geometry: its effective
            # cutoff at the last geometry is ENCUT * (V_first / V_last)^(2/3).
            "calc_volume_change": v_a / v_first - 1, "effective_cutoff_ratio": (v_first / v_a) ** (2 / 3),
            "calc_steps": len(frames_by_calc[i]), **energy_jumps,
            "same_species": list(a.atoms.symbols) == list(b.atoms.symbols),
            "max_abs_q": float(np.abs(q).max()), "rotation_deg": rotation_angle(rot),
            "dE_per_atom": (b.energy - a.energy) / len(a.atoms),
            "d_hydrostatic_kbar": float(hydro),
            "d_deviatoric_kbar": float(np.linalg.norm(d - hydro * np.eye(3))),
            "a_hydrostatic_kbar": float(np.trace(sa) / 3), "b_hydrostatic_kbar": float(np.trace(sb) / 3),
            "a_max_force": float(np.abs(a.forces).max()), "b_max_force": float(np.abs(b.forces).max()),
            "d_max_force": float(np.abs(b.forces - a.forces).max()),
        })
    return out


def _bfgs(pairs: Sequence[tuple[np.ndarray, np.ndarray]], dim: int) -> tuple[np.ndarray, int]:
    accepted = [(s, y) for s, y in pairs if y @ s > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y)]
    if not accepted:
        return np.eye(dim), len(pairs)
    s, y = accepted[-1]
    hessian = np.eye(dim) * (y @ y) / (y @ s)
    for s, y in accepted:
        hs = hessian @ s
        hessian = hessian + np.outer(y, y) / (y @ s) - np.outer(hs, hs) / (s @ hs)
    return hessian, len(pairs) - len(accepted)


def _symmetric_secant_fit(steps: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """Symmetric ``H`` minimising ``sum_i |H s_i - y_i|^2`` (rows of ``steps``/``grads``)."""
    d = steps.shape[1]
    iu = np.triu_indices(d)
    design = np.zeros((steps.shape[0] * d, len(iu[0])))
    for col, (a, b) in enumerate(zip(*iu)):
        e = np.zeros((d, d)); e[a, b] = e[b, a] = 1.0
        design[:, col] = (steps @ e.T).ravel()
    params, *_ = np.linalg.lstsq(design, grads.ravel(), rcond=None)
    h = np.zeros((d, d)); h[iu] = params
    return h + np.triu(h, 1).T


def _harmonic_drop(hessian: np.ndarray, gradient: np.ndarray) -> tuple[float, int]:
    values, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    c2 = (vectors.T @ gradient) ** 2
    positive = values > 0
    return float(0.5 * np.sum(c2[positive] / values[positive])), int((~positive).sum())


def dft_secant_hessian(frames: Sequence[Frame], symprec: float, q_floor: float = 1e-3,
                       mlip_hessian: np.ndarray | None = None, max_fit_dim: int = 40) -> dict:
    """T4: a Hessian from DFT alone, in the final step's symmetric subspace.

    ``mlip_hessian`` is the reduced Hessian of :func:`estimate_at` at the same final step
    (same symprec, hence the same basis), for a like-for-like harmonic drop in the
    explored subspace and for checking the MLIP's negative modes against DFT there.
    """
    final = frames[-1]
    coords = gm.GeneralizedCoordinates(final.atoms, symprec)
    basis = coords.basis
    qs, gs = [], []
    for frame in frames:
        q, rot = relative_coordinates(coords, frame.atoms)
        qs.append(basis.T @ q)
        gs.append(basis.T @ gradient_in(coords, q, rot, frame.forces, frame.stress, frame.atoms.get_volume()))
    qs, gs = np.array(qs), np.array(gs)
    m = basis.shape[1]
    steps, grads = np.diff(qs, axis=0), np.diff(gs, axis=0)
    keep = np.linalg.norm(steps, axis=1) >= q_floor
    steps, grads = steps[keep], grads[keep]
    g_final = gs[-1]
    record = {"n_modes": m, "n_pairs": int(len(steps)), "gradient_norm": float(np.linalg.norm(g_final))}
    if m == 0 or len(steps) == 0:
        return record
    _, sv, vt = np.linalg.svd(steps, full_matrices=False)
    explored = vt[sv >= max(q_floor, 1e-6 * sv.max())].T          # (m, d)
    d = explored.shape[1]
    g_e = explored.T @ g_final
    record.update(explored_dim=d,
                  gradient_fraction_explored=float(g_e @ g_e / (g_final @ g_final)) if g_final @ g_final > 0 else np.nan)
    hessian, skipped = _bfgs(list(zip(steps, grads)), m)
    record["bfgs_skipped_pairs"] = skipped
    record["delta_bfgs_explored"], record["bfgs_negative_explored"] = _harmonic_drop(explored.T @ hessian @ explored, g_e)
    fit = None
    if d <= max_fit_dim:
        fit = _symmetric_secant_fit(steps @ explored, grads @ explored)
        record["delta_fit_explored"], record["fit_negative_explored"] = _harmonic_drop(fit, g_e)
        record["fit_min_eigenvalue"] = float(np.linalg.eigvalsh(fit).min())
    if mlip_hessian is not None and mlip_hessian.shape == (m, m):
        record["delta_mlip_explored"], record["mlip_negative_explored"] = _harmonic_drop(
            explored.T @ mlip_hessian @ explored, g_e)
        values, vectors = np.linalg.eigh(mlip_hessian)
        checks = []
        for lam, vec in zip(values, vectors.T):
            if lam >= 0:
                continue
            inside = explored.T @ vec
            overlap = float(np.linalg.norm(inside))
            entry = {"mlip_eigenvalue": float(lam), "overlap_explored": overlap,
                     "gradient_along": float(abs(vec @ g_final))}
            if overlap > 0.3:
                w = inside / overlap
                entry["bfgs_curvature"] = float(w @ explored.T @ hessian @ explored @ w)
                if fit is not None:
                    entry["fit_curvature"] = float(w @ fit @ w)
            checks.append(entry)
        record["mlip_negative_modes"] = checks
    return record


def synthetic_task(atoms: Atoms, calculator: Calculator, symprec: float, relax1_steps: int,
                   fmax: float = 1e-3, max_steps: int = 300, maxstep: float = 0.05,
                   ediff: float = 1e-6, task_id: str = "synthetic") -> dict:
    """A two-calculation relaxation in the task format, with ``calculator`` as the "DFT".

    ``relax2`` restarts from ``relax1``'s last geometry with a fresh optimizer, and its
    first step repeats that geometry, as VASP's does.
    """
    work = atoms.copy()
    work.calc = calculator
    work.set_constraint(FixSymmetry(work, symprec=symprec))

    def record() -> dict:
        energy = float(work.get_potential_energy())
        stress = work.get_stress(voigt=False, apply_constraint=False)
        return {"lattice": work.cell.array.tolist(), "species": list(work.get_chemical_symbols()),
                "frac": work.get_scaled_positions(wrap=False).tolist(),
                "e_fr_energy": energy, "e_wo_entrp": energy, "e_0_energy": energy,
                "forces": work.get_forces(apply_constraint=False).tolist(),
                "stress": (-stress * KBAR).tolist(), "n_electronic": 0}

    calcs = []
    for name, budget in (("relax1", relax1_steps), ("relax2", max_steps)):
        steps = [record()]
        optimizer = BFGS(FrechetCellFilter(work), maxstep=maxstep, logfile=None)
        for i, converged in enumerate(optimizer.irun(fmax=fmax, steps=budget)):
            if i > 0:
                steps.append(record())
            if converged and name == "relax2":
                break
        calcs.append({"name": name, "incar": {"ISIF": 3, "IBRION": 2, "ISMEAR": 0, "EDIFF": ediff},
                      "kpoints": None, "steps": steps})
    return {"immutable_id": task_id, "task_id": task_id, "group": "synthetic", "energy_key": "e_fr_energy",
            "lemat_site_order": list(range(len(atoms))), "calcs": calcs}
