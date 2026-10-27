"""MLIP molecular dynamics of Li conduction, the ICML 2026 pipeline's protocol (sec. 3.4).

* A supercell of at least 100 atoms whose inscribed sphere has a diameter of at least
  10 A (every perpendicular width >= 10 A).
* Positions relaxed to 0.005 eV/A in the fixed cell.
* 10 ps of thermalisation, then 100 ps of NVT production.  The paper used a Berendsen
  then a Nose-Hoover thermostat; this uses Langevin BAOAB for both, as
  ``scripts/cu_ge_te_switching/s4_md.py`` does.  The time step is 2 fs, 0.5 fs with H
  (the paper does not give one; 1 fs would double a cost of 16-110 ms per ORB step).
* Diffusion coefficients from 15 ps windows of the production run, the MSD slope fitted
  between 3 and 15 ps in each window; mean and standard error over windows (Kahle et
  al. 2020).  The cell's centre-of-mass drift, which the Langevin thermostat does not
  stop, is subtracted first.

Filters of the paper's Table 1: the slowest non-Li species at 1000 K must have
``6 D * 100 ps <= 2 A^2`` (the framework does not melt), the Li MSD at 500 K must exceed
4 A^2 at some time, and the 300 K conductivity is extrapolated from 500 K with a
pessimistic 0.5 eV activation energy.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from wyckoff_transformer.transport.sources import E_CHARGE, K_B

logger = logging.getLogger(__name__)

K_B_EV = 8.617333262e-5


def _widths(cell: np.ndarray) -> np.ndarray:
    volume = abs(np.linalg.det(cell))
    return np.array([volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
                     for i in range(3)])


def md_supercell(atoms, min_atoms: int = 100, min_width: float = 10.0, max_repeat: int = 64):
    """The smallest supercell with every perpendicular width >= *min_width*, >= *min_atoms*.

    Sizes are tried in increasing order; for each, ASE's most cubic supercell of that
    size is checked against the width rule.  A diagonal repetition is the fallback, so a
    96-atom primitive garnet becomes its 192-atom cube rather than a 768-atom 2x2x2.
    """
    from ase.build import find_optimal_cell_shape, make_supercell

    cell = np.asarray(atoms.cell)
    start = max(1, math.ceil(min_atoms / len(atoms)))
    for n in range(start, max_repeat + 1):
        try:
            matrix = find_optimal_cell_shape(cell, n, "sc", verbose=False)
        except Exception as exc:  # noqa: BLE001 - ASE raises a bare error when it finds none
            logger.debug("no %d-fold supercell shape: %r", n, exc)
            continue
        if matrix is None or round(abs(np.linalg.det(matrix))) != n:
            continue
        if _widths(matrix @ cell).min() >= min_width - 1e-6:
            return make_supercell(atoms, matrix), matrix
    widths = _widths(cell)
    repeats = np.maximum(1, np.ceil(min_width / widths - 1e-9)).astype(int)
    while len(atoms) * repeats.prod() < min_atoms:
        repeats[np.argmin(widths * repeats)] += 1
    return make_supercell(atoms, np.diag(repeats)), np.diag(repeats)


@dataclass
class Trajectory:
    """Unwrapped positions of a production run, sampled every *dt_ps*."""

    symbols: list[str]
    positions: np.ndarray            # (frames, atoms, 3), A
    dt_ps: float
    temperature: float
    volume: float
    seconds: float
    mean_temperature: float = math.nan
    extra: dict = field(default_factory=dict)


def run_nvt(atoms, calculator, temperature: float, thermalise_ps: float = 10.0,
            production_ps: float = 100.0, timestep_fs: float | None = None,
            sample_fs: float = 20.0, seed: int = 0, relax_fmax: float = 0.005,
            friction_tau_fs: float = 100.0) -> Trajectory:
    """Relax positions, thermalise and run NVT production; return the sampled trajectory."""
    from ase import units
    from ase.md.langevinbaoab import LangevinBAOAB
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.optimize import FIRE

    atoms = atoms.copy()
    atoms.calc = calculator
    if timestep_fs is None:
        timestep_fs = 0.5 if "H" in atoms.get_chemical_symbols() else 2.0
    if relax_fmax:
        FIRE(atoms, logfile=None).run(fmax=relax_fmax, steps=2000)
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    Stationary(atoms)
    dyn = LangevinBAOAB(atoms, timestep_fs * units.fs, temperature_K=temperature,
                        T_tau=friction_tau_fs * units.fs, rng=rng)
    started = time.time()
    dyn.run(round(thermalise_ps * 1000 / timestep_fs))
    every = max(1, round(sample_fs / timestep_fs))
    n_samples = round(production_ps * 1000 / timestep_fs / every)
    frames = np.empty((n_samples + 1, len(atoms), 3), dtype=np.float64)
    temperatures = np.empty(n_samples + 1)
    frames[0] = atoms.get_positions()
    temperatures[0] = atoms.get_temperature()
    for k in range(1, n_samples + 1):
        dyn.run(every)
        frames[k] = atoms.get_positions()
        temperatures[k] = atoms.get_temperature()
    return Trajectory(symbols=atoms.get_chemical_symbols(), positions=frames,
                      dt_ps=every * timestep_fs / 1000, temperature=temperature,
                      volume=atoms.get_volume(), seconds=time.time() - started,
                      mean_temperature=float(temperatures.mean()),
                      extra={"timestep_fs": timestep_fs, "n_atoms": len(atoms)})


def windowed_diffusion(positions: np.ndarray, dt_ps: float, window_ps: float = 15.0,
                       fit_from_ps: float = 3.0) -> tuple[float, float, int]:
    """``(D, s.e.m., windows)`` in A^2/ps for one set of atoms.

    The run is cut into consecutive windows of *window_ps*; in each, the MSD from the
    window's first frame is averaged over the atoms and its slope fitted between
    *fit_from_ps* and the window's end.  D is a sixth of the slope.
    """
    per_window = round(window_ps / dt_ps)
    start_fit = round(fit_from_ps / dt_ps)
    slopes = []
    for start in range(0, positions.shape[0] - per_window, per_window):
        block = positions[start:start + per_window + 1]
        msd = ((block - block[0]) ** 2).sum(axis=-1).mean(axis=-1)
        times = np.arange(len(msd)) * dt_ps
        slope = np.polyfit(times[start_fit:], msd[start_fit:], 1)[0]
        slopes.append(slope / 6.0)
    slopes = np.array(slopes)
    if not len(slopes):
        return math.nan, math.nan, 0
    sem = slopes.std(ddof=1) / math.sqrt(len(slopes)) if len(slopes) > 1 else math.nan
    return float(slopes.mean()), float(sem), len(slopes)


def conductivity(d_a2_ps: float, n_carriers: int, volume_a3: float, temperature: float) -> float:
    """Nernst-Einstein conductivity in S/cm from D in A^2/ps."""
    d_m2_s = d_a2_ps * 1e-20 / 1e-12
    n_m3 = n_carriers / (volume_a3 * 1e-30)
    return n_m3 * E_CHARGE ** 2 * d_m2_s / (K_B * temperature) / 100.0


def extrapolate(sigma: float, from_k: float, to_k: float, activation_ev: float) -> float:
    """sigma*T Arrhenius: sigma(T2) = sigma(T1) T1/T2 exp(-Ea/k (1/T2 - 1/T1))."""
    return sigma * from_k / to_k * math.exp(-activation_ev / K_B_EV * (1 / to_k - 1 / from_k))


def remove_centre_of_mass_drift(positions: np.ndarray, symbols) -> np.ndarray:
    """Positions with the cell's mass-weighted centre-of-mass displacement subtracted.

    A Langevin thermostat does not conserve momentum, so the whole cell random-walks
    with D = kT tau / M: 0.014 A^2/ps for a 104-atom Li6PS5Cl cell at 500 K, enough to
    make a rigid framework look molten.
    """
    from ase.data import atomic_masses, atomic_numbers

    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    com = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
    return positions - (com - com[0])[:, None, :]


def analyse(trajectory: Trajectory, mobile: str = "Li") -> dict:
    """Per-species diffusion, the Li conductivity and the ICML filter quantities.

    The centre-of-mass drift is removed first (:func:`remove_centre_of_mass_drift`).
    """
    symbols = np.array(trajectory.symbols)
    trajectory = Trajectory(**{**trajectory.__dict__, "positions": remove_centre_of_mass_drift(
        trajectory.positions, trajectory.symbols)})
    out = {"temperature": trajectory.temperature, "mean_temperature": trajectory.mean_temperature,
           "md_seconds": trajectory.seconds, **trajectory.extra}
    framework_d = []
    for species in sorted(set(symbols)):
        mask = symbols == species
        d, sem, n = windowed_diffusion(trajectory.positions[:, mask], trajectory.dt_ps)
        out[f"D_{species}"], out[f"D_{species}_sem"] = d, sem
        out["windows"] = n
        if species != mobile:
            framework_d.append(d)
    li = trajectory.positions[:, symbols == mobile]
    msd_li = ((li - li[0]) ** 2).sum(axis=-1).mean(axis=-1)
    out["li_msd_max"] = float(msd_li.max())
    out["framework_D_slowest"] = float(min(framework_d)) if framework_d else math.nan
    n_li = int((symbols == mobile).sum())
    d, sem = out[f"D_{mobile}"], out[f"D_{mobile}_sem"]
    out["sigma"] = conductivity(d, n_li, trajectory.volume, trajectory.temperature)
    out["sigma_sem"] = conductivity(sem, n_li, trajectory.volume, trajectory.temperature) \
        if math.isfinite(sem) else math.nan
    return out
