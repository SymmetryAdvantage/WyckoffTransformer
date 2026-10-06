"""Stage 7: read the MD runs.

Heating runs (pseudo-cubic supercells): per frame, the deviation of the cell
angles from 90 deg (the macroscopic rhombohedral strain of GeTe) and the local
Peierls distortion, i.e. for each Ge the mean of its three long Ge-Te bonds
minus the mean of its three short ones.  Ferroelectric GeTe has ~0.3-0.4 A at
300 K; a displacive transition sends both to zero, an order-disorder one keeps
the local distortion while the cell becomes cubic.

Melt-quench runs: the summary, plus coordination numbers of the relaxed glass
and the fraction of Ge in tetrahedral rather than defective-octahedral
environments (the known structural signature of amorphous GeTe).

Writes ``md_heat.csv`` and ``md_glass.csv``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from common import OUT_DIR, setup_logging

log = setup_logging()
BOND_CUTOFF = 3.2


def peierls(atoms) -> float:
    i, j, d = neighbor_list("ijd", atoms, 4.0)
    symbols = np.array(atoms.get_chemical_symbols())
    values = []
    for ge in np.where(symbols == "Ge")[0]:
        mask = (i == ge) & (symbols[j] == "Te")
        dist = np.sort(d[mask])[:6]
        if len(dist) == 6:
            values.append(dist[3:].mean() - dist[:3].mean())
    return float(np.mean(values)) if values else float("nan")


def te_crystalline_fraction(atoms, cutoff: float | None = None) -> float:
    """Fraction of Te atoms in a crystal-like environment of the Te sublattice.

    The Te atoms form an fcc lattice in rocksalt-like GeTe and in the
    zinc-blende-derived ternaries alike, so one measure serves all of them:
    Steinhardt q6 bond-order vectors over Te-Te neighbours (first shell, ~12),
    and a Te counts as crystalline when at least 8 of its neighbours have a
    normalised q6.q6* > 0.7 (ten Wolde-Frenkel connections).
    """
    from scipy.special import sph_harm_y

    te = atoms[[s == "Te" for s in atoms.get_chemical_symbols()]]
    if cutoff is None:
        cutoff = 1.2 * (te.get_volume() / len(te) * 4) ** (1 / 3) / np.sqrt(2)  # 1.2 x fcc nn distance
    i, j, vec = neighbor_list("ijD", te, cutoff)
    if len(i) == 0:
        return float("nan")
    r = np.linalg.norm(vec, axis=1)
    theta = np.arccos(np.clip(vec[:, 2] / r, -1, 1))
    phi = np.arctan2(vec[:, 1], vec[:, 0])
    l = 6
    q = np.zeros((len(te), 2 * l + 1), dtype=complex)
    for m in range(-l, l + 1):
        y = sph_harm_y(l, m, theta, phi)
        q[:, m + l] = np.bincount(i, weights=y.real, minlength=len(te)) \
            + 1j * np.bincount(i, weights=y.imag, minlength=len(te))
    norm = np.linalg.norm(q, axis=1)
    q = q / np.where(norm > 0, norm, 1)[:, None]
    dots = np.real(np.sum(q[i] * np.conj(q[j]), axis=1))
    connections = np.bincount(i, weights=(dots > 0.7).astype(float), minlength=len(te))
    return float((connections >= 8).mean())


def heat_runs() -> pd.DataFrame:
    rows = []
    for run in sorted((OUT_DIR / "md").glob("heat_*")):
        if not (run / "log.csv").exists():
            continue
        frames = read(run / "trajectory.extxyz", ":")
        log_rows = pd.read_csv(run / "log.csv")
        for frame, lr in zip(frames[-len(log_rows):], log_rows.itertuples()):
            angles = frame.cell.cellpar()[3:]
            rows.append({"run": run.name, "target_T": lr.target_T, "T": lr.T,
                         "volume_per_atom": lr.volume_per_atom, "epot_per_atom": lr.epot_per_atom,
                         "angle_dev_deg": float(np.mean(np.abs(angles - 90.0))),
                         "peierls_A": peierls(frame),
                         "te_crystalline": te_crystalline_fraction(frame)})
    return pd.DataFrame(rows)


def coordination(atoms) -> dict:
    i, j = neighbor_list("ij", atoms, BOND_CUTOFF)
    symbols = np.array(atoms.get_chemical_symbols())
    counts = np.bincount(i, minlength=len(atoms))
    out = {}
    for el in sorted(set(symbols)):
        out[f"cn_{el}"] = float(counts[symbols == el].mean())
    ge = np.where(symbols == "Ge")[0]
    if len(ge):
        positions = atoms.get_positions()
        tetra = 0
        for g in ge:
            nb = j[i == g]
            if len(nb) != 4:
                continue
            vecs = np.array([atoms.get_distance(g, n, mic=True, vector=True) for n in nb])
            vecs /= np.linalg.norm(vecs, axis=1)[:, None]
            cosines = vecs @ vecs.T
            angles = np.degrees(np.arccos(np.clip(cosines[np.triu_indices(4, 1)], -1, 1)))
            if angles.mean() > 103:  # 109.5 tetrahedral vs 90 octahedral-like
                tetra += 1
        out["ge_tetrahedral_fraction"] = tetra / len(ge)
        del positions
    homopolar = sum(1 for a, b in zip(i, j) if symbols[a] == symbols[b] and a < b)
    out["homopolar_bonds_per_atom"] = homopolar / len(atoms)
    return out


def glass_runs() -> pd.DataFrame:
    rows = []
    for run in sorted((OUT_DIR / "md").glob("meltquench_*")):
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        glass = read(run / "amorphous_relaxed.extxyz")
        crystal_like = read(run / "start.extxyz")
        row = {**summary, **coordination(glass), "te_crystalline": te_crystalline_fraction(glass)}
        row.update({f"crystal_{k}": v for k, v in coordination(crystal_like).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def anneal_runs() -> pd.DataFrame:
    rows = []
    for run in sorted((OUT_DIR / "md").glob("anneal_*")):
        traj = run / "trajectory.extxyz"
        if not traj.exists():
            continue
        frames = read(traj, ":")
        for k, frame in enumerate(frames):
            rows.append({"run": run.name, "time_ps": 2.0 * (k + 1),
                         "epot_per_atom": frame.get_potential_energy() / len(frame)
                         if frame.calc is not None else np.nan,
                         "volume_per_atom": frame.get_volume() / len(frame),
                         "te_crystalline": te_crystalline_fraction(frame)})
    return pd.DataFrame(rows)


def main() -> None:
    heat = heat_runs()
    if len(heat):
        heat.to_csv(OUT_DIR / "md_heat.csv", index=False)
        for run, g in heat.groupby("run"):
            g = g.copy()
            g["T_bin"] = (g["target_T"] // 50) * 50
            log.info("%s\n%s", run, g.groupby("T_bin")[["volume_per_atom", "angle_dev_deg", "peierls_A"]]
                     .mean().round(3).to_string())
    anneal = anneal_runs()
    if len(anneal):
        anneal.to_csv(OUT_DIR / "md_anneal.csv", index=False)
        for run, g in anneal.groupby("run"):
            log.info("%s: crystalline Te fraction %.2f -> %.2f over %.0f ps", run,
                     g["te_crystalline"].iloc[:3].mean(), g["te_crystalline"].iloc[-3:].mean(),
                     g["time_ps"].max())
    glass = glass_runs()
    if len(glass):
        glass.to_csv(OUT_DIR / "md_glass.csv", index=False)
        log.info("\n%s", glass.T.to_string())


if __name__ == "__main__":
    main()
