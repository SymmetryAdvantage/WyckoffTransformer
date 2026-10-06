"""Stage 3b: polarisation-reversal barriers of the polar states.

An electric field couples to a polar state through its polarisation, and the
switch it drives is +P -> -P: the same crystal with the polar displacements
reversed.  The -P variant is the inverted structure, translated and permuted so
that every atom moves as little as possible; its cell is the same, so the path
between them is an ordinary fixed-cell NEB whose midpoint is centrosymmetric.

Reported per state: the barrier (meV/atom and meV per formula unit), the largest
polar displacement, and a point-charge polarisation with nominal charges
(Cu +1, Ge +4 or +2 in GeTe, Te -2).  Born effective charges in these
tellurides are anomalously large (GeTe: Z* ~ 6-8), so the nominal estimate is a
lower bound on |P|; the barrier is a coherent single-domain upper bound.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from common import OUT_DIR, REPO_ROOT, frozen, orb_calculator, relax, setup_logging
from pymatgen.core import Composition
from scipy.optimize import linear_sum_assignment

log = setup_logging()
E_CHARGE = 1.602176634e-19


def nominal_charges(symbols, formula: str) -> np.ndarray:
    ge = 2.0 if Composition(formula).reduced_formula == "GeTe" else None
    n_cu = symbols.count("Cu")
    n_ge = symbols.count("Ge")
    n_te = symbols.count("Te")
    if ge is None:
        ge = (2 * n_te - n_cu) / n_ge if n_ge else 0.0  # balance the formula
    table = {"Cu": 1.0, "Ge": ge, "Te": -2.0}
    return np.array([table[s] for s in symbols])


def reversed_variant(atoms):
    """-P variant of *atoms*: inverted, translated and permuted to minimise motion."""
    frac = atoms.get_scaled_positions()
    inv = -frac
    symbols = np.array(atoms.get_chemical_symbols())
    cell = atoms.cell[:]
    best = None
    ref = 0
    for cand in np.where(symbols == symbols[ref])[0]:
        shift = frac[ref] - inv[cand]
        moved = inv + shift
        perm = np.empty(len(atoms), dtype=int)
        worst = 0.0
        for el in set(symbols):
            idx = np.where(symbols == el)[0]
            d = moved[idx][None, :, :] - frac[idx][:, None, :]
            d -= np.round(d)
            dist = np.linalg.norm(d @ cell, axis=-1)
            rows, cols = linear_sum_assignment(dist)
            perm[idx[rows]] = idx[cols]
            worst = max(worst, dist[rows, cols].max())
        if best is None or worst < best[0]:
            best = (worst, moved[perm])
    worst, target = best
    target = target - np.round(target - frac)  # nearest images
    target = target - (target - frac).mean(axis=0)  # no net translation: only the polar motion
    out = atoms.copy()
    out.set_scaled_positions(target)
    return out, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=50.0, help="meV/atom")
    parser.add_argument("--images", type=int, default=7)
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max-atoms", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    states = pd.read_csv(OUT_DIR / "states.csv")
    chosen = states[states["polar"] & ~states["vacuum_artefact"]
                    & (1000 * states["e_above_hull"] <= args.window)
                    & (states["n_sites"] <= args.max_atoms)]
    tern = chosen["reduced_formula"].str.contains("Cu") & chosen["reduced_formula"].str.contains("Ge") \
        & chosen["reduced_formula"].str.contains("Te")
    chosen = chosen[tern | chosen["reduced_formula"].isin(["GeTe", "Cu2Te", "Cu3Te2", "CuTe"])]
    chosen = chosen.groupby("reduced_formula").head(3)
    if args.only:
        chosen = states[states["state_id"].isin(args.only)]
    log.info("%d polar states", len(chosen))

    calculator = orb_calculator(args.device)
    root = OUT_DIR / "polar"
    root.mkdir(parents=True, exist_ok=True)
    done_path = root / "polar.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(l)["state_id"] for l in done_path.read_text().splitlines()}
    for row in chosen.itertuples():
        if row.state_id in done:
            continue
        atoms, _ = relax(read(REPO_ROOT / row.cif), calculator, fmax=0.005)
        minus, worst = reversed_variant(atoms)
        minus, _ = relax(minus, calculator, fmax=0.01, cell=False)
        n = len(atoms)
        e_plus = atoms.get_potential_energy()
        e_minus = minus.get_potential_energy()
        ts = np.linspace(0, 1, args.images + 2)
        fa, fb = atoms.get_scaled_positions(), minus.get_scaled_positions()
        fb = fb - np.round(fb - fa)
        fb = fb - (fb - fa).mean(axis=0)
        images = []
        for t in ts:
            im = atoms.copy()
            im.set_scaled_positions((1 - t) * fa + t * fb)
            images.append(im)
        images[0], images[-1] = atoms, minus
        images[-1].set_scaled_positions(fb)  # same periodic image as the band
        for im in images:
            im.calc = calculator
        neb = NEB(images, climb=True, method="improvedtangent", allow_shared_calculator=True)
        converged = bool(FIRE(neb, logfile=None).run(fmax=args.fmax, steps=args.steps))
        energies = np.array([im.get_potential_energy() for im in images])
        out = root / row.state_id.replace("#", "_")
        out.mkdir(exist_ok=True)
        write(out / "path.extxyz", frozen(images, energies))
        mid = 0.5 * (fa + fb)
        q = nominal_charges(atoms.get_chemical_symbols(), row.reduced_formula)
        dipole = (q[:, None] * ((fa - mid) @ atoms.cell[:])).sum(axis=0)  # e*A
        p_nominal = np.linalg.norm(dipole) * E_CHARGE * 1e-10 / (atoms.get_volume() * 1e-30)  # C/m^2
        fu = Composition(row.reduced_formula).num_atoms
        record = {
            "state_id": row.state_id, "symbol": row.symbol, "reduced_formula": row.reduced_formula,
            "e_above_hull_meV": 1000 * row.e_above_hull, "n_atoms": n, "converged": converged,
            "minus_minus_plus_meV": 1000 * (e_minus - e_plus) / n,
            "max_polar_displacement_A": float(worst / 2),
            "barrier_meV_per_atom": 1000 * (energies.max() - e_plus) / n,
            "barrier_meV_per_fu": 1000 * (energies.max() - e_plus) / n * fu,
            "midpoint_meV_per_atom": 1000 * (energies[len(energies) // 2] - e_plus) / n,
            "p_nominal_C_m2": float(p_nominal),
            "path_meV": (1000 * (energies - e_plus) / n).round(2).tolist(),
        }
        with done_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("%s (%s): barrier %.1f meV/atom, max displacement %.2f A, P_nominal %.2f C/m2, "
                 "-P minus +P %.2f meV/atom, converged=%s", row.state_id, row.symbol,
                 record["barrier_meV_per_atom"], record["max_polar_displacement_A"], p_nominal,
                 record["minus_minus_plus_meV"], converged)


if __name__ == "__main__":
    main()
