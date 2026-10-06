"""Stage 8: enthalpy under hydrostatic pressure -- which states pressure switches.

Each candidate is relaxed (cell and positions) at a series of pressures with
ORB, and H = E + PV per atom is compared within a composition.  A crossing is a
pressure-driven transition between two states of the same composition; the
space group after relaxation is recorded because a state can also transform
continuously under load (GeTe's rhombohedral distortion is expected to vanish).

Runs on CPU in parallel workers; ``pressure.csv`` holds one row per
(state, pressure).
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from ase import units
from ase.io import read
from common import OUT_DIR, REPO_ROOT, setup_logging

log = setup_logging()
PRESSURES_GPA = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
FORMULAS = ("GeTe", "Cu2GeTe3", "Cu3GeTe4", "Cu7GeTe8", "CuGeTe2")


def run_state(task):
    state_id, cif, threads = task
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch

    torch.set_num_threads(threads)
    warnings.filterwarnings("ignore")
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from common import orb_calculator
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    calc = orb_calculator("cpu")
    atoms = read(cif)
    rows = []
    for p in PRESSURES_GPA:
        atoms.calc = calc
        filt = FrechetCellFilter(atoms, scalar_pressure=p * units.GPa)
        converged = bool(FIRE(filt, logfile=None).run(fmax=0.005, steps=1500))
        e = atoms.get_potential_energy() / len(atoms)
        v = atoms.get_volume() / len(atoms)
        try:
            sg = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms), symprec=0.1).get_space_group_symbol()
        except Exception:  # noqa: BLE001
            sg = "?"
        rows.append({"state_id": state_id, "P_GPa": p, "energy_per_atom": e, "volume_per_atom": v,
                     "enthalpy_per_atom": e + p * units.GPa * v, "space_group_after": sg,
                     "converged": converged})
        atoms = atoms.copy()  # continue the pressure ramp from this geometry
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--window", type=float, default=0.06)
    parser.add_argument("--per-formula", type=int, default=8)
    args = parser.parse_args()

    states = pd.read_csv(OUT_DIR / "states.csv")
    chosen = states[states["reduced_formula"].isin(FORMULAS) & ~states["vacuum_artefact"]
                    & (states["e_above_hull"] <= args.window) & (states["n_sites"] <= 48)]
    chosen = chosen.groupby("reduced_formula").head(args.per_formula)
    tasks = [(r.state_id, str(REPO_ROOT / r.cif), args.threads) for r in chosen.itertuples()]
    rocksalt = OUT_DIR / "gete_wells" / "fm3m_orb.cif"
    if rocksalt.exists():
        tasks.append(("GeTe#rocksalt", str(rocksalt), args.threads))
    log.info("%d states x %d pressures", len(tasks), len(PRESSURES_GPA))
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(run_state, tasks):
            rows.extend(result)
            log.info("%s done", result[0]["state_id"])
    table = pd.DataFrame(rows)
    table["reduced_formula"] = table["state_id"].str.split("#").str[0]
    ref = table.groupby(["reduced_formula", "P_GPa"])["enthalpy_per_atom"].transform("min")
    table["dH_meV"] = 1000 * (table["enthalpy_per_atom"] - ref)
    table.to_csv(OUT_DIR / "pressure.csv", index=False)
    pivot = table.pivot_table(index="state_id", columns="P_GPa", values="dH_meV")
    log.info("enthalpy above the lowest state of the composition, meV/atom:\n%s", pivot.round(1).to_string())


if __name__ == "__main__":
    main()
