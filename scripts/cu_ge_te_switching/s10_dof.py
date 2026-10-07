"""Stage 10: structural degrees of freedom of every distinct state.

Positional DoF is the project's convention (``data.py``, ``DoFCounter``): the sum
of ``wp.get_dof()`` over the occupied Wyckoff orbits of the pyxtal
symmetrisation (``tol=0.1``), i.e. the free x, y, z parameters. Lattice DoF is
the number of free cell parameters of the crystal system (triclinic 6 ...
cubic 1). In polar groups one coordinate per polar axis only fixes the origin,
so the positional count is one higher than the physical number there.

Writes ``dof.csv``, one row per state within 60 meV/atom of the hull.
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from common import OUT_DIR, REPO_ROOT, setup_logging

log = setup_logging()
LATTICE_DOF = {"triclinic": 6, "monoclinic": 4, "orthorhombic": 3, "tetragonal": 2,
               "trigonal": 2, "hexagonal": 2, "cubic": 1}


def state_dof(row: dict) -> dict:
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure
    from pyxtal import pyxtal

    try:
        crystal = pyxtal()
        crystal.from_seed(Structure.from_file(REPO_ROOT / row["cif"]), tol=0.1)
    except Exception as exc:  # noqa: BLE001 - record and move on
        return {**row, "ok": False, "error": str(exc)[:80]}
    orbits, positional = [], 0
    for site in crystal.atom_sites:
        dof = site.wp.get_dof()
        positional += dof
        orbits.append(f"{site.specie}:{site.wp.multiplicity}{site.wp.letter}({dof})")
    lattice = LATTICE_DOF[crystal.lattice.ltype]
    return {**row, "ok": True, "error": "", "sg_pyxtal": crystal.group.number,
            "sg_symbol_pyxtal": crystal.group.symbol, "lattice_type": crystal.lattice.ltype,
            "positional_dof": positional, "lattice_dof": lattice, "total_dof": positional + lattice,
            "n_orbits": len(crystal.atom_sites),
            "n_atoms_conv": sum(site.wp.multiplicity for site in crystal.atom_sites),
            "orbits": " ".join(orbits)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=0.06, help="eV/atom above hull")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    states = pd.read_csv(OUT_DIR / "states.csv")
    states = states[states["e_above_hull"] <= args.window]
    columns = ["state_id", "reduced_formula", "name", "cif", "e_above_hull", "symbol", "space_group",
               "n_sites", "vacuum_artefact", "polar", "volume_per_atom"]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(state_dof, states[columns].to_dict("records"), chunksize=8))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "dof.csv", index=False)
    ok = table[table["ok"]]
    log.info("%d / %d states symmetrised; median positional DoF %.0f, %d space-group disagreements",
             len(ok), len(table), ok["positional_dof"].median(),
             int((ok["sg_pyxtal"] != ok["space_group"]).sum()))


if __name__ == "__main__":
    main()
