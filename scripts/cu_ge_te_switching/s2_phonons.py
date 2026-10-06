"""Stage 2: harmonic phonons with ORB for the selected states and the hull vertices.

For each structure: a tight relaxation (cell and positions), finite-displacement
force constants with phonopy, and on a q-mesh

* the lowest frequency -- an imaginary mode means the structure is a saddle,
  not a state that can hold;
* the harmonic vibrational free energy F_vib(T) and entropy S(T), per atom.

``phonons.csv`` holds one row per structure; ``thermal.csv`` the F_vib(T) curves.
The free energy of a state is E0 + F_vib(T) at the 0 K volume (harmonic, no
thermal expansion).
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from common import OUT_DIR, REPO_ROOT, orb_calculator, relax, setup_logging
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

log = setup_logging()
EV_PER_KJMOL = 1.0 / 96.485332
TEMPERATURES = np.arange(0, 1201, 25)
HULL_VERTEX_FORMULAS = ("Cu", "Ge", "Te", "GeTe", "CuTe", "Cu3Te2", "Cu3GeTe4", "Cu5Ge", "Cu3Ge", "Cu2Te")


def supercell_matrix(atoms: Atoms, min_length: float) -> np.ndarray:
    """Diagonal supercell with every perpendicular width >= *min_length*."""
    cell = atoms.cell
    widths = cell.volume / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    return np.diag(np.maximum(1, np.ceil(min_length / widths)).astype(int))


def phonons(atoms: Atoms, calculator, min_length: float, mesh_density: float):
    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:],
                        scaled_positions=atoms.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=supercell_matrix(atoms, min_length),
                 primitive_matrix="auto", symprec=1e-3)
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        sc_atoms = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        sc_atoms.calc = calculator
        f = sc_atoms.get_forces()
        forces.append(f - f.mean(axis=0))
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    ph.run_mesh(mesh_density, with_eigenvectors=False, is_gamma_center=True)
    freqs = ph.get_mesh_dict()["frequencies"]
    weights = ph.get_mesh_dict()["weights"]
    acoustic_cut = 3  # three acoustic modes at Gamma are zero by construction
    flat = np.sort(freqs.ravel())
    n_imag = int((flat < -0.25).sum())
    ph.run_thermal_properties(temperatures=TEMPERATURES)
    tp = ph.get_thermal_properties_dict()
    n_prim = len(ph.primitive)
    return {
        "n_displacements": len(forces),
        "supercell_atoms": len(ph.supercell),
        "min_freq_THz": float(flat[0]),
        "n_imag_modes_mesh": n_imag,
        "imag_fraction": n_imag / flat.size,
        "mean_freq_THz": float(np.average(freqs.mean(axis=1), weights=weights)),
        "f_vib": (tp["free_energy"] * EV_PER_KJMOL / n_prim).tolist(),
        "s_vib_kB": (tp["entropy"] / 8.314462618 / n_prim).tolist(),
        "acoustic_cut": acoustic_cut,
    }


def select_states(states: pd.DataFrame, entries: pd.DataFrame, window: float, per_formula: int):
    chosen = states[(states["e_above_hull"] <= window) & ~states["vacuum_artefact"]]
    ternary = chosen[chosen["reduced_formula"].str.contains("Cu") & chosen["reduced_formula"].str.contains("Ge")
                     & chosen["reduced_formula"].str.contains("Te")]
    ternary = ternary.groupby("reduced_formula").head(per_formula)
    binaries = chosen[chosen["reduced_formula"].isin(HULL_VERTEX_FORMULAS)].groupby("reduced_formula").head(per_formula)
    picked = pd.concat([ternary, binaries])[["state_id", "reduced_formula", "name", "cif", "e_above_hull"]]
    # hull vertices (elements included) are needed for the finite-temperature hull
    vertices = entries[entries["e_above_hull"] < 1e-6]
    extra = vertices[~vertices["name"].isin(picked["name"])].copy()
    extra["state_id"] = extra["reduced_formula"] + "#vertex"
    picked = pd.concat([picked, extra[["state_id", "reduced_formula", "name", "cif", "e_above_hull"]]])
    return picked.drop_duplicates("name").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=0.05)
    parser.add_argument("--per-formula", type=int, default=6)
    parser.add_argument("--min-length", type=float, default=11.0)
    parser.add_argument("--mesh", type=float, default=40.0)
    parser.add_argument("--max-atoms", type=int, default=64, help="skip larger unit cells")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", help="state_ids to run")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    states = pd.read_csv(OUT_DIR / "states.csv")
    entries = pd.read_csv(OUT_DIR / "hull_entries_corrected.csv")
    picked = select_states(states, entries, args.window, args.per_formula)
    if args.only:
        picked = picked[picked["state_id"].isin(args.only)]
    log.info("%d structures selected", len(picked))
    out = OUT_DIR / "phonons"
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "phonons.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(line)["name"] for line in done_path.read_text().splitlines()}

    calculator = orb_calculator(args.device)
    for row in picked.itertuples():
        if row.name in done:
            continue
        atoms = read(REPO_ROOT / row.cif)
        if len(atoms) > args.max_atoms:
            log.info("skip %s (%d atoms)", row.state_id, len(atoms))
            continue
        relaxed, converged = relax(atoms, calculator, fmax=1e-3, steps=3000)
        e0 = relaxed.get_potential_energy() / len(relaxed)
        relaxed.write(out / f"{row.name}.cif")
        try:
            result = phonons(relaxed, calculator, args.min_length, args.mesh)
        except Exception as exc:  # noqa: BLE001 - record and move on
            log.warning("%s failed: %s", row.state_id, exc)
            continue
        record = {"state_id": row.state_id, "name": row.name, "reduced_formula": row.reduced_formula,
                  "e_above_hull_input": row.e_above_hull, "e0_per_atom": e0,
                  "volume_per_atom": relaxed.get_volume() / len(relaxed),
                  "relax_converged": converged, **result}
        with done_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("%s: min freq %.2f THz, %d imaginary, mean %.2f THz", row.state_id,
                 result["min_freq_THz"], result["n_imag_modes_mesh"], result["mean_freq_THz"])


if __name__ == "__main__":
    main()
