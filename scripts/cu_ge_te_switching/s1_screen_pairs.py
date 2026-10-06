"""Stage 1: collapse the hull entries into distinct states and pair them.

For every composition with at least two low-energy entries:

* group duplicate structures (default ``StructureMatcher``) and keep the lowest
  energy of each group as one *state*;
* describe each state by space group, polarity, volume per atom and the widest
  empty slab in the cell (a cell with a > 4 A slab is a stack of layers held
  apart by nothing, the artefact suspected in the expanded GeTe polymorphs);
* pair states of the same composition and ask whether a loose matcher can map
  one onto the other.  A match means the two are related by atomic displacements
  and a strain of a few percent -- the precondition for a diffusionless switch --
  and its RMS displacement is a first proxy for how far the atoms must move.

Outputs ``states.csv`` and ``pairs.csv``.
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from common import OUT_DIR, REPO_ROOT, is_polar, setup_logging
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

log = setup_logging()
VACUUM_SLAB_A = 4.0  # real Cu-Te layer gaps reach ~3.6 A


def widest_empty_slab(structure: Structure) -> float:
    """Widest atom-free slab, in angstrom, parallel to any pair of cell vectors."""
    lattice = structure.lattice
    widest = 0.0
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        normal = np.cross(lattice.matrix[others[0]], lattice.matrix[others[1]])
        spacing = lattice.volume / np.linalg.norm(normal)
        frac = np.sort(structure.frac_coords[:, axis] % 1.0)
        gaps = np.diff(np.concatenate([frac, [frac[0] + 1.0]]))
        widest = max(widest, float(gaps.max() * spacing))
    return widest


def describe_formula(task):
    formula, rows, max_states = task
    warnings.filterwarnings("ignore")
    structures = [Structure.from_file(REPO_ROOT / cif) for cif in rows["cif"]]
    groups = StructureMatcher().group_structures(structures)
    index_of = {id(s): i for i, s in enumerate(structures)}
    states = []
    for group in groups:
        members = sorted(index_of[id(s)] for s in group)
        best = min(members, key=lambda i: rows["e_above_hull"].iloc[i])
        structure = structures[best]
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            number, symbol = sga.get_space_group_number(), sga.get_space_group_symbol()
        except Exception:  # noqa: BLE001 - spglib failures on odd cells
            number, symbol = 1, "P1"
        states.append({
            "reduced_formula": formula,
            "name": rows["name"].iloc[best],
            "source": rows["source"].iloc[best],
            "cif": rows["cif"].iloc[best],
            "e_above_hull": rows["e_above_hull"].iloc[best],
            "energy_per_atom": rows["energy_per_atom"].iloc[best],
            "n_duplicates": len(members),
            "space_group": number,
            "symbol": symbol,
            "polar": is_polar(number),
            "n_sites": len(structure),
            "volume_per_atom": structure.volume / len(structure),
            "widest_empty_slab": widest_empty_slab(structure),
        })
    states = pd.DataFrame(states).sort_values("e_above_hull").reset_index(drop=True)
    states["state_rank"] = np.arange(len(states))
    states["state_id"] = [f"{formula}#{i}" for i in states["state_rank"]]
    states["vacuum_artefact"] = states["widest_empty_slab"] > VACUUM_SLAB_A

    loose = StructureMatcher(ltol=0.3, stol=0.6, angle_tol=15, primitive_cell=True,
                             scale=True, attempt_supercell=True)
    lookup = {row.state_id: Structure.from_file(REPO_ROOT / row.cif) for row in states.head(max_states).itertuples()}
    pairs = []
    for a, b in combinations(states.head(max_states).itertuples(), 2):
        sa, sb = lookup[a.state_id], lookup[b.state_id]
        rms = None
        try:
            fit = loose.get_rms_dist(sa, sb)
        except Exception:  # noqa: BLE001 - supercell search can blow up
            fit = None
        if fit is not None:
            rms = fit[0]
        pairs.append({
            "reduced_formula": formula,
            "state_a": a.state_id, "state_b": b.state_id,
            "sg_a": a.symbol, "sg_b": b.symbol,
            "e_a_meV": 1000 * a.e_above_hull, "e_b_meV": 1000 * b.e_above_hull,
            "delta_e_meV": 1000 * (b.e_above_hull - a.e_above_hull),
            "delta_v_rel": b.volume_per_atom / a.volume_per_atom - 1.0,
            "delta_v_A3": b.volume_per_atom - a.volume_per_atom,
            "polar_a": a.polar, "polar_b": b.polar,
            "displacive_match": rms is not None,
            "rms_displacement": rms,
            "any_vacuum_artefact": bool(a.vacuum_artefact or b.vacuum_artefact),
        })
    return states, pd.DataFrame(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=0.06, help="eV/atom above hull")
    parser.add_argument("--max-states", type=int, default=25,
                        help="Distinct states kept per composition for pairing")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    entries = pd.read_csv(OUT_DIR / "hull_entries_corrected.csv")
    entries = entries[entries["e_above_hull"] <= args.window]
    counts = entries["reduced_formula"].value_counts()
    tasks = []
    for formula, rows in entries.groupby("reduced_formula"):
        if counts[formula] < 2:
            continue
        tasks.append((formula, rows.sort_values("e_above_hull").head(200).reset_index(drop=True), args.max_states))
    log.info("%d entries in %d compositions with >= 2 entries", len(entries), len(tasks))

    all_states, all_pairs = [], []
    tasks.sort(key=lambda t: -len(t[1]))  # longest first for load balance
    with ProcessPoolExecutor(args.workers) as pool:
        for states, pairs in pool.map(describe_formula, tasks):
            all_states.append(states)
            all_pairs.append(pairs)
    states = pd.concat(all_states, ignore_index=True)
    pairs = pd.concat([p for p in all_pairs if len(p)], ignore_index=True)
    states.to_csv(OUT_DIR / "states.csv", index=False)
    pairs.to_csv(OUT_DIR / "pairs.csv", index=False)
    log.info("%d distinct states, %d pairs, %d displacive matches, %d states flagged as vacuum stacks",
             len(states), len(pairs), int(pairs["displacive_match"].sum()),
             int(states["vacuum_artefact"].sum()))


if __name__ == "__main__":
    main()
