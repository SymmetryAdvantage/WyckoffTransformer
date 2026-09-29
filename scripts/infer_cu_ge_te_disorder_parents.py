"""Find possible occupationally disordered parents of Cu-Ge-Te campaign cells.

This is an inverse *screen*, not a claim that a parent is an experimental phase.
For each ordered relaxed cell, merge one pair of species into a dummy species,
recover the symmetry of that masked framework, and group geometrically matching
primitive parents. A child may belong to more than one proposed family. The
frequency of a child in this selected campaign is not an occupancy probability.

The run writes one membership row per child and proposed mixing pair, a family
summary, and representative partially occupied parent CIFs. It reads the
campaign's hull table and the CIF paths recorded there; it never reads a dataset
cache directly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
from scipy.spatial import cKDTree

MASKS = (("Cu", "Ge"), ("Cu", "Te"), ("Ge", "Te"))
MATCHER = StructureMatcher(
    ltol=0.15, stol=0.25, angle_tol=5, primitive_cell=False, scale=True,
)


def _read_child(task: tuple[str, str, str, float, float]):
    """Return symmetry-lifted candidate parents for one relaxed structure."""
    name, cif, formula, ehull, symprec = task
    os.environ["SPGLIB_WARNING"] = "OFF"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Issues encountered while parsing CIF:.*",
                category=UserWarning,
            )
            child = Structure.from_file(cif)
    except Exception as exc:  # noqa: BLE001 - one damaged CIF must not abort the campaign
        return name, formula, ehull, cif, [], f"CIF: {exc}"
    species = [site.specie.symbol for site in child]
    present = set(species)
    candidates = []
    for mask in MASKS:
        if not set(mask) <= present:
            continue
        numbers = [1 if symbol in mask else Element(symbol).Z for symbol in species]
        cell = (child.lattice.matrix, child.frac_coords, numbers)
        try:
            ds = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=5)
            standardized = spglib.standardize_cell(
                cell, to_primitive=True, no_idealize=False,
                symprec=symprec, angle_tolerance=5,
            )
            if ds is None or standardized is None:
                continue
            lattice, positions, parent_numbers = standardized
            parent = Structure(
                Lattice(lattice), [Element.from_Z(int(z)) for z in parent_numbers],
                positions, coords_are_cartesian=False,
            )
            site_counts = [Counter() for _ in parent]
            for symbol, primitive_index in zip(species, ds.mapping_to_primitive):
                site_counts[int(primitive_index)][symbol] += 1
            if any(not counts for counts in site_counts):
                continue
            occupancies = [
                {symbol: count / sum(counts.values()) for symbol, count in sorted(counts.items())}
                for counts in site_counts
            ]
            parent_signature = (
                "-".join(mask), int(ds.number), len(parent),
                tuple(sorted(Counter(int(z) for z in parent_numbers).items())),
            )
            candidates.append((parent_signature, parent, occupancies))
        except (ValueError, TypeError, spglib.SpglibError):
            continue
    return name, formula, ehull, cif, candidates, None


def _metric(parent: Structure) -> np.ndarray:
    """A cheap necessary geometry screen before StructureMatcher."""
    cube_length = parent.volume ** (1 / 3)
    axes = np.log(np.sort(parent.lattice.abc) / cube_length)
    distances = parent.distance_matrix[np.triu_indices(len(parent), k=1)]
    if len(distances):
        radial = np.quantile(distances / cube_length, [0.1, 0.5, 0.9])
    else:
        radial = np.zeros(3)
    return np.r_[axes, np.log(parent.volume / len(parent)), radial]


def _family_for(parent: Structure, metric: np.ndarray,
                candidate_families: list[dict], index: dict,
                max_volume_ratio: float):
    nearby = []
    if index["tree"] is not None:
        nearby.extend(index["tree"].query_ball_point(metric, r=0.30, p=np.inf))
    nearby.extend(range(index["indexed"], len(candidate_families)))
    nearby = sorted(set(nearby))
    for family_index in nearby:
        family = candidate_families[family_index]
        if np.max(np.abs(metric - family["metric"])) > 0.30:
            continue
        representative = family["parent"]
        ratio = parent.volume / representative.volume
        if not 1 / max_volume_ratio <= ratio <= max_volume_ratio:
            continue
        if MATCHER.fit(representative, parent):
            return family
    return None


def _cluster_signature(job: tuple[tuple, list[tuple], float]):
    """Cluster one mask/symmetry/site-count class in an independent process."""
    signature, items, max_volume_ratio = job
    candidate_families: list[dict] = []
    index = {"tree": None, "indexed": 0}
    memberships = []
    for name, formula, ehull, parent, occupancies in items:
        metric = _metric(parent)
        family = _family_for(parent, metric, candidate_families, index,
                             max_volume_ratio)
        if family is None:
            family = {
                "local_id": len(candidate_families),
                "signature": signature, "parent": parent, "metric": metric,
                "representative": name, "formulas": set(), "energies": [],
                "members": [], "occupancies": occupancies,
            }
            candidate_families.append(family)
            if len(candidate_families) - index["indexed"] >= 64:
                index["tree"] = cKDTree(np.array(
                    [item["metric"] for item in candidate_families]
                ))
                index["indexed"] = len(candidate_families)
        family["formulas"].add(formula)
        family["energies"].append(ehull)
        family["members"].append(name)
        memberships.append({
            "name": name, "reduced_formula": formula, "e_above_hull": ehull,
            "mask": signature[0], "local_id": family["local_id"],
            "parent_spacegroup": signature[1],
            "parent_primitive_sites": signature[2],
        })
    return signature, candidate_families, memberships


def analyse(rows: list[tuple[str, str, str, float, float]], workers: int,
            max_volume_ratio: float = 1.15):
    """Group children by explicitly matched symmetry-lifted parent geometry."""
    started = time.monotonic()
    if workers == 1:
        results = list(map(_read_child, rows))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = []
            for result in pool.map(_read_child, rows, chunksize=16):
                results.append(result)
                if len(results) % 1000 == 0:
                    print(f"Parsed {len(results)}/{len(rows)} CIFs in "
                          f"{time.monotonic() - started:.0f}s", flush=True)
    print(f"Parsed {len(results)} CIFs in {time.monotonic() - started:.0f}s; "
          "matching parent frameworks", flush=True)
    results.sort(key=lambda result: (result[2], result[0]))
    by_signature: dict[tuple, list[tuple]] = defaultdict(list)
    failures = []
    for name, formula, ehull, cif, candidates, error in results:
        if error:
            failures.append({"name": name, "cif": cif, "error": error})
        for signature, parent, occupancies in candidates:
            by_signature[signature].append(
                (name, formula, ehull, parent, occupancies)
            )
    small_jobs = [(signature, items, max_volume_ratio)
                  for signature, items in by_signature.items() if len(items) <= 2]
    large_jobs = [(signature, items, max_volume_ratio)
                  for signature, items in by_signature.items() if len(items) > 2]
    clustered = list(map(_cluster_signature, small_jobs))
    print(f"Matched {len(small_jobs)} small classes locally; "
          f"{len(large_jobs)} larger classes remain", flush=True)
    if workers == 1:
        clustered.extend(map(_cluster_signature, large_jobs))
    elif large_jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_cluster_signature, job) for job in large_jobs]
            for future in as_completed(futures):
                clustered.append(future.result())
                if (len(clustered) - len(small_jobs)) % 100 == 0:
                    print(f"Matched {len(clustered) - len(small_jobs)}/"
                          f"{len(large_jobs)} larger classes "
                          f"in {time.monotonic() - started:.0f}s", flush=True)
    families = []
    memberships = []
    for signature, group, member_rows in sorted(clustered, key=lambda item: item[0]):
        id_map = {}
        for family in group:
            global_id = f"P{len(families) + 1:05d}"
            id_map[family["local_id"]] = global_id
            family["id"] = global_id
            families.append(family)
        for member in member_rows:
            member["family_id"] = id_map[member.pop("local_id")]
            memberships.append(member)
    return families, memberships, failures


def _write_parent_cif(family: dict, destination: Path):
    """Symmetrize the representative's stoichiometry over parent Wyckoff orbits."""
    parent = family["parent"].copy()
    ds = spglib.get_symmetry_dataset(
        (parent.lattice.matrix, parent.frac_coords, parent.atomic_numbers),
        symprec=0.1, angle_tolerance=5,
    )
    if ds is None:
        raise ValueError(f"Cannot symmetrize inferred parent {family['id']}")
    orbits = defaultdict(list)
    for index, orbit in enumerate(ds.equivalent_atoms):
        orbits[int(orbit)].append(index)
    for sites in orbits.values():
        counts = defaultdict(float)
        for index in sites:
            for symbol, fraction in family["occupancies"][index].items():
                counts[symbol] += fraction / len(sites)
        for index in sites:
            parent.replace(index, dict(counts))
    CifWriter(parent, symprec=0.1).write_file(str(destination))
    match = re.search(r"_symmetry_Int_Tables_number\s+(\d+)", destination.read_text())
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=Path, default=Path("artifacts/cu_ge_te_hull/cu_ge_te_hull_entries.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cu_ge_te_hull/disorder_parents"))
    parser.add_argument("--max-ehull", type=float, default=0.1)
    parser.add_argument("--symprec", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="For smoke runs only")
    args = parser.parse_args()
    entries = pd.read_csv(args.entries)
    selected = entries.loc[
        (entries.source == "generated") & (~entries.is_stable)
        & (entries.e_above_hull <= args.max_ehull),
        ["name", "cif", "reduced_formula", "e_above_hull"],
    ].copy()
    if args.limit is not None:
        selected = selected.head(args.limit)
    tasks = [(*row, args.symprec) for row in selected.itertuples(index=False, name=None)]
    families, memberships, failures = analyse(tasks, args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(memberships).to_csv(args.output_dir / "memberships.csv", index=False)
    summary = []
    for family in families:
        rendered_spacegroup = None
        if len(family["members"]) >= 2:
            rendered_spacegroup = _write_parent_cif(
                family, args.output_dir / f"{family['id']}.cif"
            )
        summary.append({
            "family_id": family["id"], "mask": family["signature"][0],
            "parent_spacegroup": family["signature"][1],
            "representative_cif_spacegroup": rendered_spacegroup,
            "parent_primitive_sites": family["signature"][2],
            "representative": family["representative"],
            "n_structures": len(family["members"]),
            "n_formulas": len(family["formulas"]),
            "formulas": ";".join(sorted(family["formulas"])),
            "min_e_above_hull": min(family["energies"]),
            "max_e_above_hull": max(family["energies"]),
        })
    pd.DataFrame(summary).sort_values(
        ["n_formulas", "n_structures"], ascending=False,
    ).to_csv(args.output_dir / "families.csv", index=False)
    metadata = {
        "input": str(args.entries), "selected_structures": len(selected),
        "assigned_structures": len({row["name"] for row in memberships}),
        "candidate_memberships": len(memberships),
        "max_ehull": args.max_ehull, "symprec_angstrom": args.symprec,
        "matcher_ltol": 0.15, "matcher_stol": 0.25, "matcher_angle_tol": 5,
        "max_volume_ratio": 1.15, "candidate_families": len(families),
        "families_with_multiple_formulas": sum(len(f["formulas"]) > 1 for f in families),
        "families_with_at_least_three_formulas_and_ten_structures": sum(
            len(f["formulas"]) >= 3 and len(f["members"]) >= 10 for f in families
        ),
        "cif_errors": failures,
        "interpretation": "Candidate substitutional parent frameworks only; no experimental disorder, occupancy frequency, or material count is inferred.",
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "cif_errors"}, indent=2))


if __name__ == "__main__":
    main()
