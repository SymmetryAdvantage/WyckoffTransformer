"""Stage 0: put the published ORB references on the same footing as our structures.

The ``LeMat-Bulk-MLIP-Hull`` energies are ORB single points on the DFT
geometries (``verify_hull_energies``), while every generated structure was
fully relaxed with ORB.  The references therefore sit slightly too high, and
the campaign's "hull lowering" by de novo polymorphs may be partly that.  This
stage relaxes the 42 Cu-Ge-Te references with ORB, rebuilds the hull from both
sets and writes ``hull_entries_corrected.csv``.
"""
from __future__ import annotations

import argparse
import io
from collections import Counter

import pandas as pd
from ase.io import read as ase_read
from common import ELEMENTS, HULL_CSV, OUT_DIR, orb_calculator, relax, setup_logging
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
from wyckoff_transformer.evaluation.hull_mlips import DEFAULT_LEMAT_CIF_CSV
from wyckoff_transformer.paths import resolve_store_path

log = setup_logging()


def reference_cifs() -> pd.DataFrame:
    calc = HullEnergyCalculator("orb_conserv_inf")
    published = pd.read_parquet(calc.source, columns=["immutable_id", "nsites", "species_at_sites", "energy"])
    published = published[[set(s) <= set(ELEMENTS) for s in published["species_at_sites"]]]
    wanted = set(published["immutable_id"])
    found = []
    for chunk in pd.read_csv(resolve_store_path(DEFAULT_LEMAT_CIF_CSV), chunksize=50_000,
                             usecols=["immutable_id", "cif"]):
        found.append(chunk[chunk["immutable_id"].isin(wanted)])
    cifs = pd.concat(found)
    merged = published.merge(cifs, on="immutable_id", how="left")
    missing = merged["cif"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} reference CIFs not found in {DEFAULT_LEMAT_CIF_CSV}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = OUT_DIR / "references"
    out.mkdir(parents=True, exist_ok=True)

    refs = reference_cifs()
    log.info("%d Cu-Ge-Te references", len(refs))
    calculator = orb_calculator(args.device)
    rows = []
    for _, row in refs.iterrows():
        atoms = ase_read(io.StringIO(row["cif"]), format="cif")
        atoms.calc = calculator
        single_point = float(atoms.get_potential_energy())
        relaxed, converged = relax(atoms, calculator, fmax=0.01)
        energy = float(relaxed.get_potential_energy())
        cif_path = out / f"{row['immutable_id']}.cif"
        relaxed.write(cif_path)
        rows.append({
            "name": f"ref_{row['immutable_id']}",
            "reduced_formula": Composition(dict(Counter(row["species_at_sites"]))).reduced_formula,
            "n_atoms": len(relaxed),
            "published_energy": float(row["energy"]),
            "single_point_energy": single_point,
            "relaxed_energy": energy,
            "relaxation_gain_per_atom": (energy - float(row["energy"])) / len(relaxed),
            "converged": converged,
            "cif": str(cif_path.relative_to(OUT_DIR.parents[1])),
        })
        log.info("%s %s: relaxation lowers %.1f meV/atom (converged=%s)", row["immutable_id"],
                 rows[-1]["reduced_formula"], -1000 * rows[-1]["relaxation_gain_per_atom"], converged)
    ref_table = pd.DataFrame(rows)
    ref_table.to_csv(out / "references_relaxed.csv", index=False)

    generated = pd.read_csv(HULL_CSV)
    generated = generated[generated["source"] == "generated"]
    entries = []
    for r in ref_table.itertuples():
        comp = Composition(r.reduced_formula)
        entries.append(PDEntry(comp, r.relaxed_energy / r.n_atoms * comp.num_atoms, name=r.name))
    for r in generated.itertuples():
        comp = Composition(r.reduced_formula)
        entries.append(PDEntry(comp, r.energy_per_atom * comp.num_atoms, name=r.name))
    hull = PhaseDiagram(entries)
    meta = {r.name: r for r in generated.itertuples()}
    table = []
    for entry in entries:
        source = "reference" if entry.name.startswith("ref_") else "generated"
        cif = (ref_table.set_index("name").loc[entry.name, "cif"] if source == "reference"
               else meta[entry.name].cif)
        table.append({
            "name": entry.name,
            "reduced_formula": entry.composition.reduced_formula,
            "source": source,
            "cif": cif,
            "energy_per_atom": entry.energy_per_atom,
            "formation_energy_per_atom": hull.get_form_energy_per_atom(entry),
            "e_above_hull": hull.get_e_above_hull(entry),
        })
    table = pd.DataFrame(table).sort_values("e_above_hull")
    table.to_csv(OUT_DIR / "hull_entries_corrected.csv", index=False)
    stable = table[table["e_above_hull"] < 1e-6]
    log.info("Corrected hull vertices:\n%s", stable[["name", "reduced_formula", "source",
                                                     "formation_energy_per_atom"]].to_string())


if __name__ == "__main__":
    main()
