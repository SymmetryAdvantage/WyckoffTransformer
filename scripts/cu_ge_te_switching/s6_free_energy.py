"""Stage 6: harmonic free energies -> the hull at temperature and state crossovers.

G(T) = E0 + F_vib(T) per atom for every structure with phonons and no
imaginary modes (``phonons/phonons.jsonl``).  From these:

* ``hull_vs_T.csv``: each structure's distance to the hull built from the same
  set at each temperature -- which metastable states the vibrational entropy
  stabilises, and the hull vertices that drop off;
* ``crossovers.csv``: for every two states of one composition, G_B - G_A at 0,
  300, 600 and 900 K and the temperature where the order flips, if it does
  below 1200 K.

Harmonic and at the 0 K volume: no thermal expansion, no anharmonicity, no
configurational entropy.  GeTe's ferroelectric transition is anharmonic and is
out of reach here; that is what the heating MD is for.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from common import OUT_DIR, setup_logging
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from s2_phonons import TEMPERATURES

log = setup_logging()
IMAG_TOLERANCE = 0.002  # fraction of mesh modes allowed below -0.25 THz (numerical noise)


def load() -> pd.DataFrame:
    rows = [json.loads(line) for line in (OUT_DIR / "phonons" / "phonons.jsonl").read_text().splitlines()]
    frame = pd.DataFrame(rows)
    frame["stable_phonons"] = frame["imag_fraction"] <= IMAG_TOLERANCE
    return frame


def hull_at(frame: pd.DataFrame, t_index: int) -> pd.Series:
    entries = []
    for row in frame.itertuples():
        comp = Composition(row.reduced_formula)
        g = row.e0_per_atom + row.f_vib[t_index]
        entries.append(PDEntry(comp, g * comp.num_atoms, name=row.name))
    hull = PhaseDiagram(entries)
    return pd.Series({e.name: hull.get_e_above_hull(e) for e in entries})


def main() -> None:
    frame = load()
    log.info("%d structures with phonons, %d dynamically stable", len(frame), frame["stable_phonons"].sum())
    stable = frame[frame["stable_phonons"]].reset_index(drop=True)
    elements = {c for f in stable["reduced_formula"] for c in Composition(f).elements}
    log.info("elements present: %s", sorted(str(e) for e in elements))

    hull_rows = []
    for ti, t in enumerate(TEMPERATURES):
        if t % 100:
            continue
        e_hull = hull_at(stable, ti)
        for name, value in e_hull.items():
            hull_rows.append({"name": name, "T": int(t), "e_above_hull_meV": 1000 * value})
    hull_vs_t = pd.DataFrame(hull_rows).merge(stable[["name", "state_id", "reduced_formula"]], on="name")
    hull_vs_t.to_csv(OUT_DIR / "hull_vs_T.csv", index=False)

    cross = []
    for formula, group in stable.groupby("reduced_formula"):
        for a, b in combinations(group.itertuples(), 2):
            dg = 1000 * ((b.e0_per_atom + np.array(b.f_vib)) - (a.e0_per_atom + np.array(a.f_vib)))
            sign = np.sign(dg)
            flips = np.where(sign[1:] != sign[:-1])[0]
            t_cross = float(TEMPERATURES[flips[0] + 1]) if len(flips) else None
            row = {"reduced_formula": formula, "state_a": a.state_id, "state_b": b.state_id,
                   "name_a": a.name, "name_b": b.name,
                   "ds_vib_kB_300K": b.s_vib_kB[12] - a.s_vib_kB[12], "T_cross": t_cross}
            for t in (0, 300, 600, 900):
                row[f"dG_{t}K_meV"] = float(dg[list(TEMPERATURES).index(t)])
            cross.append(row)
    cross = pd.DataFrame(cross)
    cross.to_csv(OUT_DIR / "crossovers.csv", index=False)
    log.info("%d same-composition pairs, %d cross below 1200 K", len(cross), cross["T_cross"].notna().sum())


if __name__ == "__main__":
    main()
