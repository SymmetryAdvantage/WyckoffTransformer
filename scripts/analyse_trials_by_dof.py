"""What a second PyXtal trial buys, resolved by the gene's positional DoF.

A trial is a random draw of a gene's free internal coordinates, so what a second
one can buy is bounded by how many there are.  This pairs two measurements over
the same 2500 genes of run `upi73i4k`:

* per-trial energies from `cryspr_trial_energies.csv` (3 trials, written by
  `scripts/analyse_cryspr_trial_spread.py` from the BFGS logfiles), which give
  the per-gene penalty of keeping fewer trials;
* ORB `e_above_hull` from the protocol run of the same genes (1 trial), which is
  the quantity the protocol actually thresholds.

`e_hull` is affine in the total energy at fixed composition, so a per-atom energy
penalty maps onto it one-to-one:

    e_hull(best-of-k) = e_hull(best-of-1) - (E_1 - E_bestk) / n_atoms

Two caveats, both in `docs/de_novo_ranking_protocol.md`: the penalties are MACE's
while `e_hull` is ORB's, and they were measured *before* the rattle stage, which
does part of the same job.  Rerun this over a multi-trial run with the rattle on
-- `structures.csv` now carries `dof_positional` and `n_trials`, and every trial
carries `rattle.json` -- to settle how much the two overlap.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyxtal.symmetry import Group
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
GENES = REPO / "generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz"
TRIALS = REPO / "generated/upi73i4k/cryspr_trial_energies.csv"
PROTOCOL = REPO / "generated/upi73i4k/protocol/structures.csv"

#: eV/atom.  The protocol's development readout.
METASTABLE = 0.1
#: |E/atom| above this is a collapsed cell, which the MLIP reports as ~-1e15.
NONPHYSICAL = 50.0

BINS = [-0.5, 0.5, 2.5, 5.5, 10.5, np.inf]
LABELS = ["0", "1-2", "3-5", "6-10", ">10"]

#: Candidate schedules, as {bin label -> trials}, spelled out over these bins.
#: `0:1,2:2,*:3` is `DEFAULT_TRIAL_SCHEDULE`.
SCHEDULES = {
    "flat 1": {"0": 1, "1-2": 1, "3-5": 1, "6-10": 1, ">10": 1},
    "0:1,*:2": {"0": 1, "1-2": 2, "3-5": 2, "6-10": 2, ">10": 2},
    "flat 2": {"0": 2, "1-2": 2, "3-5": 2, "6-10": 2, ">10": 2},
    "0:1,2:2,*:3 (default)": {"0": 1, "1-2": 2, "3-5": 3, "6-10": 3, ">10": 3},
    "flat 3": {"0": 3, "1-2": 3, "3-5": 3, "6-10": 3, ">10": 3},
}

#: Relaxation-equivalents per trial.  Four stages run -- the fix-cell warm-up,
#: the symmetric cell+positions stage, the unconstrained one, and the rattle --
#: but the unconstrained stage takes zero optimiser steps in 78.2% of trials
#: (docs/cryspr_reconstruction_report.md), so it costs about a fifth of a stage.
STAGES_PER_TRIAL = 3.2

_groups: dict[int, Group] = {}


def positional_dof(gene: dict) -> int:
    """Free internal coordinates PyXtal has to draw for this gene."""
    number = int(gene["group"])
    group = _groups.setdefault(number, Group(number))
    return sum(
        int(group[str(site)[-1]].get_dof())
        for species_sites in gene["sites"]
        for site in species_sites
    )


def n_per_arm(p: float, relative: float = 0.2, alpha: float = 0.05,
              power: float = 0.8) -> float:
    """Genes per arm to detect a *relative* change of `relative` in `p`."""
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    p1, p2 = p, p * (1 + relative)
    mean = (p1 + p2) / 2
    return (
        z_a * np.sqrt(2 * mean * (1 - mean))
        + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2 / (p2 - p1) ** 2


def load() -> pd.DataFrame:
    """One row per gene: its DoF bin and its e_hull at 1, 2 and 3 trials."""
    with gzip.open(GENES, "rt") as handle:
        genes = json.load(handle)
    dof = pd.Series([positional_dof(gene) for gene in genes], name="dof_pos")

    trials = pd.read_csv(TRIALS)
    # The unconstrained stage is the last one this run logged, and the one the
    # rattle would be perturbed away from; the symmetric stage stands in where
    # it is missing.  The two differ by more than 1 meV/atom in 0.4% of trials.
    energy = trials["E_3_no-sym"].where(trials["E_3_no-sym"].notna(), trials["E_2_sym"])
    trials = trials.assign(e_per_atom=energy / trials["n_atoms"])
    trials = trials[trials["e_per_atom"].abs() < NONPHYSICAL]
    wide = trials.pivot_table(
        index="gene", columns="trial", values="e_per_atom"
    ).dropna()  # genes where every trial survived

    protocol = pd.read_csv(PROTOCOL, index_col="index")
    frame = pd.DataFrame({"e_hull1": protocol["e_above_hull"]}).join(dof)
    frame["gain2"] = (wide[0] - wide[[0, 1]].min(axis=1)).reindex(frame.index)
    frame["gain3"] = (wide[0] - wide.min(axis=1)).reindex(frame.index)
    frame = frame.dropna(subset=["e_hull1", "gain3"])
    frame["e_hull2"] = frame["e_hull1"] - frame["gain2"]
    frame["e_hull3"] = frame["e_hull1"] - frame["gain3"]
    frame["bin"] = pd.cut(frame["dof_pos"], BINS, labels=LABELS)
    return frame


def main() -> None:
    frame = load()
    groups = {label: chunk for label, chunk in frame.groupby("bin", observed=True)}
    weight = {label: len(chunk) / len(frame) for label, chunk in groups.items()}
    p_of_t = {
        label: {t: (chunk[f"e_hull{t}"] <= METASTABLE).mean() for t in (1, 2, 3)}
        for label, chunk in groups.items()
    }

    print(f"{len(frame)} genes with all three trials surviving\n")
    print("positional DoF  share  p@1     p@2     p@3     agree<1meV  shortfall@1")
    for label in LABELS:
        p, chunk = p_of_t[label], groups[label]
        print(f"{label:>14}  {weight[label]:.3f}  {p[1]:.3f}   {p[2]:.3f}   "
              f"{p[3]:.3f}   {(chunk['gain3'] < 0.001).mean():.3f}       "
              f"{p[3] - p[1]:+.3f}")

    print("\nschedule             p       trials/gene  n/arm  total work")
    for name, schedule in SCHEDULES.items():
        p = sum(weight[d] * p_of_t[d][t] for d, t in schedule.items())
        per_gene = sum(weight[d] * t for d, t in schedule.items())
        n = n_per_arm(p)
        print(f"{name:<20} {p:.4f}  {per_gene:.2f}         {n:6.0f} "
              f"{n * per_gene * STAGES_PER_TRIAL:10.0f}")


if __name__ == "__main__":
    main()
