"""How much do CrySPR's relaxations disagree, across trials and across stages?

Each Wyckoff gene is relaxed n_trials times from independent PyXtal samplings, and
each trial runs three stages (fix-cell, symmetry-constrained cell+positions, then
unconstrained). Only the lowest-energy trial's final structure is kept, so the
spread across trials is what the extra trials buy, and the stage-3 drop is what
releasing the symmetry constraint buys.

Energies come from the BFGS logfiles' last row -- the CIFs carry no energy.
"""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kna/WyckoffTransformer/generated/upi73i4k/cryspr")
GENES = Path("/home/kna/WyckoffTransformer/generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz")

STAGE_LOG = {  # stage label -> logfile suffix written by stepwise_relax
    "1_fix-cell": "_fix-cell_relax.log",
    "2_sym": "_sym_cell+positions_relax.log",
    "3_no-sym": "_no-sym_cell+positions_relax.log",
}
_ROW = re.compile(r"^\s*\w+:\s+\d+\s+[\d:]+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def final_energy(log_path: Path):
    """Last (energy, fmax) row of a BFGS logfile, or None if it never stepped."""
    last = None
    try:
        for line in log_path.read_text().splitlines():
            m = _ROW.match(line)
            if m:
                last = (float(m.group(1)), float(m.group(2)))
    except OSError:
        return None
    return last


def main() -> None:
    genes = json.load(gzip.open(GENES, "rt"))
    rows = []
    for gene_dir in sorted((p for p in ROOT.iterdir() if p.is_dir() and p.name.isdigit()),
                           key=lambda p: int(p.name)):
        gid = int(gene_dir.name)
        n_atoms = sum(genes[gid]["numIons"])
        for trial_dir in sorted(gene_dir.glob("trial-[0-9]*")):
            record = {"gene": gid, "trial": int(trial_dir.name.split("-")[1]), "n_atoms": n_atoms}
            for stage, suffix in STAGE_LOG.items():
                matches = list(trial_dir.glob(f"*{suffix}"))
                got = final_energy(matches[0]) if matches else None
                record[f"E_{stage}"] = got[0] if got else np.nan
                record[f"fmax_{stage}"] = got[1] if got else np.nan
            rows.append(record)
    df = pd.DataFrame(rows)
    df.to_csv("/tmp/claude-1000/-home-kna-WyckoffTransformer/4b6c59a2-482a-4508-8554-9d9f94c76d94/scratchpad/trial_energies.csv", index=False)

    df["E_final_pa"] = df["E_3_no-sym"] / df.n_atoms
    done = df.dropna(subset=["E_final_pa"])
    print(f"trials with a final energy: {len(done)} over {done.gene.nunique()} genes\n")

    # --- spread across trials of the same gene
    g = done.groupby("gene")["E_final_pa"]
    spread = (g.max() - g.min()).dropna()
    multi = spread[g.count() > 1]
    print(f"genes with >1 successful trial: {len(multi)}")
    print("spread (max-min) of final energy per atom, eV/atom:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"   p{int(q*100):<3d} {multi.quantile(q):.4f}")
    print(f"   mean {multi.mean():.4f}   max {multi.max():.4f}")
    print(f"   trials agree within 1 meV/atom: {(multi < 0.001).mean():.1%}")
    print(f"   trials agree within 10 meV/atom: {(multi < 0.01).mean():.1%}")
    print(f"   spread > 100 meV/atom:          {(multi > 0.1).mean():.1%}")

    # --- how much the winning trial beats the first one
    first = done[done.trial == 0].set_index("gene")["E_final_pa"]
    best = g.min()
    common = first.index.intersection(best.index)
    gain = (first.loc[common] - best.loc[common]).dropna()
    print("\ngain from taking the best trial rather than trial-0 (eV/atom):")
    print(f"   median {gain.median():.4f}   mean {gain.mean():.4f}   p90 {gain.quantile(0.9):.4f}")
    winner = done.loc[g.idxmin()]
    print("   winning trial index:",
          {int(k): int(v) for k, v in winner.trial.value_counts().sort_index().items()})

    # --- what each stage contributes
    st = done.dropna(subset=["E_1_fix-cell", "E_2_sym", "E_3_no-sym"])
    d12 = (st["E_1_fix-cell"] - st["E_2_sym"]) / st.n_atoms
    d23 = (st["E_2_sym"] - st["E_3_no-sym"]) / st.n_atoms
    print(f"\nper-stage energy drop, eV/atom (n={len(st)}):")
    print(f"   stage 1 -> 2 (release cell, keep symmetry): median {d12.median():.4f}  mean {d12.mean():.4f}")
    print(f"   stage 2 -> 3 (release symmetry):            median {d23.median():.4f}  mean {d23.mean():.4f}")
    print(f"   stage 3 changed E by >1 meV/atom in {(d23.abs() > 0.001).mean():.1%} of trials")


if __name__ == "__main__":
    main()
