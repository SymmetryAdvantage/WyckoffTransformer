# Candidate disorder parents in the Cu–Ge–Te hull campaign

**Measurement, 2026-09-29.** Source campaign: commit `54c955b`, W&B run
[`pglsoqms`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/pglsoqms).
This analysis was run from base checkout commit `a50d319` with the
`scripts/infer_cu_ge_te_disorder_parents.py` implementation and its test added
in the same change as this report.
W&B run: [`139vcdkm`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/139vcdkm).
W&B artifact: [`cu_ge_te_disorder_parents_20260929:v0`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/artifacts/disorder_parent_inference/cu_ge_te_disorder_parents_20260929/v0).

The script tests whether ordered ORB-relaxed cells share a more symmetric
substitutional framework. It masks Cu–Ge, Cu–Te, or Ge–Te as one species, reduces
the masked cell to its idealized primitive parent with spglib, then groups parents
only when their geometry passes `StructureMatcher`. Each ternary cell can support
three different hypotheses, so family counts and memberships **must not be
summed into a count of materials**. The partially occupied CIF for each family
uses the composition of its lowest-energy representative, averaged over the
parent's symmetry-equivalent sites; campaign frequencies are not physical site
occupancies.

At `symprec=0.15 Å`, `ltol=0.15`, `stol=0.25`, `angle_tol=5°`, and a 15% parent
primitive-volume limit, the 17,734 generated entries within 100 meV/atom of the
ORB hull yielded:

| Readout | Count |
|---|---:|
| Entries with a Cu–Ge, Cu–Te, or Ge–Te mask | 17,729 |
| Unassigned elemental Te entries | 5 |
| Child-to-parent-hypothesis memberships | 22,889 |
| Candidate parent geometries across all masks | 17,934 |
| Parent geometries containing more than one reduced formula | 338 |
| Stronger families: ≥3 formulas and ≥10 entries | 16 |

All 16 stronger families are Cu–Ge binary parents. The largest is an fcc
`Fm-3m` one-site framework shared by 769 ordered entries over 38 formulas; the
next is an hcp `P6₃/mmc` two-site framework shared by 361 entries over 21
formulas. These are **candidate substitutional alloy families**. A common
framework across a wide composition range does not show that the range is one
homogeneous experimental phase. Among ternary entries, 24 parent geometries
contain multiple formulas, but none meets the stronger support criterion. Most
ternary cross-formula families have only two entries.

A tighter symmetry tolerance, `symprec=0.10 Å`, finds 18,261 candidate parent
geometries, 353 containing multiple formulas, and again 16 stronger families,
all Cu–Ge. Its largest fcc and hcp groups contain 602 and 262 entries,
respectively, and are subsets of the corresponding `0.15 Å` groups. The broad
Cu–Ge finding survives this tolerance change; the exact family counts do not.

This is a first-pass inversion of *substitutional* disorder. It does not recover
vacancy disorder, prove that different orderings form one finite-temperature
phase, test miscibility, or compare the inferred parent with a licensed ICSD
disordered record. Some low-symmetry parents are singletons. A real-material
claim requires a forward check (enumerate ordered children of the proposed
parent and recover held-out structures), thermodynamic assessment, and
experimental phase evidence. The method is motivated by the
[order–disorder family-tree framework](https://arxiv.org/html/2604.21386),
which also benchmarks WyFormer, but this campaign analysis uses its own explicit
geometry check and does not reproduce that paper's ICSD benchmark.

Reproduce the analysis on zeus from this checkout with:

```bash
.venv/bin/python scripts/infer_cu_ge_te_disorder_parents.py --workers 12
.venv/bin/python scripts/infer_cu_ge_te_disorder_parents.py --workers 12 \
    --symprec 0.10 --output-dir artifacts/cu_ge_te_hull/disorder_parents_symprec010
```

The primary files are `families.csv`, `memberships.csv`, `analysis.json`, and
representative partial-occupancy CIFs under
`artifacts/cu_ge_te_hull/disorder_parents/`. The tighter-tolerance run is under
`artifacts/cu_ge_te_hull/disorder_parents_symprec010/`.
