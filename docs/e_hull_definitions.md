# Every `e_hull` in this repository

Six different numbers in this codebase are called some variant of `e_hull`. They
are not interchangeable, and two of them are not even the same *kind* of
quantity — one is a distance to the hull, the other is the hull's own energy at
a composition. This is the inventory, what has been measured to agree, and the
rules for keeping them apart.

## The inventory

| # | name in code | what it is | energy source | reference set | sign |
|---|---|---|---|---|---|
| 1 | `e_hull`, `e_form` in `lemat_pbe_ehull.csv.gz` | distance above hull, eV/atom | PBE `energy_corrected` (MP2020-corrected) | the whole LeMat-Bulk PBE archive, self-referentially | ≥0, or NaN |
| 2 | `energy_above_hull` in the training caches | (1) clipped at zero | as (1) | as (1) | ≥0 |
| 3 | `e_hull_at_composition`, `hull_energy` | the hull's *formation energy* at a composition, eV/atom | as (1) | rows of (1) within 1 meV/atom of the hull | any |
| 4 | `e_above_hull` in `protocol/structures.csv` | distance above hull, eV/atom | one MLIP's total energy | `LeMat-Bulk-MLIP-Hull`, that MLIP's own split | any, incl. negative |
| 5 | LeMat-GenBench `get_energy_above_hull` | as (4) | as (4) | as (4) | any |
| 6 | `corrected_chgnet_ehull` | distance above hull, eV/atom | CHGNet | MP-20-era reference, external | ≥0 |

Where each comes from:

1. **`scripts/compute_e_hull.py`** — shuyayamazaki's, commit `e943883`
   (2025-11-18). `PDEntry(full_formula, energy_corrected)` against one
   `PhaseDiagram` per chemical system, built from the archive itself. Produces
   `e_form` (formation energy per atom) and `e_hull` (distance) into
   `data/lemat-bulk/lemat_pbe_ehull.csv.gz`.
2. **`scripts/build_lemat_bulk_fmax.py`** — the conditioning label. `max(0,
   e_hull)`, because pymatgen's non-negative return lands at −2.7e-15 on one
   row and `log1p` will not take it. Rows with NaN `e_hull` cannot be carried at
   all, which is where the exclusions below bite.
3. **`formula_energy/dataset.py`** (`e_form − e_hull`, per formula) and
   **`formula_energy/screen.py:HullLookup`** (rebuilt from the near-hull subset,
   converted from pymatgen's absolute hull energy to a formation energy). Used
   by `wyformer-gene-screen` and `wyformer-dft-screen`, which predict *formation*
   energies and need something on the same scale to compare against.
4. **`evaluation/hull_energy.py:HullEnergyCalculator`** — the de novo ranking
   protocol. See [the protocol doc](de_novo_ranking_protocol.md).
5. **LeMat-GenBench** `preprocess.reference_energies.get_energy_above_hull`.
   (4) is a port of it; `tests/test_genbench_equivalence.py` pins them to 1e-9.
6. **The WyFormer-paper-era CrySPR pipelines**, via
   `evaluation/generated_dataset.py`. Read from per-dataset
   `*.ehull.csv.gz` files (column `ehull_refs_to_conventional_vc-relax`)
   produced outside this repository; the reference set and corrections are not
   recoverable from anything here. Fine for reproducing published tables in
   `generated/datasets.yaml`, not for new work.

## What has been measured to agree

**(1) and (5) are the same DFT hull.** Of the 144,127 entries in the `dft`
split of `LeMat-Bulk-MLIP-Hull` — every LeMat-Bulk entry within 1 meV/atom of
the PBE hull — 120,341 also carry an `e_hull` from `compute_e_hull.py`, and
**all 120,341 of them are ≤ 0.001 eV/atom**, max 0.00100, median 0.00000. Two
independent constructions, over the same archive, agreeing to the resolution of
the threshold that defines the split.

**The energy scales are identical.** LeMat's `true_energy` equals our
`energy_corrected` for **100%** of those rows (`|Δ| < 1e-6` eV/atom). There is
no correction, functional or reference offset between them to reconcile.

**Each hull split's `energy` column is its own model's.** `energy` matches
`<mlip>_energy` exactly for all five MLIP splits, and for `dft` to 1.4e-6
eV/atom (float32 storage of a float64 column). (4) reads `energy`, so the
self-consistent pairing that makes `e_above_hull` meaningful holds.

**(3)'s two routes agree.** `e_form − e_hull` from the table and
`HullLookup.hull_energy_per_atom` from the rebuilt phase diagram are checked
against each other at run time; `cli/dft_screen.py` raises if any formula
disagrees by more than 1e-6 eV/atom.

## What differs, and whether it should

**MLIP vs PBE ((4) vs (1)) — intentional.** `e_above_hull` is only meaningful
when the structure energy and the hull come from the same potential; per-atom
offsets between MLIPs run to 0.057 eV/atom, against the ~0.023 eV/atom effects
the protocol resolves. Never compare a number from (4) with one from (1).

**Whole archive vs the 1 meV/atom slice ((1) vs (4), (3)) — equivalent.** A hull
vertex sits at exactly 0, so a positive threshold cannot cut one; what it drops
is entries strictly inside the hull, which `PhaseDiagram` ignores. This is why
(3) can rebuild from the near-hull subset and still get (1)'s answer.

**Distance vs level ((1), (4) vs (3)) — a different quantity.** Related by

```
e_hull(structure)  =  e_form(structure)  −  e_hull_at_composition(its formula)
```

and (4)'s absolute convention differs from (3)'s formation convention by the
elemental reference energies, which cancel in the difference. That identity is
what lets a formation-energy screener and an MLIP-energy protocol talk about the
same threshold; `tests/test_hull_conventions.py` pins it on a synthetic hull.

**Below-hull rows: NaN in (1), negative in (4) — a real asymmetry.**
`compute_e_hull.py` calls `PhaseDiagram.get_e_above_hull`, which *raises* for an
entry below the hull, and the surrounding `except` turns that into
`(None, None)` — losing `e_form` as well. (4) calls
`get_decomp_and_e_above_hull(..., allow_negative=True)` and returns the negative
value. It costs nothing today, because (1)'s reference is the target set itself,
so nothing can be below it; it would bite the moment a *different* reference is
passed with `--ref-file`. That case is the answer key, and
`formula_energy/answer_key.py` already routes around it by comparing formation
energies against the shallow hull's *level* (3) rather than reading a negative
distance.

**Element exclusions: (1) only.** `compute_e_hull.py` refuses three classes of
system. Measured over all 5,335,299 archive rows, `e_hull` is NaN for 589,250
(11.04%), and the cause decomposes as:

| exclusion | stated reason (from the deleted comments of `e943883`) | rows lost |
|---|---|---|
| any element with Z ≥ 84 (Po…Og) | "radioactive or not well-supported in common databases" | **579,217** |
| `Yb` | none — "certain elements like Yb or rare elements" | 10,032 |
| chemsys with ≥ 10 elements | "Qhull … can have issues with high-dimensional systems" | **0** |
| phase diagram genuinely failed | — | 1 |

Three observations follow. The ≥10-element guard **never fires**: no row of
LeMat-Bulk has ten or more elements. Yb costs 0.19%. And the Z ≥ 84 list is
essentially the entire loss — 10.9% of the archive — for chemistry the reference
hull covers perfectly well: the published hull splits carry Th, U, Np, Pu, Ac
and Pa, each with its own elemental reference, and 1211 Yb entries. The
exclusions also apply *only to the row being labelled*, not to the reference
dictionary, so the hull geometry is unaffected — this is lost labels, not a
distorted hull.

**One-hot encodings: never mix ours with the published matrices.**
`evaluation/hull_energy.py` and `formula_energy/screen.py` both use 118 slots
indexed by `Z − 1`, consistently with each other, and both build their own
composition matrix. The `threshold_0_001/*_composition_matrix.npz` files
published alongside the hull are **119 wide and indexed by `Z`** (verified: row
0 has bits at 1, 3, 8, 16, 33 for H-Li-O-S-As). LeMat-GenBench's own
`fingerprinting.encode_compositions` uses the 118/`Z − 1` convention and
therefore cannot be applied to those files either.

## Rules

1. **Name the kind.** A distance above the hull is `e_above_hull` (protocol) or
   `e_hull` (archive); the hull's energy at a composition is
   `e_hull_at_composition` or `hull_energy`. Never introduce a third spelling
   for either.
2. **Never mix an MLIP energy with the PBE hull, or two MLIPs.** `--mlip` is
   restricted to published hulls for exactly this reason.
3. **Record provenance.** The protocol writes `manifest.json["hull"]` — repo,
   split, revision, entry count. Anything new that computes a hull energy should
   record where its reference came from, because the number cannot be traced
   afterwards.
4. **Use (3), not (1), when comparing against a *prediction*.** The screeners
   predict formation energies; the hull level is what they must beat.
5. **Do not use (6) for new work.** Its reference is not reproducible from this
   repository.

## Open item

The Z ≥ 84 exclusion in `compute_e_hull.py` drops 579,217 rows — 10.9% of the
archive — from every training label derived from it, on a stated concern about
database support that the LeMat hull itself does not share. Recovering them
means one rerun of `compute_e_hull.py` and a rebuild of the conditioning caches,
and would also make the actinide chemistry of the archive available to the
conditioned models. The `Yb` and `≥ 10 elements` clauses can go at the same
time: the first has no stated reason, the second has never once fired.
