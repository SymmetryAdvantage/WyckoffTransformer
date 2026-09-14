# What CrySPR's repeated relaxations actually buy

Measured on the 2500-gene WyFormer run `upi73i4k` (2026-09-01), relaxed with
MACE-MP-0a-small on CPU at `--n-trials 3 --fmax 0.05`, `steps_limit` left at its
default of 500. Every number below comes from the 7462 BFGS logfiles the run left
behind, parsed by `scripts/analyse_cryspr_trial_spread.py`; the per-trial energies
are in `generated/upi73i4k/cryspr_trial_energies.csv`.

## The shape of the calculation

`func_run` draws `n_trials` independent PyXtal samplings of the same Wyckoff gene and
keeps the lowest-energy one. Each trial runs `stepwise_relax`, which is **three**
relaxations, not one:

| stage | cell | symmetry | CIF label |
|---|---|---|---|
| 1 | fixed | `FixSymmetry` | `1_fix-cell` |
| 2 | free (`FrechetCellFilter`) | `FixSymmetry` | `2_sym_cell+pos` |
| 3 | free | none | `3_no-sym_cell+pos` |

So this run did 9 relaxations per gene, and the CLI default of 6 trials would do 18.

## Trials disagree, and the disagreement is bimodal

Spread (max − min) of the final energy per atom within a gene, over 2479 genes with
more than one surviving trial:

| percentile | eV/atom |
|---|---|
| p25 | 0.0001 |
| **p50** | **0.052** |
| p75 | 0.255 |
| p90 | 0.625 |
| p95 | 1.029 |
| p99 | 2.372 |

The distribution is not centred on its median — it is two populations:

- **39.0%** of genes have every trial agree to within **1 meV/atom**: PyXtal found the
  same minimum each time, and the extra trials were wasted.
- **42.1%** differ by more than **100 meV/atom** — the samplings landed in genuinely
  different basins, and which one you keep matters more than any relaxation setting.

Taking the best of 3 rather than trial-0 alone improves 66.0% of genes, by a mean of
0.127 eV/atom (median 0.000, p90 0.303). The winning trial index is uniform —
796 / 855 / 842 — and trial 2 wins 33.6% of the time against the 33.3% that
exchangeable draws predict. There is no ordering effect to exploit; the trials are
what they claim to be.

## Three trials is about the right number

Mean final energy over the 2420 genes where all three trials survived:

| | mean (eV/atom) | median | marginal gain |
|---|---|---|---|
| best-of-1 | −5.014 | −4.894 | — |
| best-of-2 | −5.109 | −4.995 | 0.095 |
| best-of-3 | −5.135 | −5.016 | 0.026 |

The gain falls by ~3.6× per added trial. Extrapolating that ratio, trials 4–6 are worth
roughly another 0.01 eV/atom in total, for a doubling of a stage that already took
4.5 h on 20 cores. **The default of 6 trials does not earn its cost here**; 2 or 3 does.

## Stage 3 is nearly free of consequence

Energy drop contributed by each stage, over 7387 trials with all three stages logged:

| transition | median | mean | p90 |
|---|---|---|---|
| 1 → 2 (release the cell, keep symmetry) | **0.366** | 0.757 | 1.82 |
| 2 → 3 (release symmetry) | **0.000000** | 0.000508 | ~0 |

Stage 3 moves the energy by more than 1 meV/atom in **0.4%** of trials, and by more than
10 meV/atom in **0.1%**.

This is exactly what `stepwise_relax`'s docstring predicts, now with a number on it: a
structure converged to a symmetric stationary point in stage 2 stays there, because for
a symmetry-invariant potential the force and stress components along symmetry-breaking
modes vanish identically, and only the MLIP's own numerical asymmetry can seed a
descent. Stage 3 earns its place only when stage 2 hit `steps_limit`, or when the
symmetric point is unstable enough for that noise to grow.

It costs about a third of the relaxation budget to change 1 gene in 250.

Removing it is not free of risk, though: the ~1% of trials that end at non-physical
energies (72 of 7462, `|E| > 50 eV/atom`, across 62 genes — collapsed cells, which
MACE happily reports as −10¹⁵ eV) come from the stages where nothing constrains the
cell. Stage 3 is where a structure is most exposed, and it is also the stage whose CIF
is kept.

## How converged the kept structures actually were

Independent evidence from LeMat-GenBench, which re-relaxes each submitted structure at
`fmax 0.02, steps 50` before reporting: the median structure needed only **6 steps**
and moved **0.121 Å** (mean 0.318 Å), so most were already at their MACE minimum. But
**107 of 2215** hit the 50-step cap, i.e. were still descending when the benchmark
stopped.

That matters because GenBench's stability and SUN metrics read `e_above_hull`, computed
from a single point on the structure **as submitted** — its own `relaxed_e_above_hull`
is written and never read. Relaxing first moves the reported numbers by:

| | as submitted | after GenBench's relaxation |
|---|---|---|
| stable (≤ 0) | 90 (4.07%) | 109 (4.93%) |
| ≤ 0.1 eV/atom | 671 (30.3%) | 736 (33.3%) |
| mean e_hull | 0.308 | 0.260 |
| median e_hull | 0.186 | 0.166 |

The relaxation lowered the hull energy for 2160 of 2213 structures, but the median
shift is only −0.007 eV/atom. The CrySPR stage is doing its job; the residual gap is
concentrated in the minority of structures that were still moving.

Feeding the relaxed energies to GenBench's own SUN metric — same thresholds, same
structure-matcher uniqueness, same LeMat-Bulk novelty, only the energy swapped into the
property the metric reads — moves the headline numbers by about a fifth:

| | as submitted | relaxed | change |
|---|---|---|---|
| SUN count (`e_hull ≤ 0`) | 23 | **29** | +26% |
| MSUN count (`0 < e_hull ≤ 0.1`) | 261 | **318** | +22% |
| **MetaSUN count** | **284** | **347** | **+22%** |
| MetaSUN rate (of valid) | 0.1282 | **0.1567** | |
| MetaSUN per generated gene | 0.114 | **0.139** | |

The as-submitted column reproduces the original benchmark run exactly (23 / 261 / 284,
90 stable, 581 metastable), which is what establishes that the energy source is the
only thing that differs between the two columns.

Reproduce with `scripts/recompute_relaxed_sun.py`: `--stage preprocess` caches both
energies (~29 min on 20 cores), `--stage sun --energy {relaxed,unrelaxed}` is then cheap.

## What to change

1. **Keep 3 trials, not 6.** The measured marginal gain does not justify the cost.
2. **Treat stage 3 as optional.** `fix_symmetry=False` already skips stage 2; a
   corresponding way to skip stage 3 would recover ~⅓ of the budget for a 0.4% effect.
   Spend it on a fourth trial instead — trials are where the energy actually is.
3. **Guard against collapsed cells.** A finite-energy sanity check after each stage
   would catch the 1% that reach −10¹⁵ eV, which currently propagate into the results
   CSV and drag every mean computed from it.
