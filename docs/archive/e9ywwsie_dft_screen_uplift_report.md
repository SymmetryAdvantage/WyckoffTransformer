# `e9ywwsie` DFT fixed-hull screen: (M)SUN uplift measurement

Run date: 2026-09-07/08. Pool: `generated/e9ywwsie_dft_attack` (5,000 sampled
genes). Code: `420cec3` on `training-loss-fixes`.

This measures whether the fixed-hull DFT screen of `docs/dft_fixed_hull_attack.md`
buys (M)SUN uplift when it is used to pick a subset of a de novo pool for the
`docs/de_novo_ranking_protocol.md` cascade. It is a ranking audit of the screen,
not an evaluation of the generator.

> **Cross-theory qualification.** The screen's two estimators and its hull are
> PBE. The protocol's energies and hull are ORB (`orb_conserv_inf`). A gene can
> therefore clear one and miss the other for reasons that have nothing to do
> with the screen's skill. Every uplift below is a lower bound on the screen's
> agreement with a PBE-evaluated funnel, by an unmeasured amount.

## Protocol

- **Generator**: wandb run `e9ywwsie`, 5,000 genes sampled unconditionally.
  5,000 valid, 4,989 unique, 68.9% gene-novel against LeMat-Bulk.
- **Screen** (`wyformer-dft-screen`): composition floor `s_F` from the local
  censored formula ensemble (`runs/formula_energy/ensemble.pt`), gene attainable
  energy `s_G` from wandb run `gene_min_energy_adamw_wsd-20260907-151306`, both
  compared against the same immutable PBE hull. Five ranking columns: the two
  composition scores (naive and epistemically adjusted), `gene_score`, and the
  conservative joint scores `s_J = max(s_F, s_G)` in both variants. Lower is
  better; the score is an energy margin above the hull in eV/atom.
- **Protocol** (`wyformer-protocol`, stages `relax` then `score`): ORB
  `orb-v3-conservative-inf-omat-20250404`, `LeMaterial/LeMat-Bulk-MLIP-Hull`
  revision `70d505b` (194,240 entries), trial schedule `0:1,2:2,*:3` (11,835
  trials, 2.372/gene), `release_symmetry` and `rattle` on, `fmax=0.05`.
  All 4,989 unique genes were relaxed — the screen was **not** used as a filter,
  so every arm below is scored on the same complete pool and the comparison is
  exact rather than sampled.
- **Compute**: PBS `16160868.pbs102` on ASPIRE 2A, one A100-40GB, 4 workers,
  11 h 29 m for the relax stage, inside the project Singularity container.
- **Analysis**: `scripts/analyse_dft_screen_uplift.py`, writing
  `generated/e9ywwsie_dft_attack/dft_screen_uplift.json`.

## Pool baseline

| stage | count | per sampled gene |
|---|---:|---:|
| sampled genes | 5,000 | — |
| unique genes relaxed | 4,989 | — |
| structure produced | 4,969 | 99.4% |
| valid structure | 4,539 | 90.8% |
| novel structure | 3,419 | 68.4% |
| metastable (`e_hull <= 0.1`) | 1,443 | 28.9% |
| stable (`e_hull <= 0`) | 51 | 1.0% |

MetaSUN 28.9%, SUN 1.02% of submitted genes. These are the denominators every
uplift below is measured against.

## Primary result: the screen is a strong stability ranker

Spearman correlation between each screen score and the ORB-relaxed
`e_above_hull`, over the 4,539 genes that produced a valid structure:

| score | rho | p |
|---|---:|---:|
| `joint_score_naive` | +0.588 | < 1e-300 |
| `joint_score_adjusted` | +0.585 | < 1e-300 |
| `gene_score` | +0.535 | < 1e-300 |
| `composition_score_adjusted` | +0.503 | 5.7e-290 |
| `composition_score_naive` | +0.443 | 4.7e-217 |

Binned by `joint_score_adjusted` decile (decile 0 = best score); the median is
over every gene in the bin, the rates over the 4,989 submitted:

| decile | median `e_hull` | metastable | valid | **novel** | MetaSUN | formula already known |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.021 | 90.2% | 98.4% | 42.9% | 34.1% | 83.2% |
| 1 | 0.046 | 84.4% | 98.6% | 59.5% | **46.7%** | 72.5% |
| 2 | 0.067 | 70.9% | 96.2% | 64.1% | 40.5% | 64.7% |
| 3 | 0.080 | 61.9% | 94.4% | 70.9% | 40.7% | 59.3% |
| 4 | 0.095 | 51.3% | 91.2% | 74.3% | 34.7% | 55.7% |
| 5 | 0.113 | 42.8% | 85.5% | 65.7% | 23.9% | 55.2% |
| 6 | 0.124 | 39.9% | 87.2% | 72.7% | 27.3% | 50.3% |
| 7 | 0.153 | 26.1% | 86.0% | 75.4% | 17.8% | 42.3% |
| 8 | 0.201 | 19.0% | 85.6% | 79.2% | 15.0% | 39.1% |
| 9 | 0.265 | 11.2% | 86.8% | 80.6% | 8.6% | 43.5% |

The stability signal is unambiguous and monotone: median `e_above_hull` runs
0.021 to 0.265 eV/atom across the deciles, and the metastable rate 90.2% to
11.2%. A PBE-trained screen predicts ORB-relaxed hull distance well enough to
sort the pool into an 8x spread. Structural validity tracks it too, 98.4% to
~86%.

## Why that only becomes 1.4x MetaSUN

Novelty runs the other way, and almost exactly cancels the gain. The best
decile is 42.9% novel against 80.6% for the worst, and 83.2% of its genes have
a formula already present in the reference set. The screen finds low-lying
structures largely by finding compositions the reference data already knows are
low-lying — which is what a hull-referenced estimator trained on that data
should be expected to do.

**MetaSUN therefore peaks at decile 1, not decile 0.** Taking the very top of
the ranking is worse than taking the second tenth. This is the central finding:
the screen's failure mode is not weak stability prediction, it is that
stability and novelty are anti-correlated under it, and (M)SUN charges for both.

Unrestricted top-`B` arms, MetaSUN rate and uplift over the 28.9% pool rate:

| arm | B=250 | B=500 | B=1000 | B=2000 |
|---|---|---|---|---|
| `joint_score_adjusted` | 0.300 / 1.04x | 0.340 / 1.18x | 0.403 / **1.39x** | 0.406 / **1.40x** |
| `joint_score_naive` | 0.292 / 1.01x | 0.346 / 1.20x | 0.381 / 1.32x | 0.397 / 1.37x |
| `composition_score_adjusted` | 0.336 / 1.16x | 0.370 / 1.28x | 0.403 / 1.39x | 0.392 / 1.36x |
| `composition_score_naive` | 0.232 / **0.80x** | 0.284 / 0.98x | 0.345 / 1.19x | 0.369 / 1.28x |
| `gene_score` | 0.364 / **1.26x** | 0.348 / 1.20x | 0.372 / 1.29x | 0.377 / 1.30x |

Two secondary observations:

- `composition_score_naive` is *anti-correlated* at the top of its own ranking
  (0.80x at B=250, worse than a random draw). The epistemic adjustment is not a
  refinement of the naive floor; it is what makes it usable at all.
- At small budget the joint scores are worthless (1.0x) and `gene_score` is the
  only arm with a defensible p-value (0.005). The composition estimators need
  several hundred picks before they pull ahead. This is the novelty cancellation
  again: the joint score concentrates known formulas hardest.

## Screening on novelty first recovers the signal

If gene novelty is applied as a filter *before* the screen ranks — restrict to
the 2,166 genes whose reduced formula is absent from the reference table, then
take the top `B` by `joint_score_adjusted`:

| arm | B=250 | B=500 | B=1000 | B=2000 |
|---|---|---|---|---|
| all genes, `joint_score_adjusted` | 1.04x | 1.18x | 1.39x | 1.40x |
| **novel-formula only, `joint_score_adjusted`** | **2.39x** (p=9e-42) | **2.12x** (p=7e-57) | 1.63x | 1.09x |
| novel-formula only, `gene_score` | 1.59x | 1.77x | 1.60x | 1.08x |

MetaSUN 69.2% at B=250 against a 28.9% pool rate. The decay to 1.09x at B=2000
is arithmetic, not a failure: the subset holds only 2,166 genes, so B=2000 takes
93% of it and the arm degenerates to the subset's base rate. The gain is real
only while the budget is a small fraction of the novel subset.

This is the operating recommendation: **the screen should rank within a
novelty-filtered pool, never rank the raw pool.** Both signals are individually
strong and their composition is where the value is; used alone the screen spends
most of its ranking power re-finding known chemistry.

## What the novelty filter is actually using

The filter above drops genes whose reduced formula appears in
`data/formula_energy/formula_table.parquet`, which is the composition
ensemble's own training table and is derived from LeMat-Bulk -- the same
reference LeMat-GenBench scores novelty against. That coupling deserves stating
plainly, because "filter on novelty, then report a novelty-dependent metric"
reads as circular whether or not it is.

Three alternatives were tested on this pool. Only the third is worth using.

**Model-internal uncertainty does not substitute.** The obvious
reference-free proxy is the ensemble's epistemic sigma: unfamiliar chemistry
should carry high variance. It does, weakly -- Spearman +0.086 against
`novel_structure`, and the mean sigma is 0.064 for absent formulas against
0.046 for known ones. But sigma also correlates +0.388 with the achieved
`e_above_hull`, so ranking optimistically (`f_hat - lambda*sigma - h`, the
sign-flipped winner's-curse term) buys novelty by buying instability. Sweeping
lambda from +1 to -3 moves the selected slice from 38% to 59% novel and peaks
at **1.31x** MetaSUN at B=250, against 2.39x for the explicit filter. There is
no free reference-free proxy hiding in this screen.

**Composition provenance is a real but much weaker signal.** Ranking only
within compositions carrying at most two entries in the table -- an
under-explored-chemistry criterion rather than a membership test -- gives 1.33x
at B=250 and 1.57x at B=500.

**Restricting to the training split keeps most of the effect and is
defensible.** The table carries a formula-hashed `split`. Of the 4,989
representatives, 2,563 have a formula in `train`, 260 in `val`/`test`, and
2,166 are absent. Dropping only the 2,563 the model was actually fitted on is
training-set deduplication, not novelty filtering: it consults nothing the
evaluation could be holding out, and it is what any real campaign does rather
than spend relaxation budget re-deriving its own training data.

| arm at B=250 | MetaSUN | uplift | novel |
|---|---:|---:|---:|
| raw pool | 0.300 | 1.04x | 38% |
| drop formulas in any split | 0.692 | 2.39x | 96% |
| **drop only trained-on formulas** | **0.620** | **2.14x** | 80% |
| optimistic sigma ranking, best lambda | 0.380 | 1.31x | 56% |
| under-explored compositions only | 0.384 | 1.33x | 48% |

The 260 val/test-formula genes left in the pool are novel at 53.8%, the same
rate as the trained-on ones (53.6%), so keeping them neither helps nor hurts
beyond their small share -- which is the point: they are judged on their merits.

A soft variant -- adding a fixed penalty to trained-on formulas instead of
dropping them -- was also tried and is **not** worth using. At +0.15 eV/atom it
selects exactly the same 250 genes as the hard drop (2.14x, identical p-value),
because the penalty exceeds the score spread at the top of the ranking. It uses
the same information, produces the same decision, and only makes the mechanism
harder to see. If the training-set restriction is defensible it should be
declared, and if it is not, hiding it in a score does not fix it.

The structural fix, untested here because it needs a retrain, is to move the
effect into the generator so no selection step consults any table: a decoder
that emits fewer training-set formulas needs no filter downstream.

## SUN

| arm | B=250 | B=500 | B=1000 | B=2000 |
|---|---|---|---|---|
| `joint_score_adjusted` | 1.57x | 2.54x | 1.66x | 1.57x |
| `joint_score_naive` | 3.91x | 2.54x | 1.86x | 1.57x |
| `composition_score_adjusted` | 2.74x | 2.15x | 2.05x | 1.47x |
| `composition_score_naive` | 2.35x | 1.96x | 2.35x | 1.66x |
| `gene_score` | 2.35x | 2.74x | 1.96x | 1.42x |

SUN uplift is positive everywhere but the pool holds only 51 stable structures,
so every small-budget cell rests on 4-10 hits and the individual multipliers are
not separable from each other. The defensible statement is the B=2000 column,
where all five arms land in 1.4-1.7x with p < 0.01. The novelty pre-filter does
**not** help SUN (1.57x at B=250, 0.64x at B=2000) — stable *and* novel is rare
enough here that this pool cannot resolve it. Answering the SUN question
properly needs a larger pool or a generator with a higher stable rate.

## Verdict

Yes, the fixed-hull screen achieves (M)SUN uplift, with two qualifications that
matter more than the headline number:

1. Used as a ranker on the raw pool it delivers ~1.4x MetaSUN at large budget
   and nothing at small budget. Ranked after dropping the formulas the
   composition model was trained on it delivers 2.14x at B=250, and 2.39x if
   the whole reference table is excluded rather than the training split alone.
   The reported configuration should be the training-split one, disclosed as
   deduplication against training data.
2. It is not usable as a *filter* in its intended conservative form. Only 8 of
   5,000 genes clear `joint_score_adjusted <= 0` and 35 clear
   `composition_score_adjusted <= 0`; the `max` fusion has essentially no pass
   set on this pool. Its value here is entirely in the ordering.

Pass-set sizes at the `score <= 0` threshold, for reference:

| score | genes passing (of 5,000) |
|---|---:|
| `joint_score_adjusted` | 8 |
| `joint_score_naive` | 30 |
| `composition_score_adjusted` | 35 |
| `gene_score` | 207 |
| `composition_score_naive` | 309 |

## Reproduction

```
wyformer-dft-screen  generated/e9ywwsie_dft_attack/wyckoff_genes.json.gz \
    --formula-ensemble runs/formula_energy/ensemble.pt \
    --regressor-wandb-run gene_min_energy_adamw_wsd-20260907-151306 \
    --out generated/e9ywwsie_dft_attack/dft_screen.csv
qsub -v POOL=generated/e9ywwsie_dft_attack,WORKERS=4,\
POST="python scripts/analyse_dft_screen_uplift.py" scripts/protocol_relax.pbs
```

Artefacts under `generated/e9ywwsie_dft_attack/`: `dft_screen.csv` (5,000 rows,
all five scores), `protocol/structures.csv` (4,989 relaxed, `e_above_hull` and
the three funnel flags), `protocol/funnel.json`, `protocol/manifest.json`,
`dft_screen_uplift.json` (all arms x budgets x metrics with bootstrapped random
baselines and Fisher p-values).

**Prerequisite fix.** This pool could not be built before `420cec3`. A gene
record from `pyxtal_notation_to_sites` carries pymatgen `Element` objects, while
a tokeniser restored from JSON by `WyckoffProcessor.from_pretrained` is keyed by
element symbols, so `filter_supported_tokens` dropped every generated gene and
the screen aborted with "All structures were dropped due to unsupported
tokens." `prediction.tokeniser_key` now resolves both conventions; regression
tests in `src/wyckoff_transformer/tests/test_prediction_vocabulary.py`.
