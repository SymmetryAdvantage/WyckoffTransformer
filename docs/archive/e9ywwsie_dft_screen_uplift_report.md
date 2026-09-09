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

## Headline: two axes

Selection down the rows -- rank the whole pool, or deduplicate genes against
the training set first -- and budget accounting across the columns, a gene
against a relaxation. The two denominators differ because the trial schedule
spends more relaxations on the high-DoF genes selection prefers, so the
per-gene column flatters every ranked arm and the per-relaxation column is the
one a compute objection is entitled to.

Unfiltered pool: **MetaSUN 0.2892/gene, 121.9 per 1k relaxations**;
**SUN 0.0102/gene, 4.3 per 1k relaxations**. Ranking is by
`joint_score_adjusted` throughout; uplift is against those pool rates.

### MetaSUN

| budget | arm | hits | per gene | per relaxation | x gene | x relax | p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 250 | naive: rank whole pool | 75 | 0.3000 | 0.1456 | 1.04 | 1.19 | 0.37 |
| 250 | **smart: dedup, then rank** | 201 | **0.8040** | **0.3059** | **2.78** | **2.51** | 1.9e-67 |
| 500 | naive: rank whole pool | 170 | 0.3400 | 0.1565 | 1.18 | 1.28 | 0.0053 |
| 500 | **smart: dedup, then rank** | 378 | 0.7560 | 0.2836 | 2.61 | 2.33 | 2e-116 |
| 1000 | naive: rank whole pool | 403 | 0.4030 | 0.1812 | 1.39 | 1.49 | 2.9e-18 |
| 1000 | **smart: dedup, then rank** | 664 | 0.6640 | 0.2476 | 2.30 | 2.03 | 1.2e-172 |
| 2000 | naive: rank whole pool | 811 | 0.4055 | 0.1763 | 1.40 | 1.45 | 3.4e-49 |
| 2000 | **smart: dedup, then rank** | 1034 | 0.5170 | 0.1932 | 1.79 | 1.58 | 1.2e-185 |

### SUN

| budget | arm | hits | per gene | per relaxation | x gene | x relax | p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 250 | naive: rank whole pool | 4 | 0.0160 | 0.0078 | 1.57 | 1.80 | 0.25 |
| 250 | **smart: dedup, then rank** | 13 | **0.0520** | **0.0198** | **5.09** | **4.59** | 7.9e-07 |
| 500 | naive: rank whole pool | 13 | 0.0260 | 0.0120 | 2.54 | 2.78 | 0.0012 |
| 500 | **smart: dedup, then rank** | 16 | 0.0320 | 0.0120 | 3.13 | 2.79 | 2.1e-05 |
| 1000 | naive: rank whole pool | 17 | 0.0170 | 0.0076 | 1.66 | 1.77 | 0.018 |
| 1000 | **smart: dedup, then rank** | 26 | 0.0260 | 0.0097 | 2.54 | 2.25 | 7.3e-07 |
| 2000 | naive: rank whole pool | 32 | 0.0160 | 0.0070 | 1.57 | 1.61 | 0.00085 |
| 2000 | **smart: dedup, then rank** | 35 | 0.0175 | 0.0065 | 1.71 | 1.52 | 3.2e-05 |

Reading it:

- **Selection is worth far more than the accounting choice costs.** Moving
  from naive to smart is 2.7x at B=250; moving from per-gene to per-relaxation
  costs 10%. Conceding the stricter denominator does not endanger the claim.
- **The naive arm is where the per-relaxation number is kinder**, not harsher
  (1.19x against 1.04x at B=250): ranking the raw pool picks *low*-DoF genes,
  2.06 trials each against the pool's 2.372. Deduplication reverses that
  (2.63), which is why the smart arm is the one that gives 0.27x back.
- **Dedup rescues SUN.** On the raw pool SUN is 1.57x at B=250 and not
  significant (p=0.25, 4 hits). After dedup it is 5.09x per gene and 4.59x per
  relaxation on 13 hits, p=8e-07 -- the largest uplift in either table, since
  the pool's SUN base rate is so low that any real enrichment shows as a big
  multiple. Thirteen hits is still thirteen hits; the multiplier is not precise,
  but it is no longer consistent with chance.
- **Both metrics decay with budget** and for the same reason: at B=2000 the arm
  has taken 58% of the 3,439 gene-novel genes and is converging on that
  subset's base rate. Selection buys the most where the budget is scarcest.

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

## Deduplicating against the training set recovers the signal

The protocol already fingerprints every gene against the reference set, and
`screen.json` records the answer. Restricting the ranking to the 3,439
gene-novel genes -- and changing nothing else -- gives:

| filter, then rank by `joint_score_adjusted` | B=250 | B=500 | B=1000 |
|---|---|---|---|
| none (raw pool) | 1.04x | 1.18x | 1.39x |
| formula absent from the composition table | 2.39x | 2.12x | 1.63x |
| **gene fingerprint novel** | **2.78x** | **2.61x** | **2.30x** |

MetaSUN 80.4% at B=250 against a 28.9% pool rate. Gene-level novelty beats the
formula-level filter on every budget and degrades far more slowly, because it
keeps 3,439 genes rather than 2,166 -- the formula test throws away every novel
structure that happens to sit on a known composition, which is 43% of the novel
structures in this pool.

This is the configuration to report, and the reason is that it is not a special
mechanism at all. Novelty here is measured against the training set, as
MatterGen and the LeMat-GenBench convention have it; the filter is a lookup
against that same training set, in the representation the model generates in,
before anything is relaxed. That is deduplication against training data. A
generator that re-emits its own training set and a generator that declines to
are being asked a different question, and the second one is the one anybody
running a real campaign asks.

## The trial-budget objection, measured

A gene is not one relaxation. The schedule `0:1,2:2,*:3` spends more trials on
high-DoF genes, and the ranked slices are higher-DoF than the pool, so a
per-gene budget silently hands them more compute:

| arm (B=250) | trials | trials/gene | MetaSUN | MetaSUN per 1k relaxations |
|---|---:|---:|---:|---:|
| random | 593 | 2.372 | 0.289 | 122 |
| `joint_score_adjusted` | 515 | 2.060 | 0.300 | 146 |
| formula-absent + joint | 705 | 2.820 | 0.692 | 245 |
| gene-novel + joint | 657 | 2.628 | 0.804 | **306** |

The selected slice costs 11% more relaxations per gene than a random one, so
the per-gene 2.78x is 2.51x once normalised by relaxations actually spent. The
right response to "you spent more compute" is that number, not an argument that
genes are cheap to generate -- which is true and beside the point, since the
budget in question is the relaxation budget. `analyse_dft_screen_uplift.py`
reports both denominators for every arm.

## A learned gene-novelty model is not worth training

The natural next move -- train a classifier for "is gene X known", the way the
energy screener was trained -- was checked against its own ceiling before being
built. Substituting an oracle that knows the *relaxed structure* novelty, which
is the quantity (M)SUN actually scores and which no gene-level model can beat:

| filter, then rank by `joint_score_adjusted` | B=250 | B=500 | B=1000 |
|---|---|---|---|
| gene fingerprint novel (free lookup) | 2.78x | 2.61x | 2.30x |
| **ORACLE: relaxed structure is novel** | 2.75x | 2.75x | 2.43x |

The free lookup is already at the ceiling -- ahead of it at B=250, 0.13x behind
at B=1000. There is no headroom for a learned model to occupy.

The lookup is not perfect: it agrees with structure-level novelty on 85.8% of
genes, with 364 gene-novel genes relaxing into known structures and 344
gene-known genes producing novel ones. Those errors nearly cancel, and more to
the point they are not concentrated at the top of the joint-score ranking,
which is the only region a budgeted screen visits.

So the answer is that this model already exists, costs nothing, and needs no
training run. It is also the more defensible artefact: a learned surrogate for
a training-set membership test is an approximation to a lookup wearing a
model's clothes, which is a harder thing to justify than the lookup, not an
easier one. The same objection that retires the soft-penalty variant below
retires the classifier.

The one version with headroom is a different task: predict, from the gene, the
*probability that a trial produces a novel structure*, and use it to allocate
trials rather than to filter genes. That is a policy over the relaxation
budget, not a membership test, and it is the only place the 14% disagreement
above could be turned into anything.

### Alternatives that were tried and do not work

**Model-internal uncertainty does not substitute for the lookup.** The
ensemble's epistemic sigma is a weak novelty signal -- Spearman +0.086 against
`novel_structure`, mean 0.064 on absent formulas against 0.046 on known ones --
and it correlates +0.388 with the achieved `e_above_hull`, so ranking
optimistically (`f_hat - lambda*sigma - h`, the sign-flipped winner's-curse
term) buys novelty by buying instability. Sweeping lambda from +1 to -3 moves
the slice from 38% to 59% novel and peaks at **1.31x** at B=250.

**Composition provenance is a real but much weaker signal.** Ranking only
within compositions carrying at most two entries in the table -- under-explored
chemistry rather than a membership test -- gives 1.33x at B=250, 1.57x at
B=500.

**A soft penalty is the same decision with the mechanism hidden.** Adding a
fixed penalty to known-formula genes instead of dropping them selects, at
+0.15 eV/atom, exactly the same 250 genes as the hard drop -- identical rate,
identical p-value -- because the penalty exceeds the score spread at the top of
the ranking. It uses the same information and reaches the same answer while
making it harder to see. If a training-set restriction is defensible it should
be declared; if it is not, burying it in a score does not fix it.

**Restricting to the training split, for the formula-level filter.** The
composition table carries a formula-hashed `split`: 2,563 of the 4,989
representatives have a formula in `train`, 260 in `val`/`test`, 2,166 are
absent. Dropping only the trained-on 2,563 gives 2.14x at B=250 against 2.39x
for dropping all splits, and consults nothing an evaluation could hold out. The
gene-level filter above does not need this refinement -- the protocol's
reference is the training set by construction -- but the option is recorded
because the formula-level variant does.



## SUN

| arm | B=250 | B=500 | B=1000 | B=2000 |
|---|---|---|---|---|
| `joint_score_adjusted` | 1.57x | 2.54x | 1.66x | 1.57x |
| `joint_score_naive` | 3.91x | 2.54x | 1.86x | 1.57x |
| `composition_score_adjusted` | 2.74x | 2.15x | 2.05x | 1.47x |
| `composition_score_naive` | 2.35x | 1.96x | 2.35x | 1.66x |
| `gene_score` | 2.35x | 2.74x | 1.96x | 1.42x |

On the raw pool the SUN column is positive everywhere but rests on 4-17 hits
per cell, and the individual multipliers are not separable from each other. The
defensible statement for the unfiltered arms is the B=2000 row, where all five
land in 1.4-1.7x with p < 0.01.

Deduplication changes that: gene-novel + `joint_score_adjusted` reaches 5.09x
per gene at B=250 on 13 hits, p=8e-07 (headline table above). An earlier read
of this pool, using the *formula*-level filter, concluded that pre-filtering
does not help SUN; that was a property of the formula filter, which discards
too many novel structures sitting on known compositions, not of deduplication.
The pool still holds only 51 stable structures in total, so the size of the
effect remains poorly determined even though its existence no longer is.

## Verdict

Yes, the fixed-hull screen achieves (M)SUN uplift, with two qualifications that
matter more than the headline number:

1. Used as a ranker on the raw pool it delivers ~1.4x MetaSUN at large budget
   and nothing at small budget. Ranked *after* deduplicating the genes against
   the training set -- the free fingerprint lookup the protocol already
   performs -- it delivers **2.78x MetaSUN and 5.09x SUN at B=250** per gene,
   or 2.51x and 4.59x per relaxation spent. Report both denominators and
   describe the step as training-set deduplication, which is what it is.
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

## Follow-up

The second lever over this same pool -- the generator's own likelihood as a
novelty estimator -- is measured in
[the `e9ywwsie` generative novelty report](e9ywwsie_generative_novelty_report.md).
It does not overturn anything here. It does qualify one conclusion: a learned
novelty *predictor* has no headroom against the free lookup, as found below, but
the likelihood used the other way round -- as a plausibility prior over the genes
the lookup has already called novel -- raises MetaSUN from 2.78x to 2.93x at
B=250.

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
