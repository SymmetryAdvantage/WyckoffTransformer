# Composition screening: what was measured

> **STATUS (2026-09-07): the generated-structure screening result is invalid and
> must be rerun.** `formula_energy.screen.HullLookup` used pymatgen's absolute hull
> energy as if it were formation energy for formulas absent from the table. The
> helper now subtracts the elemental-reference contribution and is tested with
> nonzero elemental energies. Formula-table model comparisons and descriptive
> archive statistics below were not computed through that path; the novel-formula
> MetaSUN enrichment was and must not be cited. Nothing has been submitted to a
> leaderboard.

The question was whether predicting, for a chemical formula, the lowest energy
any structure with that formula can have -- `f*(X)` -- is accurate enough to
decide where to spend a structure search. The short answer is yes, worth about a
factor of two, and most of the work went into finding out which of the
measurements saying so could be trusted.

## Summary

**What remains plausible but unmeasured.** Screening compositions before relaxing
them may improve MetaSUN at a fixed relaxation budget, but the reported **2.23x
[1.58, 3.05]** top-decile result used the wrong hull-energy scale for novel
formulas and is invalid. The campaign must be rescored or rerun.

**What did not.** The two-regression scheme the work started from is the weakest
thing tested. The censored likelihood, which replaced it, does not clearly beat
plain MSE. Neighbourhood density improves the likelihood and changes screening by
nothing. Stability and SUN could not be measured at all at these sample sizes.

**What nearly went wrong.** Three measurement errors, each of which would have
produced a confident wrong answer: a label-noise constant carried over from
another problem, two energy scales silently compared against each other, and an
evaluation that rewards predicting where a database campaign looked rather than
where low-energy structures are. Two were caught by diagnostics that were only in
place because something looked impossible; the third was caught by a control that
was only run because a feature seemed to work too well.

## The archive, before any model

LeMat-Bulk, 4,745,121 usable structures over 2,329,360 reduced formulas, with
Materials Project provenance pulled for the 138,931 `mp-` ids (45,274 turn out to
be ICSD-backed).

| | |
|---|---|
| formulas with exactly one entry | **66.6%** |
| formulas holding an ICSD-backed entry | 37,490 (1.61%) |
| formulas defining the hull | 3.67% |

**"The observed structure is the ground state" is usable but not exact.** The
ICSD-backed entry is the archive's own minimum **55.4%** of the time; the mean
excess above the minimum is **17.3 meV/atom**, and only **4.4%** of formulas are
beaten by more than 50 meV/atom. The assumption is right about the location and
wrong about the sharpness, which is why it belongs in a likelihood as a narrow
excess rather than as an exact observation.

**The experimental flag carries information a row count does not.** Paired within
formula, on formulas holding both kinds of Materials Project entry, the
ICSD-backed one is the archive minimum **56.9%** of the time against **11.4%** for
the theoretical one, and is strictly the lower of the two 81.9% of the time. That
survives stratification by how many entries the formula has, so it is not a proxy
for effort.

**Censoring is deep and unevenly distributed.** Adding Alexandria to Materials
Project and OQMD lowered the minimum by more than 50 meV/atom for **36.3%** of the
formulas the other two had already computed -- but for only **3.3%** of formulas
holding an ICSD-backed entry. The bound is tight where the experimental record
reaches and loose everywhere else.

**Most of the hull is hypothetical.** Only about an eighth of hull-defining
formulas hold an experimentally observed structure, which caps what the record can
be used to falsify.

## Three measurement errors

### The force cut is a provenance filter

`max_force <= 0.02`, inherited from the existing pipeline, keeps **95.6%** of
Alexandria rows and **35.5%** of the ICSD-backed ones, because Materials Project
reports forces from a different protocol and its median is 0.028. It is also
inconsistent with the hull it would be measured against: the hull table
applies no such filter, so **17.9%** of the entries defining the deep hull and
**52.2%** of the shallow one are above the cut.

It buys nothing. Three measurements, because the obvious one is not sufficient --
a hull is a *minimum*, and a rare anomalously low entry would be selected onto it
however it behaves on average:

* Differencing each row against its own formula's median energy, a large force
  goes with a **higher** energy: mean +0.34 eV/atom in the `>0.5` bucket against
  +0.02 below 0.02. An unconverged relaxation has not reached the minimum yet.
* A high-force entry wins its own formula at exactly the base rate -- 10.94% of
  per-formula minima against 10.71% of all rows, an enrichment of **1.02x**.
* When one does win, it wins by a hair: median gap to the runner-up **0.5
  meV/atom** for winners between 0.05 and 0.1, against 88 meV for well-converged
  winners. A spurious minimum would win by a wide margin.
* Excluding everything above 0.1 moves the per-formula minimum by a median of
  **exactly zero** (mean 0.19 meV; 0.10% of formulas move by more than 10 meV).

Dropping the cut recovers 134k formulas and takes the ICSD-backed set from 13,836
formulas to **37,490**. Corrupt energies are caught by a `|e_form| <= 5 eV/atom`
window instead, which is the cleaning Wren applies.

### Two energy scales, silently compared

Withholding Alexandria to build the evaluation raises 25 elemental references, by
up to **36 meV/atom for bromine**, 26 for iodine, 23 for silver -- and a formation
energy is measured against them. The symptom was a median `drop` of **-0.4
meV/atom**, impossible when one archive contains the other. Translating the deep
minimum onto the shallow references is a per-composition constant, so it cannot
change which polymorph is lowest, and it was worth a third of the answer key:

| | uncorrected | corrected |
|---|---|---|
| discoveries | 16,750 (5.85%) | **26,068 (9.10%)** |
| median drop | -0.0004 | 0.0000 |
| formulas with an impossible negative drop | present | none |

### A constant carried over from another problem

`censored.DEFAULT_NOISE = 0.01` is right at the gene level, where one source's
energies are compared, and wrong over a three-database corpus. It enters as
`t = (observed - location) / noise`, so against formation energies spanning
several eV it makes `t` about 500 at initialisation; the model compensated by
inflating the excess scale to 0.495 eV/atom and drove the floor below the data
almost everywhere, flagging 77% of formulas.

| noise | MAE vs deep min | violation | flagged | enrichment @100 | calibration error |
|---|---|---|---|---|---|
| 0.01 | 0.410 | 0.022 | 73% | 1.10 | 0.659 |
| **0.10** | **0.169** | 0.289 | 25% | **2.65** | **0.159** |

The first version of the model comparison was run at 0.01 and had to be discarded.

## Model comparison, and why to discount it

Trained on the shallow world, scored on 14,096 held-out formulas, prevalence 9.06%.

| model | @100 | @500 | @1000 | @5000 | MAE vs deep min |
|---|---|---|---|---|---|
| g_D (MSE, all formulas) | **3.31** | **2.91** | 2.32 | 1.84 | 0.154 |
| censored ensemble | 2.65 | 2.36 | **2.36** | **1.87** | **0.151** |
| Magpie + gradient boosting | 1.21 | 1.63 | 1.43 | 1.47 | 0.184 |
| chemical-system lookup | 0.99 | 0.40 | 0.40 | 0.95 | 0.429 |
| g_C | 0.77 | 0.64 | 0.68 | 1.08 | 0.299 |
| **g_D - g_C** | 0.44 | 0.71 | 0.53 | 0.36 | -- |

**The two-regression scheme does not work.** `g_D - g_C` is the weakest thing
tested, below a training-free chemical-system lookup at three of four budgets, and
`g_C` is next weakest. `g_C` is fit on the 11.85% of formulas whose observed
structure is the archive's own minimum and asked to extrapolate to formulas
selected for being unlike them. The difference of two extrapolations carries no
usable signal.

**Neither likelihood dominates.** Plain MSE is sharper at the very top of the
list; the censored ensemble is ahead from a thousand candidates on, with slightly
better point accuracy and a calibration error of 0.159. At comparable precision
the difference is recall: the uncertainty-adjusted rule gives g_D 0.254 precision
on 142 formulas and the censored model 0.214 on 1,037 -- six times the finds for
four points of precision.

**Wren's uncertainty adjustment holds up**, lifting g_D from 2.46 to 2.80 and the
censored model from 2.11 to 2.36 by flagging far less.

### The evaluation is partly a search-policy detector

Ranking formulas by **how many hull-defining entries their chemical system already
holds** -- one integer, no model, no training -- scores 2.98 / 3.22 / 3.06 / 2.36
at those four budgets, beating every model above from 500 candidates onward.

It is worth **nothing** on real generated structures (1.16 / 0.98 / 1.00).

Alexandria is a substitution campaign: it expands around structures that already
exist, so system density predicts **where it looked** rather than where low-energy
structures are. This is the dataset-builder problem appearing in the *evaluation*
rather than the training data, and it means every enrichment figure in the table
above is inflated by an unknown amount. The generated-structure test below is the
primary instrument.

## What screening appeared to buy a generation run — invalid pending rerun

> The numbers in this section used absolute phase-diagram hull energies for novel
> formulas while the model predicted formation energies. They are retained as an
> audit trail, not as evidence.

Sampling 1,000 genes from the finished `upi73i4k` checkpoint, relaxing all of them,
then asking whether the top slice by screener score is richer than the whole run.
Same structures, only the order varies.

The generator itself improved with training, which is the baseline the screening
has to beat:

| | earlier checkpoint (2,500) | finished (1,000) |
|---|---|---|
| structure rate | 0.996 | 0.995 |
| valid / novel structure | 0.908 / 0.722 | 0.891 / 0.708 |
| **MetaSUN** | 0.116 | **0.151** |
| **SUN** | 0.0028 | **0.006** |

Restricted to **novel** formulas -- what MetaSUN counts, and where a
structure-search budget should go:

| slice | MetaSUN enrichment (2,500-gene run, n=1,209, base 10.1%) |
|---|---|
| top 10% | **2.23 [1.58, 3.05]** |
| top 25% | 1.64 [1.27, 2.10] |
| top 50% | 1.26 [1.02, 1.55] |

The 1,000-gene run points the same way (1.53 at the top decile) but is not
individually significant at 453 novel formulas. The two are statistically
compatible.

**Over all genes the numbers look much better and should be ignored.** The top 500
are 100% compositions the archive already holds, and those are metastable 43.0% of
the time against 16.3% for novel ones, so sorting on membership alone is worth
1.39x before any model runs. Within known formulas the screener's ordering is
still worth 1.61x [1.38, 1.80] -- real signal, but not where discovery happens,
and MetaSUN excludes those structures by definition.

**Stability and SUN cannot be measured here.** The novel subsets hold three stable
structures each. Measuring SUN enrichment needs of order 10^4 genes.

### Operationally

Generation costs 22 s per 1,000 genes; relaxation costs about 10 minutes per 1,000
on 20 GPU workers. That asymmetry is the opportunity: **generate more, relax the
top slice.** Relaxing 1,000 novel genes at random yields about 101 MetaSUN;
generating 10,000 and relaxing the best 1,000 projects to about 220 -- roughly
double, for four extra minutes of sampling.

Two limits. The top-decile enrichment was measured within a 1,209-gene pool, so
"top 10% of 10,000" is a sharper threshold on a larger pool and is an
extrapolation. And enrichment decays quickly with slice size -- 2.2x at 10%, 1.26x
at 50% -- so the gain depends on oversampling hard.

## Neighbourhood density: a null result in three arms

Two thirds of formulas hold a single entry, so their own provenance counts say
almost nothing, and a never-computed formula has none at all -- every effort
channel is zero exactly where screening happens. A system-level count is the
obvious gap.

**Arity has to be controlled combinatorially, and that turns out to be
load-bearing.** A system of `a` elements contains exactly `C(a, k)` subsystems of
size `k`, so a raw total grows mechanically:

| feature | Spearman with arity | MetaSUN @10%, ranking by it alone |
|---|---|---|
| raw subsystem total | +0.72 | -- |
| exact-system count | -0.74 | 1.00 |
| entries per binary subsystem, `n2/C(a,2)` | **-0.07** | 1.90 [1.30, 2.69] |
| entries per ternary subsystem, `n3/C(a,3)` | **+0.07** | **2.39 [1.72, 3.23]** |

The uncontrolled version is worth nothing; the controlled one matches the entire
ten-model ensemble with a single integer.

Given to the model, it changes nothing, wherever it is put:

| arm | val NLL | MetaSUN @10% / @25% / @50% |
|---|---|---|
| density nowhere | -0.641 | 2.23 / 1.64 / 1.26 |
| density in the excess-scale head | **-0.703** | 2.23 / 1.61 / 1.26 |
| density in the location head | -0.586 | 2.23 / 1.64 / 1.23 |

The likelihood improves in the scale head, so the premise -- well-explored systems
carry tighter bounds -- is correct; the share of formulas whose bound moved by
more than 50 meV falls monotonically from 24.2% to 13.2% across density quintiles.
But the scale head's output never enters `score = location + sigma - hull`, and
letting the floor read the densities instead (a scoped relaxation of the exclusion
restriction) fits *worse* and screens the same.

**The model already had the information.** The control, which never sees these
features, produces a score correlating **-0.42** with entries-per-binary and -0.26
with entries-per-ternary on novel generated formulas: it infers how populated a
neighbourhood is from the element identities, which a learned element embedding
over 2.3M formulas is well placed to do. Combining the two rankings directly does
not help and flips sign between the two generation runs.

The standalone 2.39 is best read as the maximum of four features across two runs;
the same feature gives 1.37 on the smaller one. What survives is weak and
consistent rather than strong: eight positive Spearman correlations with the
MetaSUN outcome, +0.005 to +0.117.

## One environment trap

A `wyformer-protocol` run of 1,000 genes produced **2 structures** because
`libomp.so.5` was not on the library path. `magma` needs it, ORB's conservative
regressor calls `torch.linalg.det` for the stress, and every CUDA worker died
about 1.4 s in. The failure is quiet: every gene still gets a PyXtal cif and an
`_0_initial.cif`, one worker happened to fall back to CPU and succeeded, and the
run exited 0 reporting `metastable: 0`. Read naively that says the model produces
nothing.

If a GPU protocol run reports a structure rate far below the ~0.99 it should have,
check `ldconfig -p | grep libomp` before suspecting the model.

## What would change the answer

* **SUN at 10^4 genes.** The only outcome that matters for the leaderboard is the
  one this could not measure.
* **An evaluation that is not a substitution campaign.** The shallow-hull answer
  key rewards predicting where Alexandria looked. A temporal split on Materials
  Project snapshots, or DFT validation of a shortlist, would not.
* **Composing with the gene-level critic.** This chooses *which formula*; the
  critic in [the gene energy critic study](gene_energy_critic_study.md) chooses
  *which structure for that formula*. They multiply, and only one of them has been
  measured.
* **The displacement bound at scale.** `L(X)` is implemented and unit-tested
  against hand geometry but has never been run on a real shortlist.

## Reproduce

```bash
uv sync --extra mp
uv run python scripts/pull_mp_provenance.py                          # ~6 min, needs MP_API_KEY
uv run python -m wyckoff_transformer.formula_energy.dataset          # ~2 min
uv run python -m wyckoff_transformer.formula_energy.answer_key --workers 24
uv run python -m wyckoff_transformer.formula_energy.experiment       # the model comparison
uv run python -m wyckoff_transformer.formula_energy.train \
    --config yamls/models/formula_energy/censored_floor.yaml --device cuda
uv run python -m wyckoff_transformer.formula_energy.prefilter \
    generated/<run>/protocol/structures.csv --ensemble runs/formula_energy/ensemble.pt \
    --genes generated/<run>/wyckoff_genes.json.gz
CUDA_VISIBLE_DEVICES="" uv run pytest src/wyckoff_transformer/formula_energy   # 103 tests
```

GPU protocol runs need the `libomp` shim described above, on
`LD_LIBRARY_PATH`, or the relaxation stage fails silently.
