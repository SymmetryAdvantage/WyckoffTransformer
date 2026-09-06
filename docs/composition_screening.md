# Composition screening: estimating the floor under a chemical formula

> **STATUS (2026-09-05): implemented, tested, and run once.** 80 tests pass. The
> comparison below is a single run against the shallow-hull answer key, on a
> formula-level split, with no hyperparameter search beyond the one sweep the
> label-noise constant forced. Nothing has been submitted anywhere and no
> displacement filter has been run at scale.

## The question

For a chemical formula `X`, let `f*(X)` be the lowest formation energy any
structure with that composition can have. If we knew `f*` we could screen: compare
it against the hull at that composition and read off where the hull can be
lowered.

`f*` is never observed. Every entry in every archive is a structure that somebody
found, so its energy is an *upper bound* on the floor, loose by an amount nobody
recorded. Regressing the archive minimum estimates the bound, not the floor, and
the bound's looseness is a property of how hard people looked -- which is
entangled with the very thing being predicted, because a formula gets computed
when somebody expects it to be interesting.

## What the data says before any model is fitted

All from `data/formula_energy/formula_table.parquet`, built by
`python -m wyckoff_transformer.formula_energy.dataset`.

| | |
|---|---|
| structures / reduced formulas | 4,745,121 / 2,329,360 |
| formulas with exactly one entry | 66.6% |
| formulas holding an ICSD-backed entry | 37,490 (1.61%) |
| formulas that define the hull | 3.67% |

**Assumption C -- "the experimentally observed structure is the ground state" --
is usable but not exact.** The ICSD-backed entry is the archive's own minimum
55.4% of the time; the mean excess above the minimum is 17.3 meV/atom, and only
4.4% of formulas are beaten by more than 50 meV/atom. So the assumption is right
about the location and wrong about the sharpness: it belongs in the likelihood as
a *narrow* excess, not as an exact observation.

**The experimental flag carries information that a row count does not.** Paired
within formula, on formulas holding both kinds of Materials Project entry, the
ICSD-backed one is the archive minimum 56.9% of the time against 11.4% for the
theoretical one, and 81.9% of the time it is strictly the lower of the two. That
survives stratification by `n(X)`, so the flag is not a proxy for effort.

**Censoring is deep and unevenly distributed.** Adding Alexandria to Materials
Project and OQMD lowered the minimum by more than 50 meV/atom for 36.3% of the
formulas the latter two had already computed -- but for only 3.3% of the formulas
holding an ICSD-backed entry. The bound is tight exactly where the experimental
record reaches and loose everywhere else.

**Most of the hull is hypothetical.** Of the hull-defining formulas, only about an
eighth hold an experimentally observed structure. Whatever the record can falsify,
it can only falsify there.

### One filter had to go

`scripts/pipeline_lemat_20wyckoffs.py` cuts at `max_force <= 0.02`, and this
package does not. That cut keeps 95.6% of Alexandria rows and 35.5% of the
ICSD-backed ones, because Materials Project reports forces from a different
protocol and its median is 0.028 -- so it is a provenance filter wearing a
convergence costume, and provenance is the scarcest signal here.

It is also inconsistent with the hull it would be measured against.
`scripts/compute_e_hull.py` applies no force filter, so **17.9% of the entries
defining the deep hull, and 52.2% of those defining the shallow one, are above
0.02** -- the shallow hull is dominated by Materials Project, which is where the
high forces are. Filtering the training rows while comparing them to an unfiltered
hull would put the labels and the threshold on different footings.

And it is unnecessary, which took three measurements to establish because the
obvious one is not sufficient. Differencing each row against its own formula's
median energy, a large force goes with a *higher* energy, not a lower one: mean
+0.34 eV/atom in the `>0.5` bucket against +0.02 below 0.02. That is an average,
and a hull is a *minimum*, so it does not settle the question on its own -- a rare
anomalously low entry would be selected onto the hull however it behaves on
average. Two more say it does not happen:

* A high-force entry wins its own formula at exactly the base rate -- 10.94% of
  per-formula minima against 10.71% of all rows, an enrichment of **1.02x**. (The
  17.9% above is a provenance effect: hull-defining formulas are concentrated in
  well-studied chemistry, which is Materials Project, which reports higher forces.)
* When one does win, it wins by a hair. Median gap to the runner-up is 88 meV/atom
  when the winner is below 0.02 and **0.5 meV/atom** when it is between 0.05 and
  0.1 -- these are marginally different relaxations of the same structure, not
  anomalies. A spurious minimum would win by a wide margin.
* Excluding everything above 0.1 raises the per-formula minimum by a median of
  **0.00000 eV/atom** (mean 0.19 meV; 0.10% of formulas move by more than 10 meV).

Dropping the cut recovers 134k formulas and takes the ICSD-backed set from 13,836
formulas to 37,490. Corrupt energies are caught by a `|e_form| <= 5 eV/atom`
window instead, which is the cleaning Wren applies, and `max_force` survives as a
covariate on the excess-scale head.

## The model

`f*` is estimated with the censored likelihood in
`wyckoff_transformer.censored` -- the same one written for `min(E | gene)`,
unchanged, one level up. An observation is `f*(X) + Exponential(s) + Normal(0, σ)`,
so the archive minimum is fitted *as a bound* and the floor is what sits under it.

The network is the CrabNet shape: learned element embeddings, a sinusoidal
encoding of stoichiometric fraction on linear and logarithmic scales, three
transformer layers over the element multiset, attention pooling. It is written in
`formula_energy/encoder.py` rather than imported, for two reasons. The 2026
literature says the encoder is not where the accuracy is -- feeding CrabNet and
Roost embeddings to an in-context foundation model gave "only marginal or
inconsistent improvements ... suggesting that their attention-based encoders
already saturate the accessible compositional information" (npj Comput Mater
2026), and the cross-modal transfer work of the same year only *approaches*
CrabNet on composition using pretraining aimed at small data, which 2.3M formulas
is not. And no published composition model has a censored head, so the training
loop had to be written regardless.

**The two heads are the design.** Both read a trunk built from chemistry alone.
Only the excess-scale head additionally reads provenance -- the per-process counts
`(n_mp_icsd, n_mp_theoretical, n_oqmd, n_agm)`, the ICSD flag, cell-size diversity,
force statistics. So the estimate of the floor is a function of the elements and
their proportions and of nothing else, while the width of the excess above it is
free to depend on who looked and how hard.

That is an exclusion restriction expressed as an architecture, and it is enforced
by a test: change the provenance vector arbitrarily, and the location output must
not move. Without it the model learns that singleton formulas have high energy --
true of this archive, where two thirds of formulas are one-shot substitution
products, and exactly backwards for a formula nobody has tried.

The per-process counts are a vector rather than a scalar `n(X)` because the
sources are different search processes, not different amounts of one: Materials
Project is ICSD-seeded plus targeted studies, OQMD is a prototype library,
Alexandria is mass substitution. A substitution campaign's minimum converges to
the best structure *in its prototype library*, not to `f*`, so its excess does not
vanish as `n` grows.

Ten models from different initialisations, which is Wren's protocol. A point
estimate cannot be screened on: ranking millions of candidates by one selects the
largest positive errors. The ensemble's disagreement about where the floor lies
is what turns the ranking into `P(f*(X) < E_hull(X))` and lets the triage rule ask
for a margin.

## The evaluation

`formula_energy/answer_key.py`. Hold out Alexandria, rebuild the convex hull from
Materials Project and OQMD alone, train on shallow labels only, then ask which
formulas the full archive later put below that shallow hull. Those are real
discoveries relative to the shallow world's state of knowledge, and a screener
that could not have predicted them would not predict the outstanding ones either.
It is Matbench Discovery's temporal logic without needing dates.

The hull has to be genuinely recomputed rather than filtered: `e_hull` in the
archive is measured against the deep hull *and clipped at zero*, so a structure
below the shallow hull -- the case of interest -- cannot be recognised by
subsetting rows. `scripts/compute_e_hull.py` rebuilds the phase diagrams.

The two worlds also have to be put on one energy scale, and this is not a
formality. A formation energy is measured against elemental references, and those
are the lowest elemental entries *in that world*: withholding Alexandria raises
25 of them, by up to 36 meV/atom for bromine, 26 for iodine and 23 for silver.
Left uncorrected the comparison mixes scales, and it announces itself -- the raw
`drop` had a median of **-0.4 meV/atom**, which is impossible when one archive
contains the other. Translating the deep minimum onto the shallow references (a
per-composition constant, so it cannot change which polymorph is lowest) fixes it,
and the correction is worth a third of the answer key:

| | uncorrected | corrected |
|---|---|---|
| discoveries | 16,750 (5.85%) | **26,068 (9.10%)** |
| median drop | -0.0004 | 0.0000 |
| formulas with a negative drop | present | 0.0000 |

The key covers 286,348 formulas, 190,000-odd of which the shallow world had
already computed more than one structure for, and the positive rate is even across
the three splits (9.12 / 8.94 / 9.06%), so nothing is leaking through the
formula-level hash.

The key covers formulas the shallow world already knew about. A formula with no
shallow entry is a harder question with no observable answer, and is left to the
screening run rather than smuggled into the evaluation.

`formula_energy/experiment.py` scores five models on that key: the censored
ensemble; `g_D`, the same encoder under MSE on every formula; `g_C`, the same
encoder under MSE on the formulas whose observed structure is the archive's own
minimum; `g_D - g_C`, the headroom signal those two were proposed to produce; and
Magpie descriptors under gradient boosting, against a chemical-system lookup.
`g_C` and `g_D` differ from the censored model in the objective and nothing else,
so the comparison isolates the likelihood.

Each is scored under the naive rule `location < hull` and under Wren's
uncertainty-adjusted `location + sigma < hull`, which lifted precision from 38% to
53% and enrichment from 2.5 to 3.5 in the only published prospective campaign.

## The lower bound

Everything above estimates where the floor probably is. `formula_energy/screen.py`
adds the only bound pointing the other way. If a compound at composition `X`
really had energy `e`, the hull would move, and some compound people have actually
made might be pushed above it -- which is evidence against `e`, since things that
exist tend not to be unstable.

On a hand-checkable binary the bound is exact: with Na and Cl at zero and NaCl at
-1 eV/atom, a candidate at NaCl3 sits at `x_Cl = 0.75`, the tie-line from Na
through it crosses `x_Cl = 0.5` at two thirds of its depth, and NaCl comes off the
hull exactly when the candidate falls below -1.5 eV/atom. `displacement_bound`
returns -1.501.

The displacement condition is affine in `e`, but which simplex binds depends on
`e`, so `L(X)` is found by bisection on a predicate that is monotone in it:
lowering a point can only lower the hull, so once an entry is displaced it stays
displaced. Roughly thirty phase diagrams per candidate. That is affordable on a
shortlist and not on 2.3M formulas, which is the right shape -- with only an
eighth of hull-defining formulas experimentally backed, this can only ever be a
post-filter.

## Results, one run

Trained on the shallow world, scored on the 14,095 test formulas of the
answer key. Prevalence -- the share that Alexandria later put below the shallow
hull -- is 9.06%, so an enrichment of 1.0 is a coin flip.

**Enrichment among the top *k*, ranked:**

| model | @100 | @500 | @1000 | @5000 |
|---|---|---|---|---|
| g_D (MSE, all formulas) | **3.31** | **2.91** | 2.32 | 1.84 |
| **censored** | 2.65 | 2.36 | **2.36** | **1.87** |
| magpie + GBDT | 1.21 | 1.63 | 1.43 | 1.47 |
| chemical-system mean | 0.99 | 0.40 | 0.40 | 0.95 |
| g_C | 0.77 | 0.64 | 0.68 | 1.08 |
| g_D - g_C | 0.44 | 0.71 | 0.53 | 0.36 |

**Triage rules:**

| model | rule | flagged | precision | recall | enrichment | MAE vs deep min |
|---|---|---|---|---|---|---|
| g_D | uncertainty-adjusted | 142 | 0.254 | 0.028 | **2.80** | 0.154 |
| g_D | naive | 785 | 0.223 | 0.137 | 2.46 | 0.154 |
| censored | uncertainty-adjusted | 1,037 | 0.214 | **0.174** | 2.36 | **0.151** |
| censored | naive | 2,674 | 0.191 | 0.399 | 2.11 | 0.151 |
| magpie + GBDT | naive | 1,362 | 0.136 | 0.145 | 1.50 | 0.184 |
| g_C | naive | 8,270 | 0.105 | 0.679 | 1.16 | 0.299 |
| chemical-system mean | naive | 3,410 | 0.074 | 0.199 | 0.82 | 0.429 |

Expected calibration error of the censored ensemble's `P(f* < E_hull)`: **0.159**.

Read it as three findings.

**Neither likelihood dominates, and which is better depends on the budget.** g_D is
sharper at the very top of the list (3.31 against 2.65 at a hundred candidates);
the censored ensemble catches up by a thousand and stays ahead thereafter, with
slightly better point accuracy (0.151 against 0.154 eV/atom). At comparable
precision the difference is recall: the uncertainty-adjusted rule gives g_D 0.254
precision on 142 formulas and the censored model 0.214 on 1,037 -- six times as
many finds for four points of precision. For a campaign that can afford more than
a hundred structure searches, that is the trade worth making.

**Wren's uncertainty adjustment holds up.** It lifts g_D from 2.46 to 2.80 and the
censored model from 2.11 to 2.36, in both cases by flagging far less.

**The two-regression scheme does not work, and the failure is where the support
argument said it would be.** `g_D - g_C` is the weakest thing tested -- below the
training-free chemical-system lookup at three of four budgets -- and `g_C` is next
weakest. `g_C` is fit on the 11.85% of shallow formulas whose observed structure
is the archive's own minimum, and asked to extrapolate to formulas selected for
being unlike them. The difference of two extrapolations carries no usable signal.

### The label-noise constant is not a detail

The first run of this comparison put the censored model at 1.10 enrichment, below
Magpie. The cause was `noise = 0.01`, inherited from `censored.DEFAULT_NOISE`,
which is right at the gene level where one source's energies are compared and
wrong here. It enters as `t = (observed - location) / noise`, so against formation
energies spanning several eV it makes `t` about 500 at initialisation; the model
compensated by inflating the excess scale to 0.495 eV/atom and drove the floor
below the data almost everywhere, flagging 77% of formulas. The sweep is recorded
in `TrainConfig.noise`. At 0.10 the calibration error falls from 0.659 to 0.159.

## The answer key is partly a search-policy detector

Ranking formulas by **how many hull-defining entries their chemical system already
holds** -- one integer per system, no model, no training -- scores 2.98 / 3.22 /
3.06 / 2.36 at budgets of 100 / 500 / 1000 / 5000 on the answer key. That beats
every model above from 500 candidates onward.

It is worth nothing on real generated structures: 1.16 / 0.98 / 1.00 MetaSUN
enrichment on the WyFormer runs below.

The explanation is that Alexandria is a substitution campaign, so it expands
around structures that already exist, and system density predicts **where it
looked** rather than where low-energy structures are. This is the dataset-builder
problem appearing in the *evaluation* rather than the training data. The
enrichment figures in the previous section are therefore inflated by an unknown
amount for the same reason, and the generated-structure test below -- whose
outcome is an ORB relaxation rather than a campaign's choice -- is the primary
instrument.

## What screening buys a generation run

`prefilter.py` scores a protocol run retrospectively: every gene is relaxed, and
the question is whether the top slice by screener score is richer than the whole
run. Restricted to **novel** formulas, which is what MetaSUN counts and where a
structure-search budget should go. Over all genes the numbers look far better
(2.6x metastable) but the top 500 are 100% compositions the archive already holds,
and those are metastable 43.0% of the time against 16.3% for novel ones, so
sorting on membership alone is worth 1.39x before any model runs.

2,500 genes, 1,209 novel formulas, MetaSUN base 10.1%:

| slice | MetaSUN enrichment |
|---|---|
| top 10% | **2.23 [1.58, 3.05]** |
| top 25% | 1.64 [1.27, 2.10] |
| top 50% | 1.26 [1.02, 1.55] |

The 1,000-gene run from the finished checkpoint points the same way (1.53 at the
top decile) but is not individually significant at 453 novel formulas. Stability
and SUN cannot be measured at either size -- the novel subsets hold three stable
structures each.

Operationally: generation costs 22 s per 1,000 genes and relaxation ~10 minutes,
so the move is to generate more and relax the top slice. Relaxing 1,000 novel
genes at random yields about 101 MetaSUN; generating 10,000 and relaxing the best
1,000 projects to about 220, for four extra minutes of sampling. The extrapolation
to a sharper threshold on a larger pool is not itself measured.

### Neighbourhood density: right idea, wrong head

A system-level covariate is the obvious gap -- two thirds of formulas have one
entry, so their own counts say almost nothing, and a never-computed composition
has none at all. Arity has to be controlled combinatorially: a system of `a`
elements holds exactly `C(a, k)` subsystems of size `k`, so a raw subsystem count
correlates +0.72 with arity and the exact-system count -0.74, while dividing by
`C(a, k)` brings both to -0.07 and +0.07.

Controlled that way it works, and the uncontrolled version does not: ranking
generated structures by entries-per-ternary-subsystem alone gives MetaSUN
enrichment **2.39 [1.72, 3.23]**, matching the whole ten-model ensemble, where the
exact-system count gives 1.00.

Added to the excess-scale head it improves the likelihood -- validation NLL
**-0.703 against -0.641** -- and changes screening by nothing at all: 2.23 / 1.61 /
1.26 against 2.23 / 1.64 / 1.26, with the top decile and top half identical rather
than merely close.

The scale head's output never enters `score = location + sigma - hull`, so its
only route to better ranking is indirectly freeing the location head. A third arm
therefore let the four densities into the *location* head as well -- a scoped
relaxation of the exclusion restriction, four named columns reaching the floor
while the other ten stay invisible, enforced by a test.

It changes nothing either. All three arms give MetaSUN **2.23** at the top decile
of novel formulas, and sit inside each other's intervals everywhere else:

| arm | val NLL | MetaSUN @10% / @25% / @50% |
|---|---|---|
| density nowhere | -0.641 | 2.23 / 1.64 / 1.26 |
| density in the scale head | **-0.703** | 2.23 / 1.61 / 1.26 |
| density in the location head | -0.586 | 2.23 / 1.64 / 1.23 |

The reason is that the model already has the information. The **control** model,
which never sees these features, produces a score correlating **-0.42** with
entries-per-binary and -0.26 with entries-per-ternary on novel generated formulas
-- it infers neighbourhood density from element identities alone, which a learned
element embedding over 2.3M formulas is well placed to do. Combining the two
rankings directly does not help either: rank-averaging gives 1.90 and 2.06 on one
run against 2.23 for the screener alone, and 1.83 against 1.53 on the other,
flipping direction between runs.

The standalone 2.39 for entries-per-ternary is best read as the maximum of four
features tested across two runs; it falls to 1.37 on the smaller run. What
survives is weak and consistent rather than strong: all eight Spearman
correlations with the MetaSUN outcome are positive, +0.005 to +0.117.

The conclusion is that neighbourhood density is real information which the
chemistry encoder already extracts, and the exclusion restriction was not costing
anything here. It stays.

## What exists

| | |
|---|---|
| `formula_energy/dataset.py` | formula table, provenance labelling, formula-level splits |
| `formula_energy/features.py` | chemistry and provenance tensors, kept apart |
| `formula_energy/encoder.py` | two-head CrabNet-shaped encoder |
| `formula_energy/train.py` | training loop, deep ensemble, MSE variant for the baselines |
| `formula_energy/metrics.py` | precision, enrichment, calibration |
| `formula_energy/baselines.py` | `g_C`, `g_D`, Magpie + GBDT, chemsys lookup |
| `formula_energy/answer_key.py` | shallow world and the discovery key |
| `formula_energy/experiment.py` | the comparison run |
| `formula_energy/screen.py` | ranking, `P(below hull)`, `L(X)`, `HullLookup` |
| `formula_energy/prefilter.py` | what screening buys a generation run |
| `cli/screen.py` | `wyformer-screen` |
| `scripts/pull_mp_provenance.py` | the ICSD flags LeMat-Bulk does not carry |

Not built: no enumeration of novel formulas (`smact` is a dependency and would be
the natural source), no empirical-Bayes shrinkage across the ranked list, and no
link from a shortlist into `wyformer-csp` beyond writing formulas it can read.

## Reproduce

```bash
uv sync --extra mp
uv run python scripts/pull_mp_provenance.py                        # ~6 min, needs MP_API_KEY
uv run python -m wyckoff_transformer.formula_energy.dataset        # ~1 min
uv run python -m wyckoff_transformer.formula_energy.answer_key --workers 24
uv run python -m wyckoff_transformer.formula_energy.experiment --quick   # check the wiring
uv run python -m wyckoff_transformer.formula_energy.experiment          # the real run
CUDA_VISIBLE_DEVICES="" uv run pytest src/wyckoff_transformer/formula_energy
```

## See also

- [CSP mode](csp_mode.md) -- the gene-level censored regressor this generalises,
  and the consumer of a shortlist.
- [Gene energy critic study](gene_energy_critic_study.md) -- "The exploration
  paradox", which asked for a formula policy allowed to disagree with database
  frequency. This is that policy.
