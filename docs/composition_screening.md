# Composition screening: estimating the floor under a chemical formula

> **STATUS (2026-09-07): implemented, tested, and measured once.** 103 tests
> pass. This is the design; the measurements are in
> [what was measured](composition_screening_results.md), and they are single runs
> on one split. Nothing has been submitted anywhere and the displacement bound has
> never been run at scale.

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

## Results

All measurements, and the three that had to be discarded and redone, are in
[what was measured](composition_screening_results.md). In brief: screening
roughly doubles the MetaSUN rate at fixed relaxation budget (2.23x in the top
decile of novel formulas); the two-regression scheme this replaced is the weakest
thing tested; the censored likelihood does not clearly beat plain MSE; and the
shallow-hull answer key turns out to reward predicting where Alexandria looked, so
the generated-structure test is the primary instrument.

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

- [What was measured](composition_screening_results.md) -- the results log,
  including the three measurement errors that had to be caught first.
- [CSP mode](csp_mode.md) -- the gene-level censored regressor this generalises,
  and the consumer of a shortlist.
- [Gene energy critic study](gene_energy_critic_study.md) -- "The exploration
  paradox", which asked for a formula policy allowed to disagree with database
  frequency. This is that policy.
