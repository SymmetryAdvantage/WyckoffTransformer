# Conditioning on the chemical system and the space group

**The idea under test.** Unconditional generation is the wrong instrument for
(M)SUN, because a model that imitates an archive (a) re-emits structures the
archive already holds and (b) spends its budget in well-explored chemical
systems whose hull is already occupied. Naming the chemical system — and
perhaps the space group — should therefore lift the yield, possibly a lot.

**What was done.** Nothing was generated. Every `protocol_eval` artifact in the
project was pooled into one per-gene table, each gene labelled with the chemical
system and space group it was drawn in and with what LeMat-Bulk already holds
there, and the pool was asked what a different sampling distribution over those
cells would have been worth.
[`scripts/analyse_chemsys_conditioning.py`](../scripts/analyse_chemsys_conditioning.py)
is the whole measurement; its JSON lands in `<work-dir>/results/`.

**Verdict (2026-09-20, commit `0e74564`).** The idea is half right, and the half
that is right is not the half it is argued from.

| the claim | verdict |
| --- | --- |
| the models spend their budget in well-explored systems | **false in fact.** 5.9% of the budget lands in systems LeMat-Bulk has never seen, 92% in systems holding 1–316 entries, 0.075% in systems holding ≥1000 |
| crowding costs novelty | **true, and the smaller half.** Novelty odds ×0.74 per decade of archive entries, ×0.87 once arity and cell geometry are held fixed |
| crowding costs (M)SUN, because the hull there is taken | **backwards for MetaSUN** (odds ×2.75 per decade, adjusted), **weakly true for SUN** (×0.59, CI to 1.00) |
| conditioning on the system would lift MetaSUN | **true — up to 2.7× cross-fitted — by steering *toward* worked chemistry, not away from it** |
| conditioning on the space group would lift yield | **true and, uniquely here, causally identified**: 1.17× on MetaSUN and 1.40× on SUN from re-weighting the start-token table alone. The only lever whose two signs agree |
| conditioning is itself the lever | **false.** Three runs already draw (system, space group) from the archive's own prior. They land in the same cells and yield less |

## The pool

Twelve `protocol_eval` artifacts, newest version of each, 1000 sampled genes per
run, 12 000 genes in all. Every one is scored against
`cache/lemat_bulk_fmax1_stress` (train+val+test), relaxed on `orb_conserv_inf`
against its own hull, at temperature 1.0 on the `0:1,2:2,*:3` trial schedule, so
they are directly comparable — the script refuses a pool whose artifacts disagree
on the novelty reference. All numbers below are the **free** readout (post-rattle,
symmetry released) per *sampled* gene, which reproduces each run's `funnel.json`
exactly.

| run | gene novelty | novel structure | metastable | MetaSUN | SUN |
| --- | --- | --- | --- | --- | --- |
| `ehull-ssops-20260904-235534` | 0.630 | 0.612 | 0.540 | 0.281 | 0.006 |
| `e9ywwsie` | 0.663 | 0.653 | 0.491 | 0.268 | 0.005 |
| `ehull5x-20260904-213346` | 0.556 | 0.579 | 0.576 | 0.255 | 0.012 |
| `upi73i4k` | 0.660 | 0.633 | 0.469 | 0.250 | 0.005 |
| `ehull_adamw_wsd_5x-20260912-115321` | 0.555 | 0.590 | 0.556 | 0.236 | 0.012 |
| `e_all_adamw_wsd-20260909-001225` | 0.689 | 0.603 | 0.405 | 0.206 | 0.002 |
| `chemsys_e_hull_adamw_wsd-20260915-004642` \* | 0.806 | 0.820 | 0.250 | 0.201 | 0.003 |
| `chemsys_e_hull_sg_adamw_wsd-20260916-201605` \* | 0.785 | 0.785 | 0.262 | 0.200 | 0.002 |
| `19qbxo6l` | 0.665 | 0.580 | 0.415 | 0.188 | 0.007 |
| `unconditional_5x-20260912-115308` | 0.562 | 0.742 | 0.270 | 0.183 | 0.006 |
| `chemsys_e_all_adamw_wsd-20260909-151748` \* | 0.725 | 0.719 | 0.255 | 0.170 | 0.001 |
| `relational_e_all_adamw_wsd-20260909-234259` | 0.685 | 0.636 | 0.356 | 0.169 | 0.006 |

\* chemical-system-conditioned; their cohorts are drawn from the system prior
rather than from the space-group marginal alone (§7).

Pooled: 7952 novel structures, 2607 MetaSUN, **67 SUN**. That last number governs
everything the pool can and cannot say, and it is returned to at the end.

## 1. The premise about where the budget lands is false

LeMat-Bulk is not a dense archive of a few systems. It holds 5.34M entries spread
over **673 173 chemical systems with a median of 2 entries each**; its densest
system, Ag-Cu, holds 4870. Only 3.5% of its rows sit in systems with ≥317 entries.

The models reproduce that distribution rather than concentrating inside it:

| entries LeMat-Bulk holds in the generated system | share of budget | gene novelty | novel structure | metastable | MetaSUN | SUN |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (system never seen) | 5.9% | 1.000 | 0.878 | 0.130 | 0.130 | 0.0014 |
| 1–31 | 37.5% | 0.683 | 0.670 | 0.348 | 0.178 | 0.0076 |
| 32–316 | 54.2% | 0.613 | 0.629 | 0.459 | 0.238 | 0.0046 |
| ≥317 | 2.4% | 0.732 | 0.770 | 0.715 | **0.564** | 0.0069 |

The gene-weighted median generated system holds 42 entries. Each cohort of 1000
genes touches 982–995 distinct chemical systems. The element-level reading of the
same premise fails too: the generated element marginal is the archive's to within
KL 0.026 nats, mildly over-weighting the lanthanides (Lu, Gd, Tb, Er at ×1.5) and
under-weighting Ac, Pu, W, Th, Re, Mo and Fe at ×0.45–0.65.

So "the model will hit the well-explored chemical system" is not what happens.
What happens is that the model hits the archive's long tail, because that is what
the archive is.

## 2. Novelty: the premise is true, and it is the smaller half

Crowding does cost novelty, as claimed. Per decade of LeMat-Bulk entries in the
system, with run fixed effects and a bootstrap clustered on chemical system:

| | novelty | metastability | MetaSUN | SUN |
| --- | --- | --- | --- | --- |
| odds ×/decade of system entries | 0.743 [0.707, 0.784] | 1.995 [1.879, 2.108] | 1.662 [1.557, 1.777] | 0.891 [0.650, 1.358] |
| the same, holding arity, positional DoF, cell size and space-group frequency fixed | 0.869 [0.796, 0.950] | 3.217 [2.926, 3.551] | **2.754 [2.448, 3.102]** | **0.591 [0.315, 1.000]** |

Novelty falls with crowding, significantly, in both specifications. It is simply
much the weaker of the two forces.

## 3. For MetaSUN the second premise is backwards

The hypothesis reads the crowded system as a place where a good structure has
nowhere to go. The pool says the opposite, because the binding constraint is not
the hull's position but the model's competence:

* `P(metastable | known structure) = 0.732` against `P(metastable | novel) = 0.328`;
  `P(stable | known) = 0.109` against `P(stable | novel) = 0.008`. Novelty is
  expensive in stability before any conditioning question is asked.
* Novel structures whose exact system **already holds a hull-defining entry** sit
  *closer* to the hull, not further: median `e_above_hull` 0.127 against 0.172
  eV/atom, `P(metastable)` 0.390 against 0.278, `P(stable)` 0.0120 against 0.0056.
* The median `e_above_hull` of a novel structure falls monotonically with
  crowding: 0.238 → 0.173 → 0.130 → 0.037 eV/atom across the four bins above.

Adjusting *strengthens* the effect (1.66 → 2.75 odds per decade), so it is not
an artefact of crowded systems holding smaller, easier cells — it survives inside
every positional-DoF bin and every cell-size bin, and inside arity 3 it runs
0.139 → 0.238 → 0.722 as crowding rises.

The mechanism is the obvious one once stated: the archive's depth in a system is
a measure of how much the model was taught about that chemistry. Where it was
taught little, it produces structures that are novel by default and 0.24 eV/atom
above the hull.

## 4. For SUN the second premise survives, barely

SUN is the one readout that moves the hypothesis's way. Adjusted for arity and
geometry, SUN odds fall by ×0.59 [0.315, **1.000**] per decade of crowding — the
interval's upper end is exactly the null. The element-level cut agrees in sign,
if not smoothly in between: SUN is 0.0086 in the quintile of genes whose rarest
element is rarest against 0.0039 in the commonest, while MetaSUN runs the other
way, 0.176 → 0.262.

This is the coherent picture: **metastability is a competence problem and
stability is a crowding problem**, and MetaSUN — the readout the ranking protocol
develops against — is dominated by the first.

## 5. What reallocating the budget would pay

A conditioning target is a cell a caller names in advance, and conditioning lets
the whole budget be spent inside one, so a cell's price is its own rate. To keep
that from being the maximum of a noisy table, cells are ranked on a random half
of the pool and read off the other, twenty times over.

| cell definition | held-out MetaSUN | vs pool | cell usually picked |
| --- | --- | --- | --- |
| system crowding | 0.566 | 2.61× | ≥317 entries |
| low vs high symmetry | 0.292 | 1.34× | sg < 75 |
| space group | 0.376 | 1.73× | 38 (`Amm2`) |
| crowding × symmetry | 0.644 | 2.97× | ≥317, sg < 75 |
| crowding × arity | 0.728 | 3.35× | ≥317, ternary |
| crowding × arity × symmetry | 0.729 | 3.36× | ≥317, ternary, sg < 75 |

The same procedure run on SUN picks a different cell every time and never beats
the pool outside noise: the best of the six is 1.74× with a 10th-to-90th-centile
spread of 0.000–0.019 against a base of 0.0056.

The finer-grained version replaces named strata with a score fitted from archive
statistics alone — entries, distinct formulas, hull and near-hull counts and the
best known `e_hull` of the system, the same for each of its elements, and the
space group with its training frequency. Nothing about the drawn gene, its
geometry or its relaxation enters. Cross-fitted five ways with the **folds split
by chemical system**, so no system is scored by a model that has seen it:

| target | base | system features only | space group only | both |
| --- | --- | --- | --- | --- |
| MetaSUN, top 10% of score | 0.217 | 0.515 (2.37×) | 0.333 (1.53×) | **0.582 (2.68×)** |
| MetaSUN, top 50% of score | 0.217 | 0.315 (1.45×) | 0.282 (1.30×) | 0.329 (1.52×) |
| novel structure, top 10% | 0.663 | 0.888 (1.34×) | 0.863 (1.30×) | 0.930 (1.40×) |
| metastable, top 10% | 0.404 | 0.858 (2.12×) | 0.493 (1.22×) | 0.873 (2.16×) |
| SUN, top 10% | 0.0056 | 0.0042 (0.75×) | 0.0050 (0.90×) | 0.0075 (1.34×) |

The two channels are complementary and do different jobs: **the chemical system
predicts metastability** (2.12× on its own, and it is where almost all of the
MetaSUN gain comes from), **the space group predicts novelty** (1.30× on its own,
and 1.22× on metastability against the system's 2.12×). Together they are worth 2.7× on MetaSUN at a tenth
of the budget and 1.5× at half of it. On SUN, nothing survives.

## 6. The space group is the cheaper and more robust lever

Generation draws the space group from the training marginal as its start token,
and that marginal is heavily loaded on exactly the groups that pay worst. The
three largest shares of the budget, and three mid-sized groups for contrast:

| space group | share of budget | novel | metastable | MetaSUN | SUN |
| --- | --- | --- | --- | --- | --- |
| 123 `P4/mmm` | 14.4% | 0.538 | 0.461 | 0.167 | 0.0041 |
| 216 `F-43m` | 9.8% | 0.459 | 0.456 | 0.111 | 0.0017 |
| 221 `Pm-3m` | 5.4% | 0.386 | 0.346 | 0.087 | 0.0031 |
| 12 `C2/m` | 7.5% | 0.796 | 0.397 | 0.310 | 0.0045 |
| 71 `Immm` | 2.1% | 0.769 | 0.418 | 0.311 | 0.0000 |
| 8 `Cm` | 1.4% | 0.835 | 0.347 | 0.294 | 0.0059 |

Nearly 30% of every cohort starts in the three groups at the top of that table,
whose MetaSUN is 0.09–0.17 against a pool average of 0.217.

Sorted by how common the group is in training rather than by symmetry, both
readouts fall together from the rarest quintile to the commonest — 0.244 → 0.167
on MetaSUN and 0.0100 → 0.0041 on SUN, neither perfectly monotonically. No other
cut in this study moves the two the same way:

| training-frequency quintile of the space group | median training rows | novel | metastable | MetaSUN | SUN |
| --- | --- | --- | --- | --- | --- |
| Q1 rarest | 21 068 | 0.787 | 0.346 | 0.244 | **0.0100** |
| Q2 | 67 436 | 0.743 | 0.393 | 0.258 | 0.0058 |
| Q3 | 174 432 | 0.586 | 0.407 | 0.190 | 0.0043 |
| Q4 | 388 649 | 0.629 | 0.427 | 0.214 | 0.0036 |
| Q5 commonest | 687 361 | 0.538 | 0.461 | 0.167 | 0.0041 |

Per decade of training rows in the group, adjusted for arity, geometry and system
crowding: MetaSUN odds ×0.726 [0.675, 0.786] and SUN odds ×0.552 [0.367, 0.721].
The SUN interval here excludes 1 comfortably; the system channel's does not.

**Symmetry is not the same variable as frequency, and only frequency carries the
SUN signal.** Splitting at sg 75 gives MetaSUN 0.291 (low) against 0.180 (high),
a real 1.6× — but SUN 0.0055 against 0.0056, nothing at all. Low symmetry buys
novelty; rarity of the group buys stability too.

### And here, unlike everywhere else, selection *is* intervention

The space group is not something the model chooses. It is the start token, drawn
from a fixed table before the sequence begins, and the model's job is to respond
to it: the joint is `p_table(G) · p_θ(gene | G)`. Replacing `p_table` with any
other distribution over the same 228 groups therefore changes **only which cells
are visited**, and the pool's per-group rates are unbiased estimates of
`p_θ(· | G)`'s yield. The reallocation is identified, which is not true of the
chemical system — that the model picks itself, as part of the sequence, so
forcing it is a genuine intervention.

Spending the whole budget on the rarest two quintiles, over the nine runs that
draw their start token this way:

| budget | MetaSUN | SUN |
| --- | --- | --- |
| as drawn (9 unconditional runs, 9000 genes) | 0.226 [0.218, 0.235] | 0.0068 [0.0053, 0.0087] |
| rarest two quintiles of space-group frequency | **0.265 [0.251, 0.279]** | **0.0095 [0.0069, 0.0132]** |
| commonest two quintiles | 0.206 [0.193, 0.220] | 0.0051 [0.0032, 0.0081] |

1.17× on MetaSUN, which the intervals separate, and 1.40× on SUN, which they do
not. The quintile boundary was chosen after seeing the data; what was not is the
monotone trend in `log_sg` the regression fits, and the two agree.

**Picking individual space groups, by contrast, does not work.** Ranking the 228
groups by their MetaSUN on one half of the unconditional pool and spending the
other half on the above-average ones returns 1.29× [1.25, 1.34]; the same
procedure on SUN returns **0.62× [0.44, 0.77]** — worse than not selecting at
all. With 61 SUN events spread over 228 groups the per-group estimates are noise,
and acting on them destroys value. Only the smooth trend is usable.

## 7. Conditioning with the archive's own prior is already implemented, and buys nothing

Three of the pooled runs are chemical-system-conditioned models, and
`wyformer-protocol-wandb` samples their cohorts as
[`chemical_system_sampler.md`](chemical_system_sampler.md) describes: a
(system, space group) pair per structure from the training-fitted
`p(S)·p(G|S)`, with α = 5.6% of systems drawn from the novel-system proposal.
That is conditioned generation, with the archive's own prior as the target
distribution.

It lands in the same place. Per 1000 genes the prior runs touch 993.7 distinct
systems against 988.2, put 5.2% of the budget in unseen systems against 6.2%,
and sit at a median of 40 archive entries against 42. Their space-group marginal
is 0.029 nats from the training marginal; the unconditional runs' is 0.010.

One of the three, `chemsys_e_hull_sg_adamw_wsd`, also predicts the space group
from the system (`predict_start: true`) instead of taking it from the prior's
`p(G|S)`; it scores 0.200 against its parent's 0.201, so that change moved
nothing either.

And the family yields less: MetaSUN 0.190 against 0.226, SUN 0.0020 against 0.0068. The
split is informative — novelty 0.775 against 0.625, metastability 0.256 against
0.453. Within cells matched on crowding, arity and symmetry the same pattern
holds (budget-weighted gaps: novelty +0.150, metastability −0.206, MetaSUN
−0.041), so it is not only where the prior sends them; they build worse once
there. **This comparison is confounded** — three different checkpoints, not one
model with the sampler swapped — so it is a warning rather than a measurement.
The warning is that forcing a cell is not free, and the pool's reallocation
numbers above assume it is.

## What this cannot answer

* **Selection is not intervention — for the chemical system.** Every
  system-level cell rate is measured on genes the *unconditional* models chose to
  draw there. Conditioning forces the model into a cell it would not have chosen,
  and §7 is direct evidence that doing so changes what it builds. The enrichments
  in §5 are what a reallocation is worth *if* the model's within-cell behaviour is
  unchanged, and §7 says it is not. This does **not** apply to the space group,
  which is drawn from an exogenous table to begin with (§6).
* **Only cells the pool visits.** The winning `≥317` cell is 291 genes, 2.4% of
  the budget; there is no evidence at all about systems denser than 4870 entries,
  because LeMat-Bulk has none.
* **SUN is not resolved.** 67 events over 12 000 genes. Detecting a 2× SUN lift
  at 80% power needs 4181 genes per arm and a 1.5× lift needs 13 959 — which is
  the protocol's own [power finding](de_novo_ranking_protocol.md#how-many-genes),
  re-derived here. Every SUN statement above is directional.
* **One MLIP, one hull.** Crowding and the hull are not independent: the
  `orb_conserv_inf` hull in a crowded system is built from more entries, so both
  "novel" and "metastable" are *harder* tests there. Both nevertheless move as
  reported, which strengthens §3 rather than weakening it.
* **Twelve different models.** Run fixed effects absorb level differences, not
  slope differences. And uniqueness is judged within a run, so a structure two
  runs both found counts twice.

## What to do with it

1. **Do not sell conditioning as a novelty play.** It is a stability play. The
   chemical system moves metastability by 2.1× and novelty by 1.3×; the
   hypothesis's own mechanism — the occupied hull — is visible only in SUN, where
   the pool cannot resolve it.
2. **Move the space group first.** It is the cheapest change — re-weight the
   start-token distribution, which is a fixed table today and which
   `predict_start: true` already turns into a model decision — it needs no
   per-query target, it is the only lever whose MetaSUN and SUN signs agree, and
   30% of every cohort is currently spent in the three groups that pay worst.
   Unlike every other lever here, this one is causally identified: the start
   token is exogenous, so re-weighting it changes where the budget goes and
   nothing else.
3. **If the target is MetaSUN, steer toward worked chemistry**, not away from it:
   dense ternaries in low-symmetry groups, 3.4× cross-fitted. If the target is
   SUN, the system-level prior wants to go the other way, and that arm has to be
   run deliberately at ≥4000 genes to say anything at all.
4. **Price the within-cell penalty before believing the reallocation.** The
   cleanest experiment the pool suggests is not a new prior but a null one: take
   one checkpoint, generate one cohort unconditionally and one with the *system
   prior* restricted to the cells §5 picks, and compare. That isolates the
   forcing cost §7 only hints at, and it is the number every reallocation
   estimate here is multiplied by.

## See also

* [The de novo ranking protocol](de_novo_ranking_protocol.md) — where these artifacts come from
* [Chemical-system mode](chemical_system_mode.md) — how a system is given to the model
* [Sampling (chemical system, space group) inside a palette](chemical_system_sampler.md) — the prior §7 measures
* [Generative novelty screening](generative_novelty_screen.md) — the same tension, measured on the screen side
