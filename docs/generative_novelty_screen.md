# Generative novelty screening

> **Scope:** the second lever of the two-lever screen. It reads a number off the
> generator that produced the pool and needs no reference set.
> - First measured on the `e9ywwsie` pool, 2026-09
>   ([below](#what-it-was-worth-on-e9ywwsie-2026-09)).
> - **Re-measured 2026-10-09 under the [rules of engagement](rules_of_engagement.md)**
>   on the current LeMat-Bulk models: two fully relaxed 10,000-gene pools, with
>   broadside, fire-discipline and fire-control run once with the fingerprint
>   lookup and once with the surprisal in its place
>   ([below](#under-the-rules-of-engagement-2026-10)).
>
> **Result.** Lookup-free fire-discipline matches the lookup on both backbones.
> Lookup-free fire-control *beats* it on the CFG backbone (MetaSUN 0.744 against
> 0.647 at B = 1000) and loses on the unconditional one (0.526 against 0.636),
> because a quantile band has to be placed for each pool and placing it needs
> the reference. The best arm still uses both levers: the lookup, then the least
> surprising 30%, then energy (0.854 and 0.778).
>
> - **Not a symmetry artefact.** With every structure relaxed in its gene's
>   symmetry, the CFG fire-control win holds at the same size: +0.093 against
>   +0.097 ([below](#is-the-win-a-symmetry-artefact)).
> - **It works through the regressor.** Surprisal predicts the energy
>   regressor's error, and mostly its optimism, better than any size proxy: AUC
>   0.75 and 0.71 for a miss of more than 0.1 eV/atom
>   ([below](#what-the-surprisal-knows-about-the-energy-predictor)).

## Why a second lever

`docs/dft_fixed_hull_attack.md` builds an energy screen and
[the uplift report](archive/e9ywwsie_dft_screen_uplift_report.md) measures what
it is worth. The result is a strong stability ranker whose gains (M)SUN mostly
cancels: the screen finds low-lying genes partly by finding compositions the
reference archive already holds, so the best-scoring decile is 90% metastable
and 83% already-known formulas, and MetaSUN peaks one decile down.

The fix that worked there was to remove the overlap first -- drop the genes
whose Wyckoff fingerprint is already in the reference set, *then* rank by
energy. That is a free lookup, and it is also a binary one: it can only remove
the fraction of the pool it can see (31% here), and it says nothing about the
genes it keeps.

`wyformer-gene-novelty` supplies the continuous version. WyFormer is a
generative model, so it assigns every gene a probability, and a gene it emits
often is a gene its training archive is full of. The self-information

\[
I(G) = -\log p_\theta(G)
\]

is therefore a novelty score with no reference lookup in it at all.

## What the density is

WyFormer does not generate a sequence. It generates a *set* of Wyckoff sites, in
a uniformly random order, under any of the equivalent enumerations of the same
positions. Writing \(R(G)\) for the set of distinct token sequences that decode
to `G`,

\[
p_\theta(G) = \sum_{r\in R(G)} p_\theta(r)
            = |R(G)|\;\mathbb E_{r\sim U(R(G))}\, p_\theta(r),
\]

estimated by drawing representations uniformly from \(R(G)\) and taking the
importance-weighted mean. Three details are not optional:

* **\(|R(G)|\) is computed exactly**, not estimated. It is of the order of
  \(\log n!\) -- tens of nats -- so dropping it turns the score into a measure of
  gene size. Equivalent enumerations that collapse to the same multiset of
  tokens are counted once, and repeated site tuples do not multiply the count.
* **The stopping term is charged once.** Every cascade field carries STOP at the
  stopping position in the tokenised data; generation reads only the first, and
  the rest marginalise away.
* **The space group is not predicted by the model.** Generation draws it from the
  empirical training distribution, so its log-probability comes from
  `spacegroup_distribution.json` and is reported in its own column.

The estimator is a lower bound that tightens with `--permutation-samples`; the
log-mean-exp and the Jensen (ELBO) forms are both reported, and they agree when
the model's likelihood is order-invariant. On `e9ywwsie` it is not -- the median
spread across representations is ~2 nats -- so the count matters. The ranking is
stable from 32 draws (Spearman 0.988 between 16 and 32, 0.996 between 32 and 64);
64 costs about five minutes for 5000 genes on one A100.

**Score it at the condition the pool was generated at.** The density a sample
came from is the conditional one, and a novelty number read off a different
conditioning is a number about a different generator.

For an untargeted pool that means the model's **clean limit**, not the absence of
conditioning. A conditional WyFormer has no unconditional mode: "sample
unconditionally" is implemented as asking for the ideal values of whatever
quality channels the checkpoint was trained on -- `energy_above_hull=0` for the
generators here, and elsewhere `max_force=0` or a formation-energy delta of zero,
each at zero together. The channels differ by checkpoint, so read
`trainer.condition_features` rather than assuming this run's single channel;
`build_clean_relaxation_condition` applies the same convention on the critic
side, where a generated gene is scored at zero force because it has not been
relaxed yet.

**A guided pool: the condition yes, the guidance no.** A pool drawn with
classifier-free guidance came from the guided distribution,
`softmax(l_u + w (l_c - l_u))` at every token. `--guidance-scale w` scores
under exactly that density, so it is available. It is the worse estimator,
though. On the CFG pool below, drawn at w = 5:
- Against gene novelty, the guided score's AUC is 0.815 and the conditional
  model's (w = 1) is 0.943.
- The guided density is far from order-invariant: the median spread across
  representations is 8.2 nats, against 1.1 for the conditional one. Its
  importance estimate is therefore a much looser bound.
- At B = 1000 every arm built on it does worse: fire-discipline 0.538 against
  0.580, fire-control 0.612 against 0.744. The one exception in the budget sweep
  is fire-discipline at B = 250, 0.576 against 0.532.
- Score a guided pool at the condition it was drawn at, with
  `--guidance-scale 1` (the default).

## Running it

```bash
python -m wyckoff_transformer.cli.gene_novelty \
    generated/<run>/wyckoff_genes.json.gz \
    --model-path runs/<run> --condition energy_above_hull=0 \
    --permutation-samples 64 --device cuda \
    --out generated/<run>/gene_novelty.csv

python scripts/analyse_novelty_screen.py generated/<run>
```

The analysis needs `dft_screen.csv` and a relaxed, scored `protocol/` in the same
pool, exactly as `scripts/analyse_dft_screen_uplift.py` does.

The rules-of-engagement comparison below is one 4-GPU job per backbone on
ASPIRE 2A. It draws the pool and scores every gene's predicted `e_hull` and
surprisal, then relaxes every unique gene and replays the arms. Resubmitting the
same command continues from where it stopped.

```bash
bash scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh --backbone cfg      # or uncond; --pilot for 200 genes
python scripts/analyse_roe_surprisal.py $WYFORMER_RUNS/roe_surprisal/cfg    # the replay alone
python scripts/analyse_roe_surprisal.py $WYFORMER_RUNS/roe_surprisal/cfg --track fixed_symmetry
python scripts/analyse_surprisal_energy_error.py $WYFORMER_RUNS/roe_surprisal/{cfg,uncond}
```

## Under the rules of engagement (2026-10)

### Design

**Models.** All three are trained on `lemat_bulk_fmax1_stress` or its slice.
- **CFG backbone:** `ehull_adamw_wsd_5x_cfg_cont-20260930-071120`, sampled at
  `energy_above_hull = 0.05` with w = 5. That is the best cell of its parent run
  in [the guidance grid](archive/cfg_ehull_guidance_grid_20261002.md).
- **Unconditional backbone:** `unconditional_5x_ehull01_h3x_xl-20260930-033512`,
  trained on the `e_hull <= 0.1` slice and sampled at T = 1.
- **Energy predictor:** `min_energy_adamw_wsd_h7x_xl-20260930-030055`, a
  regressor of `gene_min_formation_energy_per_atom`. Each gene is compared with
  the PBE hull at its composition, averaging 8 equivalent Wyckoff descriptions.
  The best checkpoint was frozen on 2026-10-08 at val MAE 0.0253, while its
  chain was in its last link; the snapshot is
  `$WYFORMER_RUNS/roe_surprisal/models/min_energy_adamw_wsd_h7x_xl-20260930-030055.snapshot-20261008`.

**One pool per backbone, every gene relaxed.**
- Each pool holds 10,000 formally valid genes in sampling order, duplicates kept.
- Every unique gene went through the [de novo protocol](de_novo_ranking_protocol.md):
  ORB `orb_conserv_inf`, trial schedule `0:1,2:2,*:3`, the `lemat_bulk_fmax1_stress`
  novelty reference and the free (post-rattle) track.
- Each mode is then *replayed* as a selection from its pool, so every arm is
  scored against the same relaxation outcomes. Arms share genes, so the Fisher
  p-values below treat paired samples as independent and are conservative.
- MetaSUN and SUN come from the protocol's own `funnel_structure_metrics` and
  are per reconstruction slot.

| pool | unique genes | gene-novel | relaxed | MetaSUN in pool | SUN in pool |
|---|---|---|---|---|---|
| CFG | 9,228 | 75.4% | 9,173 | 4,544 | 66 |
| unconditional | 9,985 | 45.5% | 9,967 | 3,791 | 121 |

**The arms**, at a budget of B reconstruction slots:

| arm | selection |
|---|---|
| broadside | the first B draws. A duplicate occupies a slot and is a miss. |
| fire-discipline / lookup | the first B unique genes whose fingerprint is not in the reference |
| fire-control / lookup | among all of those, the B lowest predicted `e_hull` |
| fire-discipline / surprisal | the first B genes that are unique *within the pool* and in the surprisal band |
| fire-control / surprisal | among all of those, the B lowest predicted `e_hull` |
| fire-control / energy | the B lowest predicted `e_hull` among unique genes, with no novelty lever at all |
| fire-control / lookup+plausible | the lookup, then the least surprising 30% of the novel genes, then energy |

- **The surprisal arms use no reference set.** Uniqueness is checked within the
  pool, and the band is a pair of quantiles of the pool's own surprisal.
- **The band was fixed in advance** at (0.40, 0.80], the best band of the
  `e9ywwsie` report. The band surface and a split-half fit are reported as well.
- **Surprisal settings:** 64 representation draws per gene. The CFG pool was
  scored both at w = 1 and at w = 5. The tables below use w = 1; see
  [above](#what-the-density-is) for w = 5.

### Results at B = 1000

| arm | CFG MetaSUN [95% CI] | CFG SUN | uncond MetaSUN [95% CI] | uncond SUN |
|---|---|---|---|---|
| broadside | 0.458 [0.427, 0.489] | 0.012 | 0.397 [0.367, 0.428] | 0.011 |
| fire-discipline / lookup | 0.548 [0.517, 0.579] | 0.015 | 0.550 [0.519, 0.581] | 0.019 |
| fire-discipline / surprisal | **0.580** [0.549, 0.610] | 0.008 | 0.538 [0.507, 0.569] | 0.020 |
| fire-control / energy | 0.365 [0.336, 0.395] | 0.038 | 0.245 [0.219, 0.273] | 0.047 |
| fire-control / lookup | 0.647 [0.617, 0.676] | **0.039** | **0.636** [0.606, 0.665] | **0.074** |
| fire-control / surprisal | **0.744** [0.716, 0.770] | 0.014 | 0.526 [0.495, 0.557] | 0.059 |
| fire-control / lookup+plausible | **0.854** [0.831, 0.875] | 0.031 | **0.778** [0.751, 0.803] | 0.052 |

Lookup against surprisal, at the same mode (Fisher, two-sided):

| | CFG | unconditional |
|---|---|---|
| fire-discipline | surprisal +0.032 (p = 0.16) | lookup +0.012 (p = 0.62) |
| fire-control | **surprisal +0.097 (p = 3e-6)** | **lookup +0.110 (p = 8e-7)** |

MetaSUN per slot against the budget, fire-control:

| B | CFG lookup | CFG surprisal | CFG lookup+plausible | uncond lookup | uncond surprisal | uncond lookup+plausible |
|---|---|---|---|---|---|---|
| 250 | 0.432 | 0.644 | 0.812 | 0.532 | 0.548 | 0.720 |
| 500 | 0.560 | 0.682 | 0.830 | 0.592 | 0.538 | 0.734 |
| 1000 | 0.647 | 0.744 | 0.854 | 0.636 | 0.526 | 0.778 |
| 2000 | 0.724 | 0.742 | 0.782 | 0.681 | 0.571 | 0.502 (runs short) |

What each arm cost at B = 1000, in relaxation worker-hours:

| arm | CFG atoms | CFG MetaSUN per hour | uncond atoms | uncond MetaSUN per hour |
|---|---|---|---|---|
| broadside | 22.6 | 52.6 | 21.4 | 46.0 |
| fire-discipline / lookup | 25.1 | 55.2 | 25.5 | 50.5 |
| fire-discipline / surprisal | 22.0 | 69.5 | 20.9 | 66.5 |
| fire-control / lookup | 34.5 | 46.5 | 31.3 | 50.7 |
| fire-control / surprisal | 25.3 | 82.3 | 21.0 | 64.5 |
| fire-control / lookup+plausible | 17.3 | 148.6 | 20.3 | 100.1 |

### What it shows

**Without a reference, fire-discipline loses nothing.**
- The surprisal band matches the lookup on both backbones: 0.580 against 0.548,
  and 0.538 against 0.550. Neither difference is significant.
- On the unconditional pool it does so with only 60% gene-novel slots against
  the lookup's 100%. The band also drops the improbable tail, so a novel
  structure from it is more often metastable: P(metastable | novel) is 0.728
  against 0.606.
- It costs about twice the draws: 2,642 against 1,375 on CFG, 2,523 against
  2,168 on the unconditional pool. Draws cost milliseconds, against seconds of
  relaxation each.

**Fire-control without a reference beats the lookup on the CFG pool, at every budget.**
- The lookup's ranking is worst exactly where it is most selective. Its top 250
  is 0.432 MetaSUN, below broadside, and it improves as the budget loosens.
- The lowest predicted `e_hull` among novel genes is where the regressor is
  optimistic and wrong. Those are improbable genes with large cells: 34.5 atoms
  per slot against 25.3.
- The band removes that tail before the ranking runs. P(metastable | novel
  structure) is 0.803 for the surprisal arm against 0.724 for the lookup arm.
- Metastability falls steadily with surprisal: 0.89 in the most typical decile,
  0.23 in the most surprising.

**On the unconditional pool, the fixed band is in the wrong place.**
- The surprisal still separates novel from known genes as well as on the CFG
  pool: AUC 0.940 against 0.943.
- But the pool is only 45.5% novel. Gene novelty is below 50% in the six most
  typical surprisal deciles and climbs only through the last four (0.72, 0.89,
  0.97, 1.00). So only 59% of the (0.4, 0.8] band is novel, and fire-control
  spends the rest on known genes.
- **Fitting the band rescues it,** but the fit is not reference-free. Chosen on
  one half of the pool and spent on the other (40 splits, 500 slots per half),
  fire-control reaches:
  - 0.715 on the unconditional pool, with band (0.6, 0.8], against 0.521 for the
    fixed band;
  - 0.775 on the CFG pool, with band (0.4, 0.6], against 0.744.

  Both are above the lookup's 0.636 and 0.647 at the same selection strength.
  The fit, though, reads relaxed outcomes whose novelty was judged against the
  reference. The band's *position* depends on how novel the pool is, which is
  what the lookup measures. A quantile band set once, without a reference, is
  safe only for a pool about as novel as the one it was set on.

**Both levers together are best, as on `e9ywwsie`.**
- Lookup, then the least surprising 30%, then energy: 0.854 on CFG and 0.778 on
  the unconditional pool, and the most MetaSUN per relaxation hour (148.6 and
  100.1).
- Once novelty is guaranteed, low surprisal means plausible. Among gene-novel
  genes, AUC against metastability is 0.734 on CFG and 0.729 on the
  unconditional pool.
- It runs out of candidates at B = 2000 on the unconditional pool. Its 30% of
  4,548 novel genes is 1,364.

**Energy alone is worse than doing nothing.**
- Fire-control with no novelty lever scores 0.365 and 0.245, below broadside on
  both pools.
- Its top 1000 is 49.5% and 29.6% gene-novel: the regressor's best genes are the
  ones LeMat-Bulk already holds. This repeats the
  [uplift report](archive/e9ywwsie_dft_screen_uplift_report.md)'s finding on a
  stronger regressor.

**SUN goes the other way.**
- Lookup fire-control finds more stable novel structures than surprisal
  fire-control: 39 against 14 on CFG, 74 against 59 on the unconditional pool.
- The band discards the most typical genes, and that is where stable structures
  live. As on `e9ywwsie`, a surprisal band is a MetaSUN lever, not a SUN lever.

### Is the win a symmetry artefact?

**Not this one.**
- **The concern.** The protocol keeps the structure it relaxed *after*
  releasing the gene's symmetry and rattling it. A hit can therefore end up
  as a structure that is not its gene's Wyckoff representation. If the
  surprisal arm won by picking genes that relax *away* from themselves, its
  win would say little about the genes it chose.
- **The test.** The protocol also keeps a second readout, each gene relaxed
  with its symmetry held, which is still exactly the gene's Wyckoff
  representation. Replaying every arm on that readout
  (`--track fixed_symmetry`) gives:

| arm, B = 1000 | CFG free | CFG fixed symmetry | uncond free | uncond fixed symmetry |
|---|---|---|---|---|
| broadside | 0.458 | 0.319 | 0.397 | 0.291 |
| fire-discipline / lookup | 0.548 | 0.383 | 0.550 | 0.434 |
| fire-discipline / surprisal | 0.580 | 0.409 | 0.538 | 0.408 |
| fire-control / lookup | 0.647 | 0.502 | 0.636 | 0.524 |
| fire-control / surprisal | **0.744** | **0.595** | 0.526 | 0.449 |
| fire-control / lookup+plausible | 0.854 | 0.766 | 0.778 | 0.705 |

- **The ordering is the same on both readouts.** Fire-control on surprisal
  beats the lookup by 0.093 with symmetry held (p = 4e-5), against 0.097 free.
  At B = 250 the gap is wider still: 0.468 against 0.308. On the unconditional
  pool the lookup still wins, by 0.075 (p = 9e-4). The fire-discipline
  differences stay insignificant.
- **Releasing the symmetry lifts every arm by about the same amount:** +0.139
  broadside, +0.145 lookup fire-control, +0.149 surprisal fire-control. Only the
  lookup+plausible arm gains less, +0.088.
- **Hits whose fingerprint changes are common in every arm,** so they are not
  where the surprisal arm's lead comes from. On the free readout 51% of
  broadside's hits relax to a fingerprint other than their gene's, against 39%
  for lookup fire-control, 44% for surprisal fire-control and 29% for
  lookup+plausible.
- **A changed fingerprint is not always a broken symmetry.** Between 11% and 15%
  of hits change fingerprint even with the symmetry held, which can only be
  symmetry *gained*.

### What the surprisal knows about the energy predictor

**The question.** The best arm keeps the least surprising 30% of the novel genes
*before* ranking by predicted `e_hull`. Does that work because the surprisal
predicts where the regressor is wrong?

**The method.** `scripts/analyse_surprisal_energy_error.py` reads every
gene-novel representative. A known gene's minimum may be among the regressor's
training targets, so known genes are excluded.
- The outcome is the gene's own energy: ORB `e_above_hull` of the
  fixed-symmetry relaxation, which is still the gene's Wyckoff representation.
- The error is `realized - predicted`, after removing the median. Positive
  means the regressor was optimistic.
- The prediction is against the PBE hull and the outcome against the ORB one,
  so a cross-theory offset is expected: +0.027 on CFG, +0.017 on the
  unconditional pool.

| | CFG (w = 1) | unconditional |
|---|---|---|
| novel genes | 6,901 | 4,531 |
| mean \|error\|, eV/atom | 0.138 | 0.103 |
| Spearman, surprisal against \|error\| | **0.374** | **0.341** |
| the same, within genes of equal site count | 0.357 | 0.254 |
| AUC for \|error\| > 0.1: surprisal | **0.753** | **0.708** |
| — number of sites | 0.594 | 0.656 |
| — number of atoms | 0.562 | 0.631 |
| — the prediction itself | 0.604 | 0.521 |
| AUC for an optimistic miss (realized > predicted + 0.1): surprisal | **0.769** | **0.703** |

**Yes, and mostly as a bias.**
- Surprisal predicts the regressor's error better than any size proxy, and
  most of the signal survives holding the site count fixed.
- The error is mostly *optimism*. From the most typical decile to the most
  surprising, the median signed error on CFG climbs from −0.027 to +0.217
  eV/atom. The share of optimistic misses climbs from 8% to 71% (13% to 65% on
  the unconditional pool).
- On CFG the regressor still *ranks* genes about as well inside every decile
  (Spearman 0.39–0.49). What goes wrong is the level.
- On the unconditional pool the ranking collapses too, in the two most
  surprising deciles: 0.32 and 0.14.
- **Why this would happen:** the two models were trained on the same archive,
  so a gene the generator finds improbable is also outside the regressor's
  support. There the regressor shrinks towards typical values, and for these
  genes that is optimistic. Improbable genes are mostly unstable.

**This is the mechanism behind the fire-control results.** Among the 1000 lowest
predictions over the novel genes, split by the surprisal tercile each gene falls
in:

| surprisal tercile | CFG n | predicted | realized | metastable | uncond n | predicted | realized | metastable |
|---|---|---|---|---|---|---|---|---|
| low | 421 | 0.021 | 0.033 | 0.78 | 508 | 0.012 | 0.022 | 0.78 |
| middle | 302 | 0.020 | 0.086 | 0.56 | 239 | 0.014 | 0.043 | 0.68 |
| high | 277 | 0.013 | 0.214 | 0.19 | 253 | 0.009 | 0.221 | 0.22 |

Values are median `e_hull` in eV/atom, on the fixed-symmetry readout. On the
free readout, metastability is 0.91, 0.71 and 0.44 for CFG.

- The predictions are flat across the terciles; the outcomes are not.
- A quarter or more of lookup fire-control's picks come from the most surprising
  third. About four in five of those miss metastability with symmetry held, and
  more than half do when it is released.
- The lookup+plausible arm removes them. Unlike lookup fire-control, it is good
  from its first slots: 0.812 at B = 250 on CFG, against 0.432.

**Caveats.**
- **The "error" is not only the regressor's.** It also contains the shortfall of
  the reconstruction: three PyXtal trials may not find a gene's minimum, and
  that may be harder for an improbable gene. Holding the site count fixed closes
  the route through size, but not every route.
- **The regressor's own uncertainty has not been compared.** Two candidates are
  the spread across its augmented Wyckoff descriptions and an ensemble. Whether
  they carry the same signal as the surprisal is open.

**What follows.** A bias that is predictable can be corrected instead of cut.
- One option is to fit `residual ~ f(surprisal)` on a relaxed calibration pool,
  then rank by `predicted + f(surprisal)`. That would replace the hard band with
  a pessimistic estimate.
- The calibration needs relaxations and the hull, which fire-control already
  needs, but no novelty lookup. Unlike placing the band, it would not bring the
  reference back in.

**The protocol's uniqueness caveat does not move these numbers.** The protocol
groups structures by sampled gene, so two genes relaxing to one structure would
both count. StructureMatcher, run across genes within each reduced formula,
finds 2 such duplicates among the CFG pool's 4,544 MetaSUN structures and none
among the unconditional pool's 3,791.

### Provenance

- **Code:** branch `roe-surprisal`, commits `82862f1`–`27c5ae1`.
- **ASPIRE 2A, 4×A100 jobs:**
  - unconditional pool: `25727385`;
  - CFG pool: `25730004`. It resumed `25727384`, whose relaxations all failed on
    a node whose GPUs refused CUDA contexts; `--resume` reran exactly those
    trials.
- **W&B:** runs
  [`roe_surprisal_cfg-20261008`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/roe_surprisal_cfg-20261008)
  and
  [`roe_surprisal_uncond-20261008`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/roe_surprisal_uncond-20261008).
  Each holds its pool, the gene scores, the protocol outputs and the analysis as
  an artifact of the same name.
- **Local copies:** `$WYFORMER_RUNS/roe_surprisal/{cfg,uncond}/analysis/{report.json,tables.md,genes.csv.gz}`.

## What it was worth on `e9ywwsie` (2026-09)

Measured on `generated/e9ywwsie_dft_attack` -- 4,989 relaxed representatives,
68.9% novel by fingerprint, pool MetaSUN 0.289 and SUN 0.0102. The full study,
with the convergence check, the band surface, the split-half validation and the
per-column breakdown of what actually gets submitted, is
[the `e9ywwsie` generative novelty report](archive/e9ywwsie_generative_novelty_report.md).
Four results shape how the lever should be used.

**It is a novelty estimator, and it loses to the free lookup.** AUC 0.926 against
gene novelty and 0.807 against post-relaxation structure novelty, against the
fingerprint lookup's 1.000 and 0.834. It is a proxy for something already
computed exactly; its value is needing no reference set. `surprisal_per_site` is
not an estimator at all (AUC 0.467) -- do not "normalise for length".

**It prices novelty in stability.** Spearman +0.472 against `e_above_hull`. The
most typical decile is 82.6% metastable and 7.4% novel; the most surprising is
20.0% metastable and 100% novel; MetaSUN peaks in the *fourth* decile at 40.5%.
Maximising novelty is not the objective.

**Lookup-free, the two estimators need each other.** At B=250 the energy critic
alone is worth 1.04x MetaSUN and the likelihood alone 1.32x (its best band, spent
at random inside it -- ranking on it in either direction is worse than random).
Together, keeping a surprisal band and ranking it by energy gives **2.48x**, well
above the 1.37x their product implies, and holds **2.41x** split-half. That is
about 87% of the lookup's 2.78x with no reference set at all. The band surface is
a plateau, so the setting needs no tuning; soft rank fusion is worse everywhere.

**With the lookup, the lever flips sign and beats it.** Once novelty is
guaranteed, low surprisal predicts MetaSUN inside the novel subset (AUC 0.657),
and the mechanism is stability rather than more novelty (0.658 against
metastability, 0.533 against structure novelty). Fingerprint-novel, least
surprising 30%, then energy: **2.93x MetaSUN and 5.87x SUN at B=250**, against
the lookup's 2.78x and 5.09x. Novel *and* surprising is the worst arm on the
board.

## Caveats

Of the 2026-10 comparison:

* **The judge is a proxy.** Relaxation is ORB `orb_conserv_inf` and stability is
  read against the LeMat-Bulk ORB hull. The energy predictor's hull is PBE. These
  rates rank the screens; they are neither DFT rates nor a benchmark's own MLIP.
* **The band's quantiles see the whole pool.** A campaign drawing in batches
  would estimate them from earlier draws. At 10,000 genes the quantiles are
  stable, but the replay has that much lookahead.
* **Each backbone is one pool and one seed.** The arms within a pool are paired,
  so they can be compared to each other. The two backbones differ in model,
  training slice and sampling, so they should not be compared to each other.
* **Fire-control's selection strength is B over the candidates that pass its
  screen.** That is 1000 of 6,956 novel genes for CFG with the lookup, and 1000
  of 3,691 in-band genes with the surprisal. A different pool size changes it.

Of the `e9ywwsie` study:

* **MetaSUN is not a discovery rate.** The lookup-free arm's headline 71.6%
  MetaSUN at B=250 is "within 0.1 eV/atom of the ORB hull, valid, unique and
  novel". Its *stable* fraction is 1.6%, below the pool's own 2.8%, and its SUN
  rate is 1.2% against a pool 1.0% -- essentially no gain. The band discards the
  most typical genes, and that is where the archive-like, genuinely stable
  structures live. Validity and uniqueness are ~99% and are not the binding
  constraint; the "Meta" is.
* **The gains decay with budget.** The `least-surprising` arm is 2.93x at B=250
  and 2.16x at B=1000, below the lookup's own 2.30x there. Choose the keep
  fraction against the budget, not once and for all.
* **SUN is unresolved.** 51 stable structures in the whole pool, 15 found by the
  best arm at B=250 and 3 by the lookup-free one. No SUN multiplier here rests on
  enough hits to argue from.
* **The hull is ORB and the energy screen's is PBE**, so this measures whether
  the rankings carry signal, not a DFT claim.
* **A gene is not a relaxation.** 250 selected genes cost 680 relaxations against
  the pool average of 2.37 per gene, because selection prefers high-DoF genes.
  Every arm is reported on both denominators.
* **The pool was generated by `e9ywwsie` and scored by `e9ywwsie`**, so the
  surprisal is the sampler's own density -- the right quantity for triaging that
  sampler's output. Scoring one model's pool with another model is a different
  and untested question.
