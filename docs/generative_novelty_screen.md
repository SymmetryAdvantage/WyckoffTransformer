# Generative novelty screening

> **Scope:** the second lever of the two-lever screen, measured on the same
> `e9ywwsie` pool as the first. It adds no new relaxations and no new labels: it
> reads a number off the generator that produced the pool.

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

## What it is worth

Measured on `generated/e9ywwsie_dft_attack`: 4989 relaxed representatives,
68.9% novel by fingerprint, pool MetaSUN 0.289 and SUN 0.0102.

### It is a novelty estimator

| AUC against | `surprisal` | fingerprint lookup |
| --- | --- | --- |
| gene novelty (the lookup itself) | **0.926** | 1.000 |
| structure novelty, after relaxation | 0.807 | **0.834** |

It does not beat the lookup, and it was never going to: gene novelty is exactly
what the lookup computes, and the surprisal is a 0.93-AUC proxy for it. What it
has instead is that it needs no reference set -- which is what makes it usable on
a model whose training archive is not to hand, or on a pool being triaged against
a reference nobody has fingerprinted.

`surprisal_per_site` is not a novelty estimator at all: 0.467 against gene
novelty, which is no signal. The properly normalised log-density is the quantity
that works, and dividing it by the site count destroys it. Both are computed so
that this stays checkable rather than assumed.

### It prices novelty in stability

Spearman(`surprisal`, `e_above_hull`) = **+0.472**: the more surprising the gene,
the worse it relaxes. The funnel is the operating-point chart --

| decile | median e_hull | gene novel | metastable | MetaSUN |
| --- | --- | --- | --- | --- |
| 0 (most typical) | 0.024 | 7.4% | 82.6% | 9.6% |
| 3 | 0.078 | 59.5% | 58.1% | **40.5%** |
| 6 | 0.127 | 95.0% | 38.7% | 34.9% |
| 9 (most surprising) | 0.184 | 100.0% | 20.0% | 17.2% |

-- and it says plainly that maximising novelty is not the objective. MetaSUN
peaks in the fourth decile, where the gene is more likely novel than not and
still relaxes somewhere.

### As a screen, at a 250-gene budget

| arm | MetaSUN | x pool | SUN | x pool |
| --- | --- | --- | --- | --- |
| random | 0.289 | 1.00 | 0.010 | 1.00 |
| energy only | 0.300 | 1.04 | 0.016 | 1.57 |
| surprisal alone | 0.160 | 0.55 | 0.000 | 0.00 |
| most-surprising 70% -> energy | 0.692 | 2.39 | 0.012 | 1.17 |
| fingerprint-novel -> energy | 0.804 | 2.78 | 0.052 | 5.09 |
| **fingerprint-novel, least-surprising 30% -> energy** | **0.848** | **2.93** | **0.060** | **5.87** |

Two things to read off this, and the second is the one that is not obvious.

**Without the reference set, the estimator recovers most of the lookup's gain.**
Thresholding on surprisal alone gives 2.39x MetaSUN, against 2.78x for the free
fingerprint filter. That is the arm to use when there is no reference set to
deduplicate against.

**With the reference set, the estimator flips sign and beats it.** Once novelty
is *guaranteed* by the lookup, the surprisal stops being a novelty estimator and
becomes a plausibility one. Inside the novel subset (3439 genes, MetaSUN 0.368),
low surprisal predicts MetaSUN at AUC 0.657 -- and the mechanism is stability,
not more novelty: 0.658 against metastability against only 0.533 against
structure novelty. Dropping the 70% the model finds most improbable and ranking
the rest by energy beats the lookup alone -- 0.848 against 0.804 MetaSUN, 2.72x
against 2.51x per relaxation, and SUN 5.87x against 5.09x.

Stacking the two in the intuitive direction -- novel *and* surprising -- is the
worst combination on the board (2.13x). Novelty and plausibility are the two
things being traded, and the fingerprint lookup already bought the novelty.

This does not contradict the earlier finding that a learned "is this gene known"
model has no headroom -- an oracle on relaxed-structure novelty scores 2.75x
against the lookup's 2.78x. The gain here is not from predicting novelty better.
It is from a quantity the novelty question does not contain: how ordinary the
gene is, which is what survives once novelty is settled.

### Caveats

* The `least-surprising` gain is a small-budget effect. At a 1000-gene budget the
  same arm gives 2.16x against the lookup's 2.30x: the novel-and-plausible subset
  has been exhausted and the filter is only removing candidates. Choose the keep
  fraction against the budget, not once and for all.
* SUN rests on 51 stable structures in the whole pool, so every SUN multiplier
  here is a handful of hits. MetaSUN is the number to argue from.
* The hull behind `e_above_hull` is ORB's, and the energy screen's is PBE. This
  measures whether the rankings carry signal, not a DFT claim.
* The pool was generated by `e9ywwsie` and scored by `e9ywwsie`. The surprisal is
  then the sampler's own density, which is the right quantity for triaging that
  sampler's output; scoring one model's pool with another model is a different
  and untested question.
