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
