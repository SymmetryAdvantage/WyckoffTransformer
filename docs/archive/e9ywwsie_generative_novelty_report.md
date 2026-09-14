# `e9ywwsie` generative novelty screen: what the likelihood lever is worth

Run date: 2026-09-08/09. Pool: `generated/e9ywwsie_dft_attack` (5,000 sampled
genes) -- the same pool as
[the fixed-hull uplift report](e9ywwsie_dft_screen_uplift_report.md), so the two
levers are measured against identical relaxation outcomes. Code: `44d14bf`,
`09e1e5d`, `f511ae6` on `training-loss-fixes`. Design:
[generative novelty screening](../generative_novelty_screen.md).

The first report found the fixed-hull energy screen to be a strong stability
ranker whose (M)SUN gains novelty cancels, and that deduplicating genes against
the training set first recovers them. This one asks whether the generator's own
likelihood is a second lever: whether `-log p(gene)` is a usable novelty
estimator, what it costs in stability, and whether energy plus likelihood can
do the job with no reference lookup at all.

> **Cross-theory qualification, unchanged from the first report.** The energy
> screen's estimators and hull are PBE; the protocol's energies and hull are ORB
> (`orb_conserv_inf`). Everything below measures whether a ranking carries signal
> about a relaxed funnel, not a DFT claim.

## Protocol

- **Generator / scorer**: wandb run `e9ywwsie`, checkpoint `runs/e9ywwsie`. The
  likelihood is read from the same model that produced the pool, so it is the
  sampler's own density.
- **Conditioning**: scored at `energy_above_hull=0`, which is what "sampled
  unconditionally" means for a conditional WyFormer, and why the first report and
  the working notes describe this pool both ways. Every model in the screen
  conditions on some quality channel -- `energy_above_hull` for the generators
  here, `max_force` for the gene critic, a formation-energy delta for other
  variants -- and untargeted sampling asks for the clean limit of whatever those
  channels are, all at zero, rather than dropping the conditioning. It is the
  same convention `build_clean_relaxation_condition` already applies on the
  critic side, where a generated gene is scored at `max_force = 0` because it has
  not been relaxed yet. `e9ywwsie` carries the single `energy_above_hull`
  channel, so its clean limit is one number.

  The likelihood agrees the pool sits there: mean log p over 600 genes is highest
  at zero and falls monotonically away from it (-18.78 at 0, -19.96 at 0.1,
  -24.56 at 0.5, -27.76 at 1.0). Since the channels differ by checkpoint, a pool
  scored with a model trained on a different conditioning has to be given that
  model's clean limit, not this one.
- **Novelty scoring** (`wyformer-gene-novelty`): 64 representation draws per
  gene, seed 0, one A100. 5,000 of 5,000 genes scored, ~6 minutes.
- **Outcomes**: `generated/e9ywwsie_dft_attack/protocol/structures.csv`, from the
  first report's relax+score run (ORB `orb-v3-conservative-inf-omat-20250404`,
  `LeMat-Bulk-MLIP-Hull` rev `70d505b`, trial schedule `0:1,2:2,*:3`, 11,835
  trials over 4,989 unique genes). Every gene in the pool was relaxed, so all
  arms below are scored on the same complete pool and the comparison is exact.
- **Analysis**: `scripts/analyse_novelty_screen.py`, writing
  `generated/e9ywwsie_dft_attack/novelty_screen.json`.
- **Baseline rates**: pool MetaSUN 0.2892, SUN 0.0102, 68.9% gene-novel.

## Estimator convergence

The estimator is a lower bound that tightens with the number of representation
draws. The median spread of `log p` across representations of the same gene is
~2.1 nats -- the model's likelihood is *not* order-invariant, so the count is not
optional. On 400 genes:

| draws | median log p | Spearman vs previous |
|---:|---:|---:|
| 4 | -18.25 | -- |
| 8 | -17.43 | 0.920 |
| 16 | -17.31 | 0.975 |
| 32 | -17.29 | 0.988 |
| 64 | -17.22 | 0.996 |

32 draws is enough for the ranking; 64 was used throughout and costs ~6 min for
5,000 genes.

## It is a novelty estimator, and it does not beat the lookup

| AUC against | `surprisal` | fingerprint lookup |
|---|---:|---:|
| gene novelty | 0.926 | 1.000 |
| structure novelty, after relaxation | 0.807 | 0.834 |
| MetaSUN | 0.527 | 0.633 |

It was never going to beat the lookup at gene novelty -- that is what the lookup
computes exactly, and the surprisal is a 0.93-AUC proxy for it. It does not beat
it at structure novelty either. What it has instead is that it needs no reference
set.

`surprisal_per_site` is not an estimator at all: AUC 0.467 against gene novelty,
which is no signal. The unnormalised log-density is the quantity that works, and
the two are nearly rank-uncorrelated (Spearman 0.05), so this is a real fork in
the road rather than a cosmetic normalisation. **Do not "normalise for length".**

## It prices novelty in stability

Spearman(`surprisal`, `e_above_hull`) = **+0.472**: the more surprising the gene,
the worse it relaxes. Spearman against the energy screen's own
`joint_score_adjusted` is +0.376, so the two levers are correlated but far from
redundant.

| surprisal decile | median e_hull | gene novel | structure novel | metastable | MetaSUN | SUN |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (most typical) | 0.024 | 7.4% | 15.2% | 82.6% | 9.6% | 1.0% |
| 1 | 0.036 | 21.6% | 31.3% | 74.9% | 19.4% | 1.4% |
| 2 | 0.062 | 37.1% | 50.1% | 65.9% | 32.1% | 2.2% |
| 3 | 0.078 | 59.5% | 68.3% | 58.1% | **40.5%** | 1.4% |
| 4 | 0.103 | 79.6% | 80.6% | 48.7% | 38.5% | 1.6% |
| 5 | 0.114 | 89.6% | 85.7% | 44.2% | 38.8% | 1.2% |
| 6 | 0.127 | 95.0% | 90.2% | 38.7% | 34.9% | 0.6% |
| 7 | 0.140 | 99.8% | 88.6% | 32.5% | 29.5% | 0.6% |
| 8 | 0.145 | 99.8% | 89.2% | 32.1% | 28.9% | 0.0% |
| 9 (most surprising) | 0.184 | 100.0% | 86.2% | 20.0% | 17.2% | 0.2% |

Maximising novelty is not the objective. MetaSUN peaks in the fourth decile,
where a gene is more likely novel than not and still relaxes somewhere. This
table is the operating-point chart for everything below.

## Lookup-free: energy and likelihood alone

The arm that matters when there is no archive to deduplicate against. Because
the useful likelihood signal is a *band* rather than a direction, the screen is
"keep a surprisal quantile band, then rank that band by energy".

### One lever at a time, B=250

| arm | MetaSUN | x pool |
|---|---:|---:|
| random | 0.289 | 1.00 |
| energy only | 0.300 | 1.04 |
| likelihood only, rank most surprising | 0.160 | 0.55 |
| likelihood only, rank least surprising | 0.092 | 0.32 |
| likelihood only, best band, drawn at random inside | 0.381 | 1.32 |
| **band, then rank by energy** | **0.716** | **2.48** |

Ranking on the likelihood in either direction is *worse than random*. The fair
single-lever arm is the best band spent at random inside itself, since the
likelihood offers no ordering within a band: 1.32x, holding 1.30x split-half, and
flat across budgets because it is a base rate rather than a ranking.

The combination is worth more than either and more than their product
(1.04 x 1.32 = 1.37 against an observed 2.48). This is the first report's
cancellation being repaired, not two gains stacking: the energy ranking alone
spends itself on the low-lying genes the archive already holds -- its best decile
is 83% known formulas -- and the band removes exactly those before the ranking
runs. The band has no ordering inside it; the ranking has no idea what is already
known.

### The band surface, MetaSUN uplift at B=250

| drop bottom \ keep up to | 0.70 | 0.80 | 0.90 | 1.00 |
|---:|---:|---:|---:|---:|
| 0.00 | 1.04 | 1.05 | 1.05 | 1.04 |
| 0.10 | 1.54 | 1.58 | 1.60 | 1.63 |
| 0.20 | 2.17 | 2.12 | 2.05 | 2.01 |
| 0.30 | 2.41 | 2.45 | 2.42 | 2.39 |
| 0.40 | 2.35 | **2.48** | 2.48 | 2.45 |
| 0.50 | 2.31 | 2.31 | 2.34 | 2.31 |
| 0.60 | -- | 2.03 | 2.25 | 2.20 |

A plateau, not a peak: everything in `drop 0.3-0.5` x `keep to 0.7-1.0` is
2.3-2.5x, so the setting needs no tuning. Dropping the most typical third or so
does nearly all the work; cutting the surprising tail adds ~0.05x.

Choosing the band on one random half of the pool and spending the budget on the
other gives **2.41x** held out over 40 splits, against 2.48x in sample -- almost
none of the grid maximum is selection bias. At B=500 the held-out figure is
2.18x; at B=1000, 1.96x.

The soft alternative -- rank by a weighted sum of the two scores' percentile
ranks, excluding nothing -- was swept over eight weights and is worse everywhere,
peaking at 2.31x (weight 0.3) at B=250 and 2.14x at B=500. Under fusion a
low-energy but wholly typical gene can buy its way back in, and that is exactly
the gene the archive already has. The hard band wins.

### What the 250 submitted genes actually are

Band `(0.40, 0.80)` then the 250 best by energy -- 5% of the pool, 680
relaxations (2.72/gene against the pool's 2.37, because selection prefers
high-DoF genes):

| | selected | pool |
|---|---:|---:|
| has structure | 100.0% | 99.6% |
| valid | 98.8% | 91.0% |
| unique | 98.8% | 91.0% |
| novel | 92.0% | 68.5% |
| metastable (<= 0.1 eV/atom) | 79.6% | 49.8% |
| **stable (<= 0)** | **1.6%** | **2.8%** |
| **MetaSUN** | **71.6%** | 28.9% |
| **SUN** | **1.2%** | 1.0% |
| median `e_above_hull` | 0.056 | 0.101 |

Uniqueness and validity are not the binding constraints -- both are ~99% and
both are already inside the MetaSUN definition. **The binding constraint is the
"Meta".** This arm buys a 2.5x MetaSUN gain and essentially no SUN gain, and it
even lowers the stable fraction relative to the pool (1.6% against 2.8%). That is
mechanical: the band deliberately discards the most typical genes, which is where
the archive-like, genuinely stable structures live. Read the 71.6% as "within
0.1 eV/atom of the ORB hull, novel and unique", never as a discovery rate.

## With the lookup, the lever flips sign and beats it

Once novelty is *guaranteed* by the fingerprint filter, the surprisal stops being
a novelty estimator and becomes a plausibility one. Inside the novel subset
(3,439 genes, MetaSUN 0.368):

| AUC of *low* surprisal against | |
|---|---:|
| MetaSUN | 0.657 |
| metastability | 0.658 |
| structure novelty | 0.533 |

The signal is stability, not further novelty. Acting on it, at B=250:

| arm | MetaSUN | x pool | SUN | x pool |
|---|---:|---:|---:|---:|
| energy only | 0.300 | 1.04 | 0.016 | 1.57 |
| lookup-free band, then energy | 0.716 | 2.48 | -- | ~1.2 |
| fingerprint-novel, then energy | 0.804 | 2.78 | 0.052 | 5.09 |
| **fingerprint-novel, least-surprising 30%, then energy** | **0.848** | **2.93** | **0.060** | **5.87** |

Stacking the two levers in the *intuitive* direction -- novel **and** surprising
-- is the worst combination on the board (2.13x).

This does not contradict the first report's finding that a learned
"is this gene known" model has no headroom (an oracle on relaxed-structure
novelty scores 2.75x against the lookup's 2.78x). The gain here is not a better
novelty prediction. It is a quantity the novelty question does not contain: how
ordinary the gene is, which is what still varies once novelty is settled.

## Verdict

1. `-log p(gene)` is a good novelty estimator (AUC 0.93 / 0.81) that loses to a
   free lookup on both counts, and its value is that it needs no reference set.
2. Lookup-free, energy + likelihood reach 2.41x MetaSUN held out at B=250 --
   about 87% of the lookup's 2.78x -- where each lever alone is worth 1.04x and
   1.32x. The combination is the point; neither lever is a screen by itself.
3. With the lookup available, the likelihood adds on top of it in the opposite
   direction, reaching 2.93x MetaSUN and 5.87x SUN at B=250.
4. Every MetaSUN number here decays with budget. The `least-surprising` arm is
   2.93x at B=250 and 2.16x at B=1000, below the lookup's own 2.30x there: the
   novel-and-plausible subset runs out. Choose the keep fraction against the
   budget, not once and for all.
5. SUN remains unresolved. The pool holds 51 stable structures, the best arm
   finds 15 of them at B=250, and the lookup-free arm finds 3. No SUN multiplier
   in this report rests on enough hits to argue from.

## Reproduction

```bash
python -m wyckoff_transformer.cli.gene_novelty \
    generated/e9ywwsie_dft_attack/wyckoff_genes.json.gz \
    --model-path runs/e9ywwsie --condition energy_above_hull=0 \
    --permutation-samples 64 --device cuda \
    --out generated/e9ywwsie_dft_attack/gene_novelty.csv

python scripts/analyse_novelty_screen.py generated/e9ywwsie_dft_attack \
    --budgets 250,500,1000
```

On ASPIRE 2A both run inside the container
(`bash scripts/run_in_singularity.sh ...`, `module load singularity` first).
