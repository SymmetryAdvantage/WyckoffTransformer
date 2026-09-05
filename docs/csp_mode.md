# CSP mode: predicting the structure of a given composition

De novo generation asks WyFormer for any plausible crystal and scores the answer
on novelty. CSP fixes the formula and asks which structure it adopts, where
there is a right answer and novelty is beside the point. This document describes
what that mode is, what is implemented, and what has not been measured.

Three pieces:

1. a backbone conditioned on the composition and on how far the structure sits
   above the best polymorph of that composition,
2. a regressor that predicts `min(E | gene)` from a Wyckoff gene, fitted with a
   censored likelihood,
3. decoding that respects the target composition exactly, ranked by (2).

Piece (2) is `wyckoff_transformer.censored`, (3) is `wyckoff_transformer.csp`
and the `wyformer-csp` CLI. Piece (1) needs a conditioned backbone to be
trained; the label it should be trained on is computed by
`censored.gene_level_polymorph_delta`, and nothing has been trained yet.

**The composition is a decode-time constraint, not a conditioning input.** The
model is never told which formula it is building. `CompositionTarget` reaches
only the mask; the model's `cond` vector carries whatever scalar the backbone
was trained on (`energy_above_hull`, say) and nothing else. So the distribution
the decoder samples from is the *unconditional* next-token distribution,
renormalised over the choices that keep the target reachable.

That is enough to guarantee the formula, and it is what makes the acceptance
rate one, but it is weaker than conditioning in three ways. The model cannot
plan -- it does not reserve room for the O3, it is merely stopped from making
that impossible. The `log_prob` a candidate carries is a renormalised
unconditional probability, not `p(gene | composition)`, which is worth
remembering because it is what `--strategy beam` searches over and what orders
the output when no regressor is given. And the gap between the two should be
widest for unusual compositions, where the unconditional prior pulls hardest
against the constraint.

Real conditioning is a training change. The tokenisers already emit
`composition_tokens` and `composition_counts` for a `counters: composition`
field (`mp_20_CSP.yaml`), but no model reads them today; wiring them into the
sequence-level input and retraining is what piece (1) actually is.

## Why `min(E | gene)` and not the energy

A Wyckoff gene fixes the space group, the species and the occupied Wyckoff
positions. It fixes neither the free coordinates nor the cell, so it does not
determine an energy — it determines a manifold of structures, whose width is
governed by the gene's positional degrees of freedom. The
[dof study](pyxtal_dof_reduction_study.md) measures how wide: the median
generated gene has 3 free coordinates and the tail runs to 186.

A regressor fitted to gene→energy pairs with an MSE therefore estimates
`E[E | gene]`, the average structure on the manifold. That is the wrong end of
the distribution for CSP. The reconstruction step is a stochastic search that
can be run repeatedly — the [trial budget](upi73i4k_orb_protocol.md#the-trial-budget)
shows a second PyXtal trial is worth ~50 meV/atom to the median gene above dof 3
— so what decides whether a gene is worth spending the budget on is the *best*
energy it can reach, not the average.

`min(E | gene)` is also a genuine deterministic function of the gene, where
`E[E | gene]` carries an irreducible spread. There is no noise floor: with
enough data the regressor can be arbitrarily accurate.

## Why the labels need a censored likelihood

The difficulty moves from the estimand to the labels. The dataset does not
contain `min(E | gene)`; a dataset entry is *one* structure on the gene's
manifold, so its energy is an upper bound:

    E_obs >= m(g)

Two consequences, and only the second is obvious.

The bound is loose, by an amount that grows with the gene's degrees of freedom.
And its looseness is **confounded with how often the gene appears in the
dataset**: a gene seen once has one loose bound, a gene seen fifty times has
fifty and its lowest is nearly tight. Regressing the minimum-of-observed
directly, or the mean with an MSE, both make the model learn part of a
popularity prior wearing an energy costume — and in a search that then minimises
the prediction, that is a mode-collapse mechanism, not a rounding error.

The censored likelihood reads the bound literally. Each observation is modelled
as the gene's optimum plus a non-negative excess, seen through label noise:

    E_obs = m(g) + eps + eta,   eps ~ Exponential(1 / s(g)),  eta ~ Normal(0, sigma^2)

which makes `E_obs` exponentially modified Gaussian and the loss its negative
log density. The excess scale `s(g)` is predicted per gene alongside the
location, so it can track the degrees of freedom that produce it; `sigma` is
fixed, and doubles as the width over which the otherwise hard `E >= m` boundary
is smoothed into something a gradient can cross.

Nothing groups the data by gene — every row contributes its own term and the
loss drops into the existing `Scalar` path in place of the MSE. The grouping
happens implicitly and correctly: a gene appearing `n` times contributes `n`
terms, each pushing the location down and each blocking it from rising above
that row's energy, so the frequency weighting the confound came from is exactly
what the likelihood now uses. The residual bias is analytic,
`E[min_i E_i] - m(g) = s / n`, and `censored.expected_min_bias` reports it.

On synthetic data where the answer is known, the fit lands on the floor
(-1.500 against a true -1.500) where an MSE fit lands on the mean (-1.199), and
recovers the excess scale to within 1%.

### Training the regressor

    uv run python -m wyckoff_transformer.train yamls/models/base_sg_energy_censored.yaml

which differs from `base_sg_energy.yaml` in two lines: `scalar_loss: censored`
and `outputs: 2`. The labels are unchanged — what changes is how they are read.

W&B gets three diagnostics alongside the NLL, which is not comparable across
runs because it moves with the fitted scale:

| metric | reads |
|---|---|
| `violation` | fraction of observations *below* the predicted minimum. Should be near zero; a rising value means the location head is being pulled towards the conditional mean. |
| `excess` | mean of `E_obs - m`, which the model claims equals `scale`. |
| `scale` | mean predicted `s`. Expected to grow with degrees of freedom; worth binning by dof when reading a trained model. |

An MAE is deliberately *not* reported as the objective. The location head aims
at the bottom of each gene's manifold and the labels scatter above it, so a
perfect `m` has an MAE of about `s`. `censored.censored_min_mae` computes one
anyway, for comparison against the MSE runs already on the board.

## Composition-constrained decoding

A gene's composition is the sum of the multiplicities of the Wyckoff positions
assigned to each element, decided one site at a time. Sampling freely and
rejecting at the end wastes nearly every draw, because a decoder that has placed
a 4-fold position for an element needing 6 more atoms has usually made the
target impossible several sites earlier.

`SpaceGroupCombinatorics` answers the reachability question so the decoder can
mask any choice that strands the composition. The rule it enforces is the one
`WyckoffProcessor.pyxtal_notation_to_sites` already applies when turning a gene
into a structure: a position with no positional freedom (`dof == 0`) is a fixed
set of points and may be occupied once, while a position with `dof > 0` is a
continuous orbit and may be occupied repeatedly at different coordinates. So the
reusable multiplicities form an unbounded coin problem and the fixed ones are a
scarce resource the elements compete for.

That split is what makes an exact answer cheap. If every element's remaining
deficit is a sum of reusable multiplicities the elements never interact and the
state is feasible with no search; if some deficit is not a sum of *any* of the
group's multiplicities it is infeasible outright; only between the two does a
bounded depth-first assignment of the scarce fixed positions run. The search
reports "feasible" if it exhausts its node budget, so a `False` is trustworthy
and a `True` is occasionally optimistic — the safe direction, since a false
negative would forbid a legal structure while an optimistic true costs one
wasted path.

Checked against exhaustive search on 15 space groups spanning triclinic to cubic
and several centrings, 1200 random (composition, budget) cases: exact agreement,
no false negatives. Every gene the decoder emits has the target composition by
construction.

### Running it

    wyformer-csp out.json.gz --model-path runs/upi73i4k \
        --regressor-path runs/<censored-regressor> \
        --formula BaTiO3 --z 1 --condition-value 0

Output is the same list of pyxtal dicts `wyformer-generate` writes, plus a `csp`
key carrying the space group, `z`, log-probability and predicted energy, so it
feeds `wyformer-cryspr` and `wyformer-protocol` unchanged.

Space groups are enumerated, not sampled: every group that can express the
composition is decoded from and the candidates are pooled and ranked together,
because which setting the formula adopts is the question being asked.
`--z` accepts several values. Identical genes are collapsed by default, since a
repeat costs a relaxation and buys no structure — at `z=1` in a high-symmetry
group there may be only a handful of legal genes and sampling will revisit them.

Sanity check against the real backbone: `BaTiO3` at `z=1` over space groups 221,
123 and 62 gives 10 unique candidates, all with exactly the target composition;
62 is correctly reported as unable to build it (Pnma's smallest multiplicity is
4). The top Pm-3m candidate is `Ba 1b, Ti 1a, O 3d` — the cubic perovskite.

### `sample` or `beam`

`sample` draws ancestrally from the composition-masked distribution. Because the
mask has already removed every choice that strands the composition, the
acceptance rate is one and each draw is a sample from the model's own
distribution restricted to the target formula.

`beam` keeps the most likely prefixes at every cascade field. It returns the
model's modal genes rather than a sample of them, which is right when the
backbone's likelihood is what you trust and wrong when the candidates are about
to be reranked by something else: beams sharing a prefix differ only in their
last sites, so a wide beam spreads the relaxation budget over much less of the
space than the same number of samples. Sequence log-probability is also a sum
over sites and so prefers short genes for reasons unrelated to their being
better structures; `CSPCandidate.normalised_log_prob` divides it out.

The default is `sample`, and reranking a diverse sample is the configuration
worth measuring first. It is a strict subset of the beam machinery and gives the
clean ablation: does regressor ranking improve the post-relaxation `e_hull`
distribution at fixed diversity?

## The conditioning label

For mode (a) — condition on `Delta_E_polymorph = 0`, the gene whose optimum is
the ground-state polymorph — the label should be assigned **per gene, not per
structure**. A per-structure `Delta_E` presents a gene-reading model with one
input and several targets, and the blurred channel is precisely the one CSP then
conditions at zero. `censored.gene_level_polymorph_delta` computes it from
`min(E | gene)`, so every structure sharing a gene carries one label.

It also returns `polymorph_count`. A composition seen with a single gene gets
`Delta_E = 0` by construction, but that zero says only that nothing better was
*seen*, not that the gene is a ground state — and in MP-20 and LeMat-Bulk a
large share of compositions are unopposed, so those zeros pile onto exactly the
value sampling conditions at. Carry the count as a second conditioning channel,
or restrict training to compositions with more than one polymorph, rather than
letting them dilute it.

Mode (b) — condition on the formation energy that puts the composition on the
hull — is an affine reparametrisation of `e_hull = 0` per composition, and is
what a backbone already conditioned on `energy_above_hull` does. Its new content
over what `upi73i4k` already ran is the composition constraint, not the energy
target, and the write-up of any ablation should say so.

## What has not been done

- **No conditioned backbone has been trained**, so the composition enters only
  as the decode-time constraint described above, and `Delta_E_polymorph` is not
  a channel any existing model has. `gene_level_polymorph_delta` produces the
  label; wiring the composition counters into the model, building the dataset
  and training on it needs a GPU.
- **No regressor has been trained.** The likelihood is verified on synthetic
  data and through the trainer's real `Scalar` path, but the ceiling on real
  data is unmeasured. The first thing to measure, and it is nearly free: group
  the training set by augmented fingerprint and look at the spread of energies
  among structures sharing a gene, bucketed by dof. That bounds what any
  gene-level regressor can achieve. Second, and also cheap: the 2500
  `upi73i4k` genes already carry ORB-relaxed energies, so a Spearman
  correlation between the regressor's prediction and the relaxed `e_hull` on
  those genes is the go/no-go for the whole ranking step.
- **The labels are still harvested, not generated.** `min(E | gene)` estimated
  from dataset entries is an upper bound however it is fitted. The unbiased
  route is to generate the labels — k PyXtal trials plus relaxation per gene,
  taking the minimum, with the k-dependence extrapolated — using the CrySPR
  pipeline that already exists. Expensive, and the only way to labels that are
  not frequency-confounded at all.
- **No end-to-end CSP benchmark.** Match rate against known structures on a
  held-out set is the measurement this mode should be judged by, and it has not
  been run.

One expectation worth setting: all of this improves the *ranking of genes*.
Most of the loss in the funnel happens after the gene — median `e_hull` 0.241
eV/atom on valid structures — and that is a reconstruction problem the dof study
is attacking. Gene-level energy selection and better free-coordinate priors are
complements, and the second is likely to move the numbers more.
