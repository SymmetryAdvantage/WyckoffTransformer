# Should WyFormer predict the gene, or rank it?

> **STATUS: UNDER CONSIDERATION. THE PROPOSALS BELOW ARE NOT IMPLEMENTED.**
>
> This is a design study written 2026-09-04/05. It records measurements that were
> actually run, and proposals that were *not* built. The artefacts it added are
> `scripts/analyse_gene_energy_critic.py` and this file. **No critic has been
> trained, no sampling policy exists, and nothing has been submitted to any
> leaderboard.**
>
> **One exception, and it postdates most of this text.** The CSP mode discussed
> in the first half *was* built in parallel on 2026-09-05, and its design differs
> from and improves on what is proposed here. It has its own document:
> [CSP mode](csp_mode.md). Where the two disagree, `csp_mode.md` is authoritative
> and this file is the earlier reasoning that fed it. The sections below have
> been annotated where that applies.
>
> Future sessions: treat every "would", "could" and "proposed" literally. Check
> git history and `docs/experiments_plan.md` before assuming any of the remaining
> proposals were later built.

The starting question was whether to replace generative sampling with a
`OneShotWyckoffCSPRegressor`: take a chemical formula and a target energy, emit
the Wyckoff gene of the structure with that formula and that energy. The
argument for it is reduced dimensionality, *if* the mapping is unique. This
document works out whether the mapping is unique (it is not, at any reachable
energy resolution), what the idea decomposes into that *is* worth building, and
where the real leverage turns out to be.

The short version: the one-shot framing is a decoding choice, not a smaller
output space; the useful residue is a composition-constrained CSP mode with a
*relative* energy condition, which has since been built; and for the *de novo*
leaderboard the largest measured lever is a gene-level energy critic used to
shape the sampling distribution, which has not. The critic's payoff there is
capped by something counter-intuitive, measured in
[the ceiling section](#what-a-critic-could-actually-buy): the stability/novelty
anticorrelation, not critic accuracy.

## The one-shot framing, in terms of the current model

A Wyckoff gene is a space group plus a variable-length set of discrete tokens.
Nothing regresses onto that. Any model that emits one is a cross-entropy-trained
decoder, which is what WyFormer already is. "Regressor versus generative model"
is therefore a choice between argmax/beam decoding and sampling, not a different
model class, and it does not shrink the output space. The proposal splits into
three parts:

| part | status when this was written | status now |
|---|---|---|
| conditioning on energy | exists (`condition_feature: energy_above_hull`, AdaLN, `--condition-value`) | unchanged |
| fixing the composition | did not exist; the cascade generates the elements | **built 2026-09-05**, see [CSP mode](csp_mode.md) |
| point-estimate decoding | free with the existing model | `--strategy beam` in `wyformer-csp` |

So the only new capability was a CSP mode. What conditioning buys is a sharper
conditional whose argmax is right more often, which is real but incremental.

**Corrected by what was built.** This study framed the composition as a
*conditioning* input through the AdaLN path. The implementation makes it a hard
*decoding constraint* first: `SpaceGroupCombinatorics` answers reachability
exactly, so every emitted gene has the target formula by construction and the
acceptance rate is one. Conditioning was then added on top as a separate,
optional channel, because a mask stops the decoder from making the composition
impossible but does not let it *plan* for the composition. Read `csp_mode.md`
for that distinction; it is sharper than the framing here.

## Does (formula, energy) determine the gene?

Measured with `analyse_gene_energy_critic.py uniqueness` on both caches. Reduced
formula is element counts divided by their GCD; the gene is the space group plus
the sorted multiset of (element, Wyckoff letter, multiplicity). Within a formula,
formation-energy gaps equal e-hull gaps, so the MP-20 column is comparable.

| | MP-20 train | LeMat-Bulk train |
|---|---|---|
| rows kept by "one ground state per formula" | 86.7% | 53.0% (77.2% within e_hull <= 0.1) |
| rows in formulas holding >= 2 distinct genes | 20.9% | 62.0% |
| gap, ground state to nearest different-gene polymorph, median | 0.010 eV/atom | 0.103 eV/atom |
| share of those gaps <= 0.05 / <= 0.10 eV/atom | 92% / 100% | 35% / 49% |
| polymorphs sharing formula *and* gene | 1.0% of rows | 3.9% of rows |
| test rows that are their formula's lowest-energy entry | 85.0% | 52.3% |
| test formula present in train | 21.0% | 62.9% |
| exact-formula lookup baseline (train ground-state gene == test gene) | 0.9% | 2.4% |

Four conclusions.

- **Unique only in the database sense.** Energy is a real number, so every entry
  has its own. As a function of energy the map is a lookup table with no
  smoothness, so the model cannot generalise along the energy axis for a new
  formula, only across formulas at a given position in the energy ordering.
- **The required resolution is out of reach.** Selecting the ground state from
  its polymorphs needs ~10 meV/atom on MP-20 and 50-100 meV/atom on LeMat-Bulk.
  [The conditioning audit](upi73i4k_ehull_conditioning_audit.md) found no
  separation between targets of 0, 0.025 and 0.05 eV/atom, and LeMat-Bulk merges
  DFT sources that disagree at that scale anyway. Energy conditioning is a coarse
  near-hull tag, not a coordinate that selects a basin.
- **At inference the condition is a constant.** The value one would need is the
  formula's unknown ground-state energy, so in practice 0 is fed. Predicting
  formation energy from the formula first adds nothing: the model already sees
  the formula, and composition-only predictors err by more than the gaps above.
- **The well-posed regression is the other direction.** Gene to coordinates is
  98.4% unique on LeMat-Bulk ([ORB protocol](upi73i4k_orb_protocol.md)).
  Regression is the right frame there and the wrong one here.

**On the "formula to ground state only" variant.** The data loss is in the table:
13% of MP-20, 47% of LeMat-Bulk, 23% within the e_hull <= 0.1 subset. The larger
problem is that "lowest in the database" is not the ground state. Seven in ten
LeMat-Bulk formulas are singletons with median e_hull 0.235, mostly substitution
hypotheticals, so an argmin model would learn "the structure someone happened to
compute". For evaluation, 15% of MP-20 and 48% of LeMat-Bulk test rows are not
their formula's lowest entry, so the target is undefined for them and any CSP
test set must be restricted to ground states. The exact-formula lookup baseline
is negligible (0.9% / 2.4%), so such a test would measure real generalisation;
the honest baseline to beat is nearest-*composition* retrieval, not exact-formula
lookup.

## The CSP mode: condition on relative energy

> **This section was written as a proposal and has since been built.** See
> [CSP mode](csp_mode.md) for what actually exists: composition as a hard
> decoding constraint, `Delta_E_polymorph` as the relative-energy label computed
> per gene by `censored.gene_level_polymorph_delta`, and z enumerated rather than
> supplied. What follows is the evidence that argued for relative energy, which
> the implementation adopted. Its concrete claims about *how* to build it are
> superseded.

Two changes to how energy enters:

1. **The composition fixed exactly.** Proposed here as an AdaLN vector; built as
   a reachability mask, with the vector as a second, optional channel.
2. **Energy measured from the formula's own minimum, not the absolute hull.**

The case for (2), in decreasing order of importance:

- **It carries information a single absolute value cannot.** "Ground state of its
  formula" is relational, an argmin over every entry with that composition. From
  one row at e_hull 0.15 the model cannot tell whether that is the best known
  structure for the formula or a polymorph 0.1 above it. Inferring that from
  absolute e_hull would require memorising the minimum for 2.1M formulas, and for
  an unseen formula it is impossible in principle.
- **The inference-time value is always defined.** Rank 0 exists for every formula
  in training; e_hull = 0 does not, since 72% of LeMat-Bulk rows sit above 0.1.
  Asking for 0 is an extrapolation for most chemistries.
- **Resolution becomes a choice, not a demand on the network.** A bin edge at
  ~20-30 meV/atom above the formula minimum says explicitly that genes within
  label noise of the minimum are all correct answers.

Binning is a *second-order* choice and was initially over-sold in discussion:
once the condition is relative, continuous relative energy keeps the ordering in
the far tail and should be the default, with discrete bins as an ablation.
Neither version resolves near-degenerate polymorphs, and neither should.

Costs to keep in view: rank is database-relative, so a singleton formula's only
entry is rank 0 even at e_hull 0.5 (keep a coarse absolute e_hull alongside, or
apply the e_hull <= 0.1 filter first); and rank 0 of an unstable formula is still
unstable, so for de novo work both signals are wanted. Train with independent
dropout on each conditioning field so either can be queried alone. The
implementation handles the singleton problem by returning `polymorph_count`
alongside the delta, to be carried as a second channel or used to restrict
training to contested compositions.

**A cheap audit this enables.** A relative condition makes a teacher-forced,
oracle-free conditioning test well posed: for a held-out ground state with gene
`g`, compare `log p(g | ground-state condition)` against `log p(g | its true
offset)`. If conditioning was learned, the ratio is positive on held-out ground
states and negative on held-out polymorphs. This uses only genes and DFT energies
already in the cache: no CrySPR, no MLIP, no hull. It answers a *narrower*
question than [the existing audit](upi73i4k_ehull_conditioning_audit.md), which
had to relax because de novo samples have no ground-truth gene to score against.

## The exploration paradox

Worth recording because it shapes what any of this can claim. E_hull is a
statement about the database, not about a structure: it measures distance from
the frontier of what has been computed. Well-explored systems have the true
ground state already in the reference, so there is nothing to find; underexplored
systems are where new hull entries live. A de novo generator that samples
compositions from the data marginal and conditions on e_hull = 0 is pointed at
explored chemistry twice over.

The bias lives in the *composition* marginal, not in the structure conditional.
Prototypes recur across chemistry: an anonymised prototype from train gives the
right geometry for ~75% of MP-20 test structures, and 33k prototypes cover two
thirds of generated genes ([DoF study](pyxtal_dof_reduction_study.md)). So
"what a ground state looks like" transfers to systems nobody has computed;
"which compositions people compute" is the part not to learn. That argues for
factorising: p(structure | formula) from the generator trained on everything, and
the formula chosen by a separate policy allowed to disagree with database
frequency.

Ideas noted, none built:

- **Value the expected *drop* in the hull, not distance from it.** Expected
  improvement handles the paradox automatically: a well-supported system has a
  tight predictive distribution above the hull, so expected drop is near zero;
  a sparse system has wide spread, so it is positive.
- **Model-free measures of how provisional a hull is.** Leave-one-out
  sensitivity (drop each hull entry and remeasure) and recent movement between
  MP 2022 and LeMat-Bulk. Both caches are present.
- **A referee that is not "some guy".** Train on the old snapshot; let entries
  that later fell below the MP 2022 hull be the answer key. This is Matbench
  Discovery's temporal logic applied to generation, and it is immune to the
  circularity.
- **Self-consistency before claiming a find.** In a sparse system, run CSP on
  the decomposition products too and let the pipeline's own relaxed structures
  update the hull.

Caveat on "nothing to fish": that holds for the hull, not for metastable
polymorphs or for large low-symmetry cells, where the database is thin. Those are
also where WyFormer is weakest, which is the DoF work's territory.

## What the LeMat-GenBench leaderboard actually permits

Read from the paper and the leaderboard app on 2026-09-05. **These quotes came
through an automated page fetch and should be re-checked against the PDF before
being relied on in writing.**

- Exactly 2,500 structures: "To ensure eligibility for the leaderboard, please
  provide exactly 2,500 representative structures."
- Scored as submitted: "All evaluations are conducted on the submitted structures
  without re-relaxation."
- Pipeline relaxation is *requested*, not merely tolerated: "users are encouraged
  to submit structures that have already been relaxed as part of their generation
  pipeline, ensuring that stability metrics reflect the model's actual outputs
  rather than post-hoc refinement". Relaxed entries carry a flag; the submission
  form has a pre-relaxed checkbox and a relaxation-settings field.
- Optional verification: "authors may also submit their packaged model, granting
  a compliance badge (optional)".
- Efficiency disclosure is mandatory: training FLOPs and time, inference time for
  2,500 structures on a reference GPU, peak memory, parameter count.
- The budget-constrained task is deferred: "In this work, we focus exclusively on
  the unconditional generation setting."
- Nothing regulates filtering, guidance, or conditioning.

Table 1, and our own run for scale:

| model | Valid | Unique | Novel | SUN | MetaSUN |
|---|---:|---:|---:|---:|---:|
| MatterGen | 95.7 | 95.1 | 70.5 | 0.2 | **15.0** |
| DiffCSP | 95.7 | 94.8 | 66.2 | 0.1 | 8.5 |
| WyFormer-DFT | 95.2 | 95.0 | 66.4 | 0.4 | 7.8 |
| PLaID++ | 96.0 | 77.8 | 24.2 | **1.0** | 7.6 |
| DiffCSP++ | 95.3 | 95.1 | 62.0 | 0.2 | 5.0 |
| WyFormer (paper) | 93.4 | 93.0 | 66.4 | 0.1 | 1.9 |
| `upi73i4k`, MACE only, ours | 88.9 | 100 | 74.0 | 1.04 | 12.8 |

Our row is a single MLIP against the MACE hull and is **not** comparable: the
ORB+MACE+UMA ensemble is harsher (stable 1.5% under ORB vs 4.1% under MACE), and
the leaderboard denominator is all 2,500, so the 11% validity loss is a direct
cost. MatterGen, PLaID++ and both WyFormer rows are flagged as internally
relaxing, i.e. the top of the board is already pipelines rather than bare
samplers. PLaID++ is believed to be preference-aligned against MLIP stability of
its own samples; **this was not verified and must be checked before being cited.**

### A ladder of acceptability

1. **Uncontroversial.** Converged model, e_hull <= 0.1 dataset, capacity, lattice
   head, DoF-aware trial budget, validity fixes, and relaxing the final 2,500
   with the ensemble's own MLIPs. Re-relaxation alone was measured to add about a
   fifth to MetaSUN, and the benchmark asks for it.
2. **Sanctioned by precedent.** Steering inside the model: sampling at a
   condition target (already done), or KL-regularised preference fine-tuning
   (PLaID++). Disclose the target value.
3. **Defensible with disclosure plus the compliance badge.** A critic as a
   rejection stage *inside the packaged pipeline*, submitted for verification, so
   the 2,500 are by the benchmark's own mechanism "the model's actual outputs".
   Submit the raw model as a second row.
4. **Not acceptable.** Rejecting final structures on the ensemble MLIPs' e_hull
   (the metric's own oracle as post-hoc refinement), or tuning thresholds on
   oracle-scored samples of our own model without saying so.

**Training the critic on the scoring MLIP's labels was proposed and then
rejected** as textbook reward hacking: it fits the oracle's idiosyncrasies rather
than physics. LeMat-Bulk carries DFT e_hull for 4M entries, so there is no data
shortage to justify it.

### Narrative framing

Rejection inflates the "how good is the learned distribution" reading of SUN
while legitimately improving the "how many stable novel structures does the
pipeline yield" reading. The defence is to never let the two be confused:

- Define the method as a **sampler**, since a proposal with a fixed acceptance
  rule is a generative model of `q ∝ p·a`, and report the acceptance rate as an
  efficiency number.
- Make **cost the headline**: stable novel structures per oracle relaxation, and
  per GPU-hour. This is where the discrete representation is a genuine advantage,
  because screening happens before a 3D structure exists. Diffusion models cannot
  screen without paying for generation first.
- Run the **shared-critic control**: apply the same critic to every baseline's
  published structures (all are on Figshare in a uniform format) and report
  filtered SUN per sample for all methods. If WyFormer still wins, the proposal
  distribution is better; if it wins only per unit compute, say exactly that.
- Publish the **raw model at acceptance 1**, and one curve of SUN against
  acceptance rate, plus the distribution-level and prototype-level novelty that
  filtering will hurt.
- Treat the critic as a **falsifiable instrument**: predict the stable fraction of
  the accepted set before relaxing anything, then report predicted against
  observed, per DoF bin and separately for novel genes. Never tune that
  calibration on oracle-scored samples of our own model.

The strongest social move is to propose the deferred budget task to the
maintainers, with cost-normalised MetaSUN as the column, and to volunteer the
shared-critic table. The paper already mandates the inputs for it.

## Three objections to a critic, and the answers

**(a) Predicting formation energy or e_hull from a gene.** Uncontroversial. This
is Wren's task and Matbench Discovery's task.

**(b) "You are using a memorised hull lookup."** The test that separates this
from a novelty lookup is whether the lookup makes the metric vacuous. A novelty
filter perfectly determines the novelty score, leaving zero residual, which is
why it feels like cheating. The hull line determines nothing on its own: it is a
known function of composition, and the quantity to be predicted is the formation
energy of a structure that does not yet exist. Two supports: Matbench Discovery
is built exactly this way and nobody calls it hardcoding; and a direct e_hull
regressor memorises the same hull line implicitly, in the weights, with error, so
the explicit version is more accurate *and* more honest. Predict formation
energy, then decompose. A direct e_hull regressor also never sees a below-hull
example, so it can only call "stable" what resembles existing hull entries, which
are the non-novel ones.

**Superseded on the estimand.** This study said "predict formation energy" and
implicitly meant `E[E | gene]` under an MSE. [CSP mode](csp_mode.md) argues, and
implements, the better target: a gene fixes neither coordinates nor cell, so it
determines a *manifold* of structures, and an MSE fit lands on the average
structure of that manifold. What matters is `min(E | gene)`, the best energy the
gene can reach, because reconstruction gets several attempts at it. That target
is a genuine deterministic function of the gene with no noise floor, and it is
fitted with a censored likelihood because every dataset entry is only an upper
bound on it. Read that document before building any critic described here.

**(c) Rejection is the hardest to sell.** It need not be a rejection step at all.
Two exact constructions, neither implemented:

1. **Value-guided (twisted) decoding.** The tilted distribution factorises token
   by token over an autoregressive model:

   ```
   V(s)    = beta * log E_{p(x|s)}[ exp(r(x)/beta) ]      # soft value / reward-to-go
   q(a|s) ∝ p(a|s) * exp( (V(s∘a) - V(s)) / beta )
   ```

   Sampling from `q` gives *exactly* the rejection distribution with acceptance
   rate 1 and no discarded samples. The cascade vocabularies are small, so the
   per-step normalisation is exact and cheap. This is classifier guidance in
   discrete form, and MatterGen sits at the top of the board with a guidance term
   nobody calls a filter. Training `V` needs Monte Carlo rollouts from the base
   model, which is cheap precisely because gene generation is nearly free.
2. **Distillation into the weights.** `q* ∝ p·exp(r/beta)` is the exact maximiser
   of expected reward minus a KL penalty to the base model, so KL-regularised
   fine-tuning bakes the tilt into the parameters and leaves no inference-time
   machinery. The KL term is also what keeps the model off PLaID++'s row
   (uniqueness 77.8, novelty 24.2).

Separately, the "flatten the distribution to avoid duplicates" factor `1/p(x)^α`
is just temperature, since `p^(1-α)` is the model at temperature `1/(1-α)`. Note
that per-token temperature is not identical to sequence-level tempering, but it
is the standard proxy and needs no defence.

## What a critic could actually buy

### Is gene to energy learnable?

- **Prior art.** Wren (Goodall et al., *Science Advances* 2022, arXiv:2106.11132)
  is exactly this model: coordinate-free Wyckoff representations regressed onto
  energy. Verbatim: "The resulting model has a mean absolute error of 31 meV per
  atom on this test set", trained on the union of MP and WBM, ~322k materials, a
  random 5% test split. It found 1,569 below-hull materials from 5,675 ab-initio
  calculations, and reported 4.5-5.4x compute saving against random search. That
  is the same propose-screen-verify pipeline, published and uncontested.
  **Caution:** a random split is in-distribution. Wrenformer is believed to do
  substantially worse on Matbench Discovery's temporal split; **check this before
  citing.** Our critic faces novel genes, so expect the harder number.
- **Infrastructure already exists.** `TargetClass.Scalar` with `nn.MSELoss` and
  `predict_scalars` in `src/wyckoff_transformer/trainer.py`; 24 Scalar yamls, 41
  already using `target_name: formation_energy_per_atom`. A LeMat-Bulk critic is
  a config change, not new code.
- **Same-gene energy spread.** `analyse_gene_energy_critic.py signal`, LeMat-Bulk
  train to test: predicting a held-out entry's e_hull by the train mean over rows
  sharing its *exact gene* gives MAE 0.060 eV/atom at 6.9% coverage; same reduced
  formula gives 0.266 at 62.9%; global mean gives 0.425. Within-formula sd of
  e_hull is 0.262 mean / 0.142 median, so composition does not determine e_hull
  and the gene has real work to do.

  **Read the 0.060 carefully.** This study first called it "the noise floor a
  critic cannot beat". That is wrong under the `min(E | gene)` estimand: it is
  the spread of *observed* energies among entries sharing a gene, which is the
  width of the gene's manifold plus cross-source DFT disagreement, i.e. roughly
  the excess scale plus label noise in the censored model. The minimum itself has
  no noise floor. The 0.060 is still the right order for how far one observation
  of a gene sits from another, and [CSP mode](csp_mode.md) asks for exactly this
  quantity bucketed by degrees of freedom, keyed on the augmented fingerprint;
  this measurement uses a plain gene key and does not bucket it, so it is a first
  cut and not that measurement.

### The ceiling, measured on the real generated set

`analyse_gene_energy_critic.py enrichment` joins the 2,500 `upi73i4k` genes to
their measured ORB energies from `generated/upi73i4k/protocol/structures.csv`,
ranks them, and reports the MetaSUN rate of the top slices. Base rates reproduce
[the published funnel](upi73i4k_orb_protocol.md) exactly: MetaSUN 290/2500 =
0.1160, SUN 7/2500 = 0.0028.

**This is a *de novo* measurement and does not transfer to CSP.** MetaSUN
requires novelty, which is the binding constraint below and is precisely what
[CSP mode](csp_mode.md) says is beside the point when there is a right answer.
The ceiling reported here bounds what critic-shaped sampling could buy on the
leaderboard, not what regressor ranking buys for structure prediction. It is
also adjacent to, but not the same as, the go/no-go that document asks for: the
Spearman correlation of a *trained* `min(E | gene)` regressor against these same
relaxed energies. What is run here is a training-free proxy plus the oracle
ceiling, which is the cheaper half of that question.

| kept of 2500 | perfect energy oracle | novel fraction there | training-free formula critic |
|---:|---:|---:|---:|
| 100 | 0.220 (1.9x) | 0.220 | 0.170 (1.47x) |
| 250 | 0.332 (2.9x) | 0.332 | 0.168 (1.45x) |
| 500 | **0.422 (3.6x)** | 0.422 | 0.156 (1.34x) |
| 1000 | 0.290 (2.5x) | 0.553 | 0.144 (1.24x) |
| 2500 | 0.116 | 0.722 | 0.116 |

The training-free critic is the mean e_hull of LeMat-Bulk train rows sharing the
generated gene's reduced formula: 49.8% coverage, Pearson 0.255 and Spearman
0.362 against the measured energy. Ranking by the formula *minimum* instead
performs the same within noise (0.284 / 0.409, peak lift 1.40x).

**What is scarce, and where selection saturates.** Per sampled gene the two
ingredients are *not* equally common: novelty is abundant at 0.722, metastability
is scarce at 0.235. Selection therefore does the useful thing, converting the
abundant resource into the scarce one:

| kept of 2500 | metastable and valid | novel and valid | MetaSUN |
|---:|---:|---:|---:|
| 100 | 0.930 | 0.220 | 0.220 |
| 250 | 0.936 | 0.332 | 0.332 |
| 500 | **0.950** | 0.422 | **0.422** |
| 1000 | 0.588 | 0.553 | 0.290 |
| 2500 | 0.235 | 0.722 | 0.116 |

Going from 2500 to 500 raises metastability by 0.715 and costs 0.300 of novelty,
which is a good trade and the whole 3.6x. **Past that point stability saturates.**
It is already 0.950 at 500 kept and cannot go higher, so tightening to 100 gains
nothing on stability and still costs 0.202 of novelty, and MetaSUN falls from
0.422 to 0.220. That is why the oracle peaks near 20% acceptance and gets worse
below it, and why MetaSUN equals the novel fraction in the top slices.

**The anticorrelation is real and measured.** Over all 2500 genes,
P(novel | metastable) = 0.493 against P(novel | not metastable) = 0.793, a phi
coefficient of −0.284. Independence would give MetaSUN 0.170; the observed 0.116
is well below it. The lowest-energy genes are disproportionately rediscoveries,
which is the [exploration paradox](#the-exploration-paradox) showing up in the
funnel: the marginal band ranked 0-100 by true energy is 93.0% metastable but
only 22.0% novel, while the band 250-500 is 96.4% metastable and 51.2% novel.

Consequences:

- **Tune acceptance near a fifth, not a twentieth.**
- **Even a mediocre critic clears the board.** A perfect energy ranking reaches
  0.422 against MatterGen's 0.150.
- **Beyond ~20% acceptance, a better critic is worth nothing.** Stability is
  already 0.950 there, so no critic can add stability and the only way up is more
  novel genes in the low-energy region. That is a proposal problem, not a critic
  problem, and it argues for widening the proposal (higher temperature, weaker
  conditioning) rather than sharpening it, which is the opposite of what sampling
  at e_hull = 0 does.
- **Reward expected MetaSUN, not energy.** Perfectly identifying every MetaSUN
  gene would give 0.580 at 500 kept, so energy ranking alone captures 73% of what
  is achievable. The clean way to handle the novelty half is to put it in the
  fine-tuning reward rather than in an inference-time lookup, which converts the
  objectionable filter into learned behaviour.
- **A gene-level critic can never beat this oracle**, because the oracle already
  knows the realised structure's energy. Its own ceiling is lower still, set by
  realisation noise: the critic predicts the energy of the gene's *intended*
  structure, while what gets scored is CrySPR's realisation. At dof 0 these
  coincide; at dof > 10 they may barely correlate. The critic and the DoF work
  are therefore the same bottleneck seen from two sides.

### On the i.i.d. constraint

If sampling must be independent (no active learning, no oracle in the loop), the
whole policy is a fixed per-sample acceptance rule, and the tilted-decoding
constructions above satisfy it. Independence holds because every factor is a
fixed function of the gene and of things available before the campaign: the
training database, regressors trained on it, and recovery rates measured on
database structures. Two details matter. Taking the top N of a pool is *not*
i.i.d., since those are order statistics, whereas a fixed threshold is. And
choosing among k CrySPR realisations of one gene by MLIP energy is relaxation,
which belongs in the relaxation-settings field, not a filter on the metric.

## Not proposed

- Training the critic on the scoring ensemble's MLIP labels (reward hacking).
- Rejecting final structures on ensemble e_hull.
- Filtering on a novelty lookup to inflate the novelty rate. Novelty belongs in
  the fine-tuning reward instead.
- Absolute-e_hull conditioning as the CSP condition; use relative energy.
- Discarding non-ground-state training rows (47% of LeMat-Bulk).
- An MSE-fitted `E[E | gene]` critic. Superseded by the censored `min(E | gene)`
  design in [CSP mode](csp_mode.md) before this document was finished.

## Still open after the CSP mode landed

The CSP work implemented the composition constraint, the relative-energy label
and the ranking regressor. It did **not** touch anything in the second half of
this study, which remains proposal only:

- No critic has been trained, so none of the enrichment numbers below the oracle
  row have been realised.
- No value-guided or twisted decoding exists, and no KL-regularised fine-tuning.
- No sampling policy, no acceptance rule, no shared-critic control against the
  published baselines.
- Nothing has been submitted to LeMat-GenBench, and the quoted leaderboard rules
  are still unverified against the PDF.

## Reproduce

```bash
uv run python scripts/analyse_gene_energy_critic.py uniqueness --cache cache/mp_20/data.pkl.gz   # ~10 s
uv run python scripts/analyse_gene_energy_critic.py uniqueness                                    # LeMat-Bulk, ~5 min
uv run python scripts/analyse_gene_energy_critic.py signal                                        # ~4 min
uv run python scripts/analyse_gene_energy_critic.py enrichment                                    # ~3 min
```

Most of the wall time is unpickling the 4M-row LeMat-Bulk cache (~60 s) and
building its keys (~25 s). All four are single-process pandas and respect the
machine's core budget.

## See also

- [CSP mode](csp_mode.md) — **what was actually built** from the first half of this study, and authoritative wherever the two disagree
- [`upi73i4k` under the ORB ranking protocol](upi73i4k_orb_protocol.md) — the funnel these enrichment numbers rank, the 98.4% gene-to-structure uniqueness
- [e-hull conditioning audit](upi73i4k_ehull_conditioning_audit.md) — why absolute e_hull conditioning does not separate 0 from 0.05
- [PyXtal DoF reduction study](pyxtal_dof_reduction_study.md) — prototype transfer, the realisation-noise side of the critic's ceiling
- [Improving de novo quality](de_novo_quality_plan.md) — the plan this study would slot into
- [De novo ranking protocol](de_novo_ranking_protocol.md) — the cheap ranking instrument and its leaderboard deviations
- [Experiments plan](experiments_plan.md) — backbone item "Wyckoff Regressor"
