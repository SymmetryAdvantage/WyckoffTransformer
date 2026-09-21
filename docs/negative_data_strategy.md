# What to do with the 4M unstable structures

Analysis 2026-09-21, `main` @ `b4b88e6`. Runs
[`unconditional_5x_ehull01-20260915-151250`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/unconditional_5x_ehull01-20260915-151250)
(artifact `protocol_unconditional_5x_ehull01-20260915-151250:v0`) and
[`ehull_adamw_wsd_5x-20260912-115321`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x-20260912-115321)
(artifact `protocol_ehull_adamw_wsd_5x-20260912-115321:v2`).

`lemat_bulk_fmax1_stress` holds 5.33M structures, of which 1.64M sit at
`e_hull <= 0.1` eV/atom and 3.69M above it. The design intent of the
`energy_above_hull` conditioning channel was that the unstable 70% would teach
the generator what *not* to emit.

**Revised 2026-09-21** after
[ehull_adamw_wsd_5x_ehull005_protocol.md](ehull_adamw_wsd_5x_ehull005_protocol.md)
sampled the same conditioned model at `e_hull = 0.05` instead of `0.0`. The
first version of this document concluded that the conditioning does not
transfer stability to novel genes. That was wrong: it was an artefact of the
target value. Sampled at 0.05 the conditioned model reaches parity with the
filtered model on every metric that matters. What survives is the *mechanism* —
querying the conditional at the edge of its support turns it into a retrieval
index — and the strategic conclusion, which the new result strengthens rather
than weakens: even queried correctly, 3.69M extra rows and 3.3x the compute buy
**parity**, not gain. The negatives still earn nothing as generator training
data.

Related: [dirty_data_conditioning.md](dirty_data_conditioning.md) argues for
conditioning over filtering for a *provenance* channel (`max_force`); that
argument is about not discarding experimentally grounded rows and is not in
conflict with this one. See [§2.4](#24-not-every-conditioning-channel-is-the-same).

---

## 1. The measurement

Both protocol runs used the same novelty reference
(`lemat_bulk_fmax1_stress` train+val+test, 4,826,004 fingerprints), the same
ORB `orb_conserv_inf` hull at revision `70d505bb`, the same trial schedule and
1000 sampled genes at temperature 1.0. The conditioned run was generated at
`energy_above_hull = 0.0`, the `DEFAULT_CONDITION_TARGETS` default that
`cli/protocol_wandb.py` applies to any conditioned model.

All rates below are over the structures the protocol actually produced
(`has_structure`, n ≈ 995), so they differ in the third decimal from the
per-sampled-gene rates in the protocol docs.

| | ehull01 (filtered, uncond.) | cond @ 0.00 | cond @ 0.05 |
| --- | ---: | ---: | ---: |
| MetaSUN | **0.420** | 0.267 | 0.395 |
| metastable | **0.658** | 0.588 | 0.628 |
| novel structure | **0.732** | 0.649 | 0.712 |
| P(metastable \| novel structure) | **0.573** | 0.412 | 0.555 |
| gene novelty (MLIP-free) | 0.592 | 0.554 | **0.605** |
| P(metastable \| gene **known**) | 0.798 | **0.804** | 0.748 |
| P(metastable \| gene **novel**) | **0.561** | 0.415 | 0.549 |
| median e_hull, known genes | 0.037 | **0.020** | 0.052 |
| median e_hull, novel genes | **0.089** | 0.124 | 0.092 |
| fraction at e_hull <= 0.001 | 0.071 | **0.133** | 0.051 |
| fraction above 0.3 eV/atom | **0.048** | 0.088 | 0.050 |
| cubic-prototype space groups | **0.068** | 0.217 | 0.159 |

Two-proportion z, cond @ 0.05 against the other arms:

| | vs cond @ 0.00 | vs ehull01 |
| --- | ---: | ---: |
| MetaSUN | +0.128, **z = 6.1** | −0.024, z = −1.1 |
| metastable | +0.039, z = 1.8 | −0.030, z = −1.4 |
| gene novelty | +0.050, **z = 2.3** | +0.012, z = 0.6 |
| P(metastable \| novel gene) | +0.134, **z = 4.6** | −0.012, z = −0.4 |

**Moving the target from 0.00 to 0.05 recovers the whole gap.** On every metric
the conditioned model at 0.05 is statistically indistinguishable from the
filtered model, and decisively better than itself at 0.00. The e_hull quantiles
say the same thing — the 0.00 arm is the odd one out, and it is odd at *both*
ends:

| decile | 10% | 25% | 50% | 75% | 90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ehull01 | 0.005 | 0.025 | 0.067 | 0.132 | 0.225 |
| cond @ 0.00 | 0.000 | 0.015 | 0.070 | **0.166** | **0.282** |
| cond @ 0.05 | 0.012 | 0.035 | 0.073 | 0.136 | 0.220 |

Conditioning at 0 produces a **bimodal** output — a memorised spike on the hull
plus a fatter unstable tail — with an unchanged median. Conditioning at 0.05
produces a unimodal distribution almost identical to the filtered model's.

**So the conditioning channel is a working control knob, not a broken one.** It
moves the energy distribution monotonically and reproducibly: raising the target
from 0.00 to 0.05 trades on-hull hits (stable 0.064 → 0.027) for metastable
yield and novelty. What fails is querying it at `e_hull = 0`, where the
conditional is dominated by the memorised rows that actually carry that label.
The signature of that failure is visible at the generation stage, before any
MLIP is involved: gene novelty 0.554 at target 0 against 0.605 at target 0.05,
and gene-known share 0.445 against 0.394.

What the conditioning does **not** buy is any advantage over simply filtering
the training set. Parity is the ceiling observed so far.

### 1.1 Confound: compute

Not matched: 131,613 optimiser steps for ehull01 against 40,000, and 4,175
against 390 passes over the respective training sets. Both validation curves
were still descending at the end; ehull01's val NLL was 19.88 at step 39.5k and
18.09 at the end.

This now matters *more*, not less. The conditioned model reaches parity with
ehull01 while having had 3.3x fewer optimiser steps, which if anything argues
that conditioning + full data is the more compute-efficient route — or that
ehull01 is over-trained for its dataset size. The cheap check is unchanged: run
the protocol on the ehull01 `best_model` artifact version nearest 40k steps
(v3/v4, 2026-09-15), roughly one GPU-hour including relaxation.

### 1.2 Confound: platform, and which way it cuts

The three arms were not scored on the same machine. `ehull01` and `cond @ 0.05`
ran on **iapetus** (6 cores, 2x K20c + GTX 750 Ti, device budgets 4032/4032/1701,
a split CPU-neighbour / GPU-forward ORB implementation); `cond @ 0.00` ran on
**aspire2a** (4 GPUs, budgets 34376). The MLIP checkpoint, hull revision
(`70d505bb`), novelty reference, `fmax`, tolerance factor and trial schedule are
identical across all three, but the platform is *perfectly confounded* with the
interesting contrast: the two arms that agree share hardware and the outlier
does not.

Measured on the most matched population available — known gene **and** zero
positional DoF, where the structure is essentially fixed by symmetry and the
relaxation has almost nothing to do, so the number is close to a pure calculator
readout:

| arm | n | median e_hull | fraction <= 0.001 |
| --- | ---: | ---: | ---: |
| ehull01 (iapetus) | 58 | 0.052 | 0.172 |
| cond @ 0.00 (aspire2a) | 121 | 0.026 | 0.273 |
| cond @ 0.05 (iapetus) | 132 | 0.057 | 0.121 |

iapetus reads roughly 0.025–0.030 eV/atom **higher** on this stratum. The gene
mix still differs within it, so this is suggestive rather than a calibration,
but the direction is consistent and it determines which conclusions are safe:

- **Conservative, therefore safe.** `cond @ 0.00` had the *favourable*
  calculator and still came last on metastability, MetaSUN, the fat tail
  (0.088 above 0.3 eV/atom) and median e_hull on novel genes. Correcting for the
  offset would widen the 0.05-vs-0.00 gap, not close it. The same applies to the
  original ehull01-vs-cond@0.00 comparison: ehull01 was penalised and still won.
- **Not safe.** Threshold-at-zero statistics are exactly where a −0.025 eV/atom
  offset bites hardest. The claim that `cond @ 0.00` is better at *sharp
  retrieval* — 0.133 at `e_hull <= 0.001` against 0.071/0.051, and SUN 0.012
  against 0.008/0.009 — is substantially calculator and should not be relied on.
- **Unaffected.** Gene novelty and gene-known share are computed from the
  generator's output against the fingerprint reference with no MLIP involved.
  The 0.554 → 0.605 shift with target is pure model behaviour.

To close this properly, re-relax one existing cohort on the other machine — the
baseline's 2435 trials on iapetus is the cheaper direction — and compare e_hull
per trial. Until then, treat cross-platform threshold counts as soft.

---

## 2. Why the target value mattered so much

### 2.1 Coverage, not control

This is return-conditioned supervised learning: train on (outcome, sample)
pairs, condition on a desirable outcome at sampling time. Brandfonbrener et al.
(NeurIPS 2022) show RCSL recovers the optimal policy only under assumptions
strictly stronger than dynamic programming needs — near-deterministic outcomes
and **coverage of the conditioning value**.

The target sweep separates the two assumptions cleanly, and it is coverage that
bit:

- **Coverage — this is the one that failed.** `e_hull = 0` is ~3% of rows, at
  the edge of support. `e_hull = 0.05` sits in a well-populated part of the
  distribution, and the same model queried there behaves normally. A 0.05 eV/atom
  move in the target is worth +0.128 MetaSUN.
- **Determinism — soft, but not fatal.** A Wyckoff gene does not determine
  `e_hull`: it determines a manifold of structures (`censored.py` is built on
  exactly this observation) and the hull it is measured against lives at other
  compositions. This is why the control is soft — the target shifts the
  distribution rather than pinning it, and the median barely moves — but soft
  control is still control, and §1 shows it is enough.

### 2.2 Rare conditioning values induce memorisation

Work on the geometry of memorisation in generative models finds that
conditioning on highly specific values lowers the local intrinsic dimension of
the sampled manifold and raises the probability of reproducing training samples.

The sweep is a direct confirmation, on quantities that are independent of the
MLIP and of the platform:

| | target 0.00 | target 0.05 |
| --- | ---: | ---: |
| gene-known share | 0.445 | 0.394 |
| gene novelty | 0.554 | 0.605 |
| cubic-prototype space groups | 0.217 | 0.159 |
| median e_hull on known genes | 0.020 | 0.052 |

The retrieval pull is monotone in how extreme the target is. Ask for the value
that only memorised rows carry and you get the memorised rows.

### 2.3 Maximum likelihood still has no "don't"

Every term of `-log p(x)` pushes probability up. Nothing in the objective pushes
a sequence down except through the normalisation constant. A condition channel
lets the model *partition* mass into `p(x|c)`; nothing forces it to put unstable
genes exclusively under `c > 0`.

This is no longer an explanation of the measured failure — §1 shows the
partition is good enough to match a filtered model. It is the explanation of the
measured *ceiling*: 3.69M extra rows and a conditioning channel reach parity
with throwing those rows away. The negatives are being modelled, not learned
from. Everything in §3 onwards is about changing that.

### 2.4 Not every conditioning channel is the same

Korbak et al. (ICML 2023) ran the closest published version of this experiment
for LM pretraining — MLE, MLE on filtered data, conditional training with
`<|good|>`/`<|bad|>` control tokens, unlikelihood, RWR and AWR — and conditional
training was *Pareto-optimal*, about 13× better than MLE on toxicity, while
filtering paid the largest capability penalty of any method. Conditioning is not
intrinsically weaker than filtering.

What differs here:

| | Korbak et al. | this run |
| --- | --- | --- |
| control signal | binary token at a threshold | continuous scalar, sampled at its extreme |
| score | near-deterministic function of the text | not a function of the gene at all |
| filtering baseline | starved of data | 1.64M rows, ample |

The first row is the one the target sweep vindicated: a binary token at a
threshold cannot be queried off-support, and a continuous scalar can. The third
row is why parity rather than a win is the outcome here — Korbak's conditional
training beat filtering largely because filtering cost capability, and 1.64M
rows is enough that it does not.

The same distinction separates this document from
[dirty_data_conditioning.md](dirty_data_conditioning.md): `max_force` is a
provenance label attached to the row, so conditioning on it keeps rows that a
filter would discard for the wrong reason. `energy_above_hull` is the objective
itself, and asking the generator to internalise it is asking a ~150k-parameter
model to memorise a global convex hull. (`max_force` conditioning has its own
problem — `max_force == 0` occurs only on zero-positional-DoF rows, so
conditioning at 0 doubles the zero-DoF gene rate. Do not use it.)

---

## 3. Where a negative gradient can actually come from

| family | mechanism | negative gradient? | cost |
| --- | --- | --- | --- |
| conditioning | partition mass by `c` | no | current |
| filtering | drop the rows | n/a (absent) | current |
| RWR / AWR | reweight the MLE terms | no — see §4 | one training run |
| unlikelihood | `+log(1-p)` on bad rows | yes, in training | one training run |
| DPO / KTO / **NCA** | contrastive, reference-relative | yes, in training | one training run |
| residual EBM (NCE) | sequence-level energy in the residual | yes, in training | one training run + resampling |
| FUDGE / DExperts / discriminator guidance | Bayes' rule at sampling time | yes, at inference | one small extra model |
| critic + best-of-n | rank before the expensive stage | n/a | no generator change |

Materials precedents on Wyckoff-based autoregressive models, both with a real
negative gradient:

- **PLaID++** (2025): Wyckoff-text LLM, iterative DPO with tiered pairs
  (stable ≻ metastable ≻ unstable, one metastable and two unstable sampled per
  stable crystal). SUN 9.85% → 22.27%. Uses dynamic sampling temperature across
  iterations specifically to prevent diversity collapse.
- **CrystalFormer-RL** (Phys. Rev. B 2025): Wyckoff autoregressive transformer,
  PPO against an Orb-v2 `e_hull` reward with a KL term. Stable 44.7% → 73.4%,
  SUN 15.3% → 21.6% — and it beats supervised fine-tuning on stable structures
  only (18.3% SUN), which is the head-to-head this document is about.

---

## 4. Why reweighting is not the fix

Reward-weighted regression maximises `Σ_i w_i log p(x_i)` with
`w_i = exp(-β·e_hull_i) > 0`. Every term still pushes probability up on an
unstable row, so the objection is correct on its face. Two things need to be
said precisely.

**It does suppress, but only relatively.** The model is normalised, so pushing
up on one sequence pushes down on all others: at each cascade step
`∇ log p(token) = e_token - p(·)`, which is a negative gradient on every
competing token, scaled by the model's current probability. The fixed point of
weighted MLE is

```
p*(x)  ∝  p_data(x) · exp(-β·e_hull(x))
```

— the exponential tilt of the data distribution, i.e. the same target GDC
(Khalifa et al., ICLR 2021) writes down explicitly. Relative to unweighted MLE,
unstable genes end up with less mass than they would have had.

**But it can never beat filtering at suppression.** The relevant counterfactual
is not `w = 1` (unweighted MLE); it is `w = 0` (the filtered run). A small but
non-zero weight leaves a small but non-zero share of the mass on unstable genes.
The support of `p*` is still the support of `p_data`. There is no setting of β
at which RWR assigns an unstable gene *less* probability than a model that never
saw it. Whatever advantage reweighting has must come from somewhere else:

1. **Shared structure.** An unstable gene is still a *valid* gene — the space
   group / Wyckoff letter / multiplicity arithmetic and the chemistry are the
   same. At small weight the negatives act as an auxiliary task that regularises
   the tail of the distribution where the stable set is sparse. This is the
   capability loss Korbak measured on filtering.
2. **Continuity.** A hard cut at 0.1 discards the information that 0.11 is
   nearly as good as 0.09 and 3.0 is far worse.
3. **A measurable knob**, with filtering as the β → ∞ limit.

**And it degrades into filtering plus noise.** Once β is large enough to matter,
the weights concentrate on a few rows and the effective sample size collapses,
which is the standard critique of importance-weighted training. In this codebase
there is a sharper version of the same problem: run `lu4xqw0w` showed that
per-step loss reweighting is annihilated by `clip_grad_norm`, so the weights
have to enter through the **sampler** — at which point sampling rows with
probability ∝ `w` is *literally* a stochastic filter.

**Conclusion.** RWR/AWR is a smoother filter, not a new mechanism. It is worth
running as a cheap baseline that measures the representation-sharing benefit of
the negatives; it is not a way to learn what not to generate. Korbak et al.
finding RWR/AWR below conditional training is consistent with this.

---

## 5. Recommendation: NCA

Of the non-RL options, **NCA** (Noise Contrastive Alignment, Chen et al.,
NeurIPS 2024) is the best fit. It is the minimal repair of the AWR objective:
it keeps the exponential-tilt weights and adds the missing decreasing force.

```
L_NCA(x, {y_i, r_i}_1..K) =
    - Σ_i [  softmax_i(r/α) · log σ( r_θ(x, y_i) )      ← increasing force
           +      (1/K)     · log σ(-r_θ(x, y_i)) ]     ← decreasing force

    with   r_θ(x, y) := β · log( π_θ(y|x) / µ(y|x) ),   µ = reference policy
```

The first term carries weight `exp(r_i/α) / Σ_j exp(r_j/α)` — the same
exponential tilt RWR uses. The second applies a uniform `1/K` *decreasing* force
to every sample in the group. The net direction at initialisation is
`softmax_i(r/α) - 1/K`, so any sample whose reward is below the group average
receives a **genuinely negative gradient**. That is the whole difference from
§4, in one term.

Why NCA rather than the alternatives:

- **It is built for scalar rewards.** No pair construction, no threshold to
  choose: `e_hull` is already a float and it goes straight in. DPO and KTO need
  pairs or a binary label; InfoNCA/DPO also exhibit the decreasing-likelihood
  pathology, which NCA fixes by optimising absolute rather than relative
  likelihood — important here, since the stable manifold must get *sharper*, not
  merely relatively favoured.
- **It is fully offline.** No rollouts during training, no learned reward model.
  Every alignment method that needs on-policy samples would put PyXtal + MLIP
  relaxation in the training loop. The DFT `e_hull` labels are already on disk
  and are better than any reward model we could fit.
- **Its optimum is an exponential tilt of the reference policy**,
  `π* ∝ µ(y|x)·exp(r/α)`, which is a distribution-matching target rather than an
  argmax. It cannot collapse onto retrieval the way reward maximisation can —
  the precise failure mode §1 documents.

### 5.1 The grouping, which is the non-obvious part

NCA needs K ≥ 2 samples per group: at K = 1 the softmax weight and `1/K` are
both 1 and the two forces cancel exactly. So the question is what plays the role
of the prompt.

**Group by composition.** The competing polymorphs of one formula are exactly
the candidates `e_hull` compares, so within a group the ranking *is* determined
by the structures — which repairs the determinism failure of §2.2 instead of
working around it. LeMat already carries `delta_e_polymorph` for this, and it is
already a `sequence_fields.no_processing` entry in the tokeniser.

Notes on the implementation:

- The grouping key need not be something the model conditions on. The loss only
  requires a set of K samples sharing a group; `r_θ` is a per-sample likelihood
  ratio either way. Grouping by composition while the model remains
  space-group-started is well defined, and what it asks for is exactly right:
  put more mass on the hull member than on its competitors at the same formula.
- `µ` is the reference policy — the current ehull01 checkpoint, frozen.
- The likelihood must be the permutation-marginalised gene likelihood, not a
  single token sequence. `gene_likelihood.py` already computes it
  (log-mean-exp over the set of representations `R(G)`); budget for its cost.
- Groups of unstable-only compositions still carry signal — the group mean is
  what the decreasing force is measured against.

### 5.2 Cheaper things to do first

Both are strictly additive and neither touches the generator.

1. **Retrain the gene critic on all 5.33M rows and add best-of-n reranking.**
   The unstable rows are the negatives that make a critic learnable at all; a
   stability predictor trained only on stable structures has no contrast to rank
   with. `censored.py` + `gene_energy.py` + `wyformer-gene-screen` already
   implement the right target — `min(E | gene)` under a censored likelihood that
   treats observed energies as upper bounds on the manifold minimum. Predict
   **formation energy** and build the hull afterwards, as Wren does: E_form is a
   function of the structure, `e_hull` is not.

   This is what Parackal et al. (arXiv:2601.21393, 2026) ride on — Wren,
   trained on Alexandria's ~4M stable *and* unstable entries, prescreened 39
   billion enumerated Wyckoff protostructures down to 15M, recovering 83.6% of
   Alexandria's known stable phases. The protocol currently spends ~2.4 PyXtal +
   ORB trials on each of 1000 genes; generating 50k genes and keeping the top
   1000 by critic score costs almost nothing and spends the same relaxation
   budget on a far better slate.

2. **Anti-expert guidance.** Train a twin of ehull01 on the `e_hull > 0.1`
   complement and sample from `softmax(z + α(z⁺ - z⁻))` (DExperts, Liu et al.,
   ACL 2021). This is the literal implementation of "what not to generate": a
   token survives only if the expert likes it *and* the anti-expert does not.
   At ~150k parameters a second model is free, and α is a diversity/fidelity
   knob measurable against the same 1000-gene protocol. FUDGE (Yang & Klein,
   NAACL 2021) is the variant that trains one discriminator on *partial*
   sequences instead, which maps onto the cascade directly; Kim et al.
   (ICML 2023) give the theory — at the optimal discriminator the log-ratio
   correction moves the model score to the data score.

   Guidance trades fidelity for diversity, which is the novelty collapse of §1
   in a form that has a dial. Restricting guidance to part of the trajectory
   recovers much of the lost diversity (Kynkäänniemi et al. 2024); the cascade
   analogue is guiding the element and site-symmetry stages only.

### 5.3 Keeping the conditioning — now the cheapest win available

Item 1 below is no longer a proposal; it is the measured result of §1, and it
cost one protocol run.

0. **Finish the target sweep.** 0.00 and 0.05 are two points on a curve whose
   optimum is not yet bracketed. 0.03, 0.07 and 0.10 are one protocol run each
   (~1 GPU-day on iapetus), and the metastability threshold is 0.1, so the
   optimum is unlikely to lie above it. Run them on **one machine**, with a
   matched re-run at 0.00, so the curve is internally consistent — §1.2.
1. Threshold or quantile tokens instead of a continuous target. `<= 0.1` is
   on-support (30% of rows); `= 0` is not. This is the structural version of the
   sweep: it removes the ability to ask for an uncovered value at all.
2. Condition on what the gene controls — formation energy, or the censored
   `min(E | gene)` — not on `e_hull`.
3. Null-condition dropout with CFG (already in flight as
   `ehull_adamw_wsd_5x_cfg-20260916-055000`), with a stage-wise guidance
   schedule to protect diversity.
4. Smooth or noise the condition label so it cannot serve as a row index. Note
   that `e_hull` references differ across MP / Alexandria / OQMD, so the channel
   is already noisy in an uncontrolled way — see
   [e_hull_definitions.md](e_hull_definitions.md).

---

## 6. Order of work

1. **Finish the target sweep on one machine** (§5.3.0), with a matched re-run at
   0.00. Cheapest remaining gain, and it is also the control that closes §1.2.
2. Critic on all 5.33M + best-of-n reranking (§5.2.1). One training run, no
   generator change, testable against the same 1000-gene budget.
3. Anti-expert / FUDGE guidance (§5.2.2). One extra small model; tests "what not
   to generate" as an explicit mechanism.
4. NCA fine-tuning of the ehull01 checkpoint, grouped by composition (§5). The
   one that needs real implementation work, and where the published gains are.
5. Matched-compute control: protocol on the ehull01 checkpoint nearest 40k
   steps, to close out §1.1.

The bar every method in §3 has to clear has moved. MetaSUN 0.267 was never the
baseline; it is 0.395–0.420, whichever of the two arms you prefer, and those two
are indistinguishable. Anything that only recovers parity with filtering is not
worth the implementation.

The conditioned model is worth keeping for a CSP or retrieval mode, where
sampling at `e_hull = 0` and getting back archive-adjacent hull entries is the
desired behaviour rather than the failure. Note that the size of that retrieval
advantage is not yet established — see §1.2.

---

## References

- Brandfonbrener et al., *When does return-conditioned supervised learning work
  for offline RL?*, NeurIPS 2022. [arXiv:2206.01079](https://arxiv.org/abs/2206.01079)
- Korbak et al., *Pretraining Language Models with Human Preferences*, ICML 2023.
  [arXiv:2302.08582](https://arxiv.org/abs/2302.08582)
- Chen et al., *Noise Contrastive Alignment of Language Models with Explicit
  Rewards*, NeurIPS 2024. [arXiv:2402.05369](https://arxiv.org/abs/2402.05369)
- Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*,
  2024. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- Khalifa et al., *A Distributional Approach to Controlled Text Generation*,
  ICLR 2021. [arXiv:2012.11635](https://arxiv.org/abs/2012.11635)
- Deng et al., *Residual Energy-Based Models for Text Generation*, ICLR 2020.
  [arXiv:2004.11714](https://arxiv.org/abs/2004.11714)
- Welleck et al., *Neural Text Generation with Unlikelihood Training*, ICLR 2020.
  [arXiv:1908.04319](https://arxiv.org/abs/1908.04319)
- Yang & Klein, *FUDGE: Controlled Text Generation With Future Discriminators*,
  NAACL 2021. [aclanthology 2021.naacl-main.276](https://aclanthology.org/2021.naacl-main.276/)
- Liu et al., *DExperts: Decoding-Time Controlled Text Generation with Experts
  and Anti-Experts*, ACL 2021. [aclanthology 2021.acl-long.522](https://aclanthology.org/2021.acl-long.522/)
- Kim et al., *Refining Generative Process with Discriminator Guidance in
  Score-based Diffusion Models*, ICML 2023. [PMLR v202](https://proceedings.mlr.press/v202/kim23i.html)
- Peng et al., *Advantage-Weighted Regression*, 2019.
  [arXiv:1910.00177](https://arxiv.org/abs/1910.00177)
- Dong et al., *RAFT: Reward rAnked FineTuning*, 2023.
  [arXiv:2304.06767](https://arxiv.org/abs/2304.06767)
- Goodall et al., *Rapid discovery of stable materials by coordinate-free coarse
  graining* (Wren), Sci. Adv. 2022.
  [10.1126/sciadv.abn4117](https://www.science.org/doi/10.1126/sciadv.abn4117)
- Parackal et al., *Screening 39 billion protostructures for materials
  discovery*, 2026. [arXiv:2601.21393](https://arxiv.org/abs/2601.21393)
- *PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials
  Design*, 2025. [arXiv:2509.07150](https://arxiv.org/abs/2509.07150)
- *CrystalFormer-RL: Reinforcement Fine-Tuning for Materials Design*,
  Phys. Rev. B 2025. [arXiv:2504.02367](https://arxiv.org/abs/2504.02367)
- *CrystalREPA: Transferring Physical Priors from Universal MLIPs to Crystal
  Generative Models*, 2026. [arXiv:2605.08960](https://arxiv.org/abs/2605.08960)
- Zeni et al., *A generative model for inorganic materials design* (MatterGen),
  Nature 2025. [10.1038/s41586-025-08628-5](https://www.nature.com/articles/s41586-025-08628-5)
