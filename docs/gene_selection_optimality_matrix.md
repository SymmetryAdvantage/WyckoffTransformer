# Optimal gene selection, as a function of what we assume and what we can measure

> **Scope:** how to spend a fixed relaxation budget to maximise the number of
> **unique, novel, metastable** structures when selection happens at the gene
> level. The answer is not one formula; it changes with three assumptions, and
> this note writes down the optimum in each case, what each case can estimate
> from data, and what it has to assume. §§0–6 take the generator to be
> unconditional, which is the baseline; §7 covers what conditional generators
> change, and which formulas survive.
>
> **Measured 2026-09-12**: every item of §11 that could be answered without
> training a new generator has been, in
> [`gene_selection_measurements.md`](gene_selection_measurements.md). Read it
> alongside §2 — the prize is smaller than this note assumes, and the
> `§ measured` callouts below say where.
>
> Companion notes: [`de_novo_ranking_protocol.md`](de_novo_ranking_protocol.md)
> (the protocol this scores against), [`dft_fixed_hull_attack.md`](dft_fixed_hull_attack.md)
> (the fixed-hull objective), [`lemat_bulk_source_databases.md`](lemat_bulk_source_databases.md)
> (the archive's construction, which is assumption row 3).

---

## 0. The objective

Genes are free to draw; relaxations cost 1 each. Fix a budget $N$ relaxations,
choose $n_k \ge 0$ restarts for each candidate gene $k$ with $\sum_k n_k = N$,
and maximise the expected count of unique novel metastable structures.

Three structural facts shape every answer below.

**Novelty is a gene-level Bernoulli shared across restarts.** All realisations of
a gene lie on one manifold and relax into the same basin family, so restarts do
not decorrelate novelty. Budget buys metastability, never novelty; $\nu(g)$
factors out as a constant multiplier on the gene's whole reward curve.

**The per-gene reward is an indicator, not a count.** Two successful restarts of
one gene are one unique structure. Each gene's value curve is therefore concave
in $n$, which is what makes the allocation problem tractable.

**Both target events are deterministic given the gene and a fixed protocol.**
Novelty is a membership check; metastability is "relax, compare to the hull".
The probabilities below are therefore epistemic — degrees of belief about
deterministic facts, relative to a stated information set. This is not a defect:
a *benchmark is a stipulated information set*, which is exactly what axis three
formalises. The only genuinely aleatoric quantity in the problem is the
realisation seed.

---

## 1. Notation

| symbol | definition | units | source |
| :--- | :--- | :--- | :--- |
| $g$ | Wyckoff gene: space group + multiset of (element, site symmetry, enumeration) | — | generator output |
| $p_\theta(g)$ | probability the generator emits **that gene** — summed over token orderings and equivalent enumerations, times the space-group start-token prior | — | `exp(-surprisal)`, `gene_likelihood.py`; use `log_likelihood`, **not** `log_likelihood_elbo` |
| $M$ | gene draws in the reference archive (rows in the train split) | count | 5,127,874 for `lemat_bulk_fmax1` train |
| $\nu(g)$ | probability the gene is **absent** from the archive | — | §8; tilt in §6.1 |
| $m(g)$ | lowest formation energy attainable anywhere on the gene's manifold | eV/atom | `censored.py`, head 1 |
| $s(g)$ | mean excess of one random realisation above $m(g)$ | eV/atom | `censored.py`, head 2 |
| $\tau(g)$ | **epistemic** uncertainty of the critic about $m(g)$ | eV/atom | critic predictive spread |
| $h(X)$ | archive convex-hull energy at the gene's composition | eV/atom | `HullLookup`, frozen reference |
| $t(g)$ | acceptance threshold, $h(X(g)) + 0.1$ | eV/atom | computed |
| $\delta(g)$ | margin, $t - \hat m$; if $\delta \le 0$ the score is 0 | eV/atom | computed |
| $z$ | $\delta/\tau$ — margin in units of what we do not know | — | computed |
| $a_n$ | $n\,\tau/s$ — epistemic scale relative to per-restart search scale | — | computed |

$\tau$ and $s$ are different objects and conflating them is the main modelling
trap in this problem. $s$ is aleatoric, under our control, and bought down with
restarts. $\tau$ is epistemic, about a deterministic $m(g)$, and bought down only
with better models or more labels.

### 1.1 The general score

Integrating the exponential-excess realisation model over a Gaussian posterior on
$m$ gives, for $n$ restarts,

\[
S(g, n) \;=\; \nu(g)\,\Big[\Phi(z) - e^{\,a_n^2/2 - a_n z}\,\Phi(z - a_n)\Big],
\qquad z=\frac{t-\hat m}{\tau},\quad a_n=\frac{n\tau}{s}.
\]

Both limits are the ones we want:

* $\tau \to 0$ (perfect critic): $S \to \nu\,(1 - e^{-n\delta/s})$ — pure search.
* $s \to 0$ (perfect relaxer): $S \to \nu\,\Phi(z)$ — pure knowledge.
* $n \to \infty$: $S \to \nu\,\Phi(z)$ — infinite restarts resolve the search but
  not our ignorance.

$S$ is concave in $n$, so the budget allocation is exact water-filling:

\[
n_k = \frac{1}{b_k}\log\frac{a_k}{\lambda}\ \ \text{where positive, else }0,
\qquad b_k=\frac{\delta_k}{s_k},\quad a_k=\frac{\nu_k \delta_k}{s_k},
\]

with $\lambda$ the unique root of $\sum_k n_k(\lambda)=N$. The *sequential*
greedy — always relax whichever gene has the highest current marginal value — is
exactly optimal and needs no $\lambda$.

**Because genes are free, the pool is effectively unbounded, and the optimum
degenerates to: score every gene, take the top $N$ by $S(g,1)$, one relaxation
each.** The second unit on a gene is worth $e^{-b_k}<1$ times its first, so a
fresh gene always wins. Multi-restart allocation only matters when the pool is
small relative to the budget.

---

## 2. Reality check: the base rate caps everything

**Current best WyFormer models reach MSUN $\approx 0.30$.** The metric already
counts uniqueness, so duplicate draws are priced in.

Reweighting a proposal by any score $r$ gives an expected yield gain of exactly

\[
\frac{\mathbb E[r^2]}{\mathbb E[r]^2} = 1 + \mathrm{CV}^2(r),
\qquad\text{and since } r\le 1,\quad \mathbb E[r^2]\le\mathbb E[r]
\;\Rightarrow\; \text{gain} \le \frac{1}{\mathbb E[r]}.
\]

At a base rate of 0.30 that is a hard ceiling of **3.3×**, attained only by a
perfect oracle that turns 30% into 100%.

### 2.1 The measured ceiling, and why it shrinks as the generator improves

The recorded **3.6× perfect-energy-oracle enrichment** was measured on
`upi73i4k`, whose base rate was MetaSUN $290/2500 = 0.116$ — arithmetic ceiling
$1/0.116 = 8.6\times$, so 3.6× sat well inside it. That number does **not**
transfer to a generator at 0.30.

The mechanism behind it is what transfers, and it is an *interior* optimum:

| kept of 2500 | MetaSUN | novel fraction |
| :--- | :--- | :--- |
| 500 | 0.422 (3.6×) | 0.422 |
| 250 | 0.332 (2.9×) | 0.332 |
| 100 | 0.220 (1.9×) | 0.220 |

Selecting on stability converts abundant novelty (0.722 per gene) into scarce
metastability (0.235 per gene). By 20% acceptance, metastable-and-valid has
saturated at 0.950 — there is no stability left to buy — so tightening further
*only* costs novelty and MetaSUN falls. The anticorrelation is measured, not
assumed: $P(\text{novel}\mid\text{metastable}) = 0.493$ against
$P(\text{novel}\mid\text{not metastable}) = 0.793$, $\phi = -0.284$.

> **Measured, and worse than this.** On `ehull5x-20260904-213346` (MetaSUN
> 0.316) the perfect-energy-oracle peak is **1.75× at $N = 500$**, and at 10%
> acceptance the oracle *loses money* (0.82×). A real cross-validated selector
> over 12 gene-level features reaches **1.27× at $N = 250$** ($p < 0.017$), and
> all of it comes from free structural descriptors — the trained critic and the
> generator likelihood each add nothing measurable. See
> [`gene_selection_measurements.md`](gene_selection_measurements.md) §1–2.
> *Correction, 2026-09-15:* the artifact these numbers come from scores most of
> its structures against the wrong genes (see the correction at the top of that
> note), so they are unsupported until re-measured.

**So the headroom for a gene-level stability critic shrinks as the generator
gets better at stability.** At 0.116 the critic had abundant novelty to spend and
scarce stability to buy. At 0.30 the trade is far less favourable: metastability
is already common, novelty binds earlier, and the saturation point arrives at a
looser acceptance rate. Do not carry 3.6× forward as the expected prize — re-run
the enrichment curve on the current model before sizing any claim.

### 2.2 Consequences

1. **Estimate the top-$N$ lift of $\hat r$ on a pilot before building anything**,
   and re-measure the oracle enrichment curve on the current generator. One
   afternoon, and together they price the entire enterprise. Report the lift,
   **not** $1+\mathrm{CV}^2(\hat r)$, unless $\hat r$ has been calibrated first:
   an over-dispersed $\hat r$ inflates the identity, measured at 1.438 against a
   delivered 1.27×.
2. **Tune acceptance near 50%, not 20% and certainly not 5%.** The optimum is
   interior and it moves right as the generator improves; measured at
   $N = 500$ of 1000 on the current model, where a *perfect* oracle at
   $N = 100$ returns 0.82×.
3. **Weigh machinery against a $\le 3.3\times$ prize** that is probably much
   smaller in practice. Free stages (dedupe, surprisal threshold, legality) earn
   their keep unconditionally; a learned critic has to beat them by enough to
   justify the labelling budget.

At 0.30, acceptance-rate arithmetic is benign: rejection sampling $\propto r$
accepts at $\mathbb E[r]/\max r \approx 1/3$, so ~3× more gene draws. Gene draws
are free.

---

## 3. The axes

**Axis 1 — realisation physics.**

1. **Deterministic.** A gene reconstructs deterministically and $g \mapsto E$ is a
   deterministic physical map. Models of it still carry epistemic uncertainty.
   Formally $s \equiv 0$.
2. **Stochastic, archive procedure unknown.** Genes with free DoF do not
   reconstruct deterministically. The archive holds realisations chosen by a
   procedure we do not know. We have PyXtal + ORB / DiffCSP++, which relax
   stochastically. $s > 0$, unknown.
3. **Stochastic, archive procedure partly known**, per
   [`lemat_bulk_source_databases.md`](lemat_bulk_source_databases.md).

**Axis 2 — labelling budget.** (a) archive only; (b) sample & relax 10k genes;
(c) sample & relax 100k genes. At MSUN 0.30 these give ~3,000 and ~30,000
positives — both ample for fitting. **The real difference between (b) and (c) is
not statistical power, it is how many design strata you can afford.**

**Axis 3 — novelty lookup.** (i) permitted; (ii) forbidden.

**Axis 4 — generator conditioning**, treated separately in §7 because it is a
modifier on the whole framework rather than a coordinate in the matrix: it
changes the *proposal*, and therefore changes what $p_\theta$ means in every
formula above. §§0–6 assume the unconditional baseline.

---

## 4. Row 1 — deterministic realisation

$s\equiv 0$, so there is no search term and no allocation problem: one relaxation
per gene, never a restart.

\[
\boxed{\,S(g) = \nu(g)\cdot\Phi\!\left(\frac{t-\hat m}{\tau}\right)}
\]

**Policy.** Rank, take the top $N$, one relaxation each.

**But the true optimum is sequential.** With a deterministic map, every
relaxation is a permanent exact label and no gene is ever worth repeating. The
budget-optimal policy is therefore a finite-horizon Bayesian experimental design
(knowledge-gradient / expected-improvement over the critic's posterior), of which
top-$N$ is the myopic one-shot approximation. Any budget large enough to split
into rounds should be split.

| budget | estimated from data | must be assumed | recommended design |
| :--- | :--- | :--- | :--- |
| **(a)** | $\hat m$, $\tau$ from the archive critic; $\nu$ calibrated on the train/test split boundary | that archive-fitted $\tau$ transfers to generated genes — it does not, and the direction is optimistic | conformalise $\tau$ on an archive held-out split for a distribution-free interval; expect the *order* to survive and the *level* not to |
| **(b)** | on-distribution recalibration of $\tau$; the link; the gene→structure novelty gap | that one round suffices | 2 rounds of 5k beats 1 round of 10k; keep 1k held out for the transfer test |
| **(c)** | everything in (b), plus a genuine active-learning loop | — | 4–5 rounds, knowledge-gradient acquisition, surprisal-stratified with recorded weights; no replicates needed (deterministic), so 100k **distinct** genes |

**Where this row is wrong in practice.** It is contradicted by the measured
rattle experiment (392/1000 genes moved off their predicted orbits) and by the
duplicate-draw structure. Treat row 1 as the *limiting case that bounds how much
of the problem is knowledge rather than search*, not as a description of the
pipeline.

---

## 5. Row 2 — stochastic realisation, archive procedure unknown

Full score and full water-filling from §1.1. Two things are materially harder
than in row 1.

**$s$ is barely identified.** The censored likelihood fits it through a shared
parameterisation; for genes seen once — the majority — that is extrapolation from
DoF-like features, not measurement. *Measured:* "the majority" is **93.1%**.
Of 4,826,004 augmentation-invariant genes in `lemat_bulk_fmax1_stress`, 195 have
20 or more rows. There is no archive-only route to a per-gene $s$. Its own docstring notes the once-observed
case is formally degenerate. The saving grace is that the link **saturates**:
$1-e^{-\delta/s}$ is flat beyond $\delta/s \gtrsim 3$, so the estimate only has
to sort genes into three buckets — infeasible ($\delta<0$), search-limited
($\delta/s \sim O(1)$), easy ($\delta/s \gg 1$). Never rely on $\hat s$ being
calibrated, only monotone: fit the EMG expression as a *link function* with a
free slope on $z$ and a free scale on $a$, so miscalibration is absorbed into two
fitted parameters. Floor $s$ physically (shrink toward a DoF regression), not at
the numerical guard `min_scale = 1e-3` — no real gene has a 1 meV/atom
realisation spread.

**The novelty/stability dependence is unidentifiable.** The archive's inclusion
probability reads stability (people compute plausible things), so
$P(g\in D \mid g, m(g)) \ne P(g \in D\mid g)$, and the product
$\nu\cdot\Phi(z)$ **overestimates** the joint — worst at the top of the ranking,
where you act. $p_\theta$ absorbs much of this (it was fitted on the selected
archive), but the residual is precisely what $\hat m$ knows and $p_\theta$ does
not. Without knowing the selection procedure you cannot correct it analytically;
you can only fit the joint directly, which needs budget (b) or (c).

| budget | estimated from data | must be assumed | recommended design |
| :--- | :--- | :--- | :--- |
| **(a)** | nothing about $s$ for *your* sampler. Archive genes with several realisations give a within-gene spread, but those realisations were chosen by the unknown procedure, so it is a biased proxy | $s$ from DoF; the uncorrected occupancy model; independence of the two factors | do not pretend to use $s$ quantitatively — use it ordinally (DoF rank) or fall back to row 1's score and accept optimising the wrong thing |
| **(b)** | one of {a good joint scorer, an $s(\text{DoF})$ law} — not both well | the other | split **7k × 1** (≈2,100 positives, enough for the scorer) + **150 genes × 20 restarts** (enough for a 2–3 parameter $s(\text{DoF})$ law, not per-gene $s$) |
| **(c)** | $s(g)$ directly, $\tau$, the ratio $\tau/s$, the empirical hit-curve shape, occupancy calibration, the gene→structure novelty gap, and the joint scorer | — | **5k genes × 20 restarts** (replicate stratum, stratified over surprisal and free-DoF) + ~50k surprisal-stratified 1-shot with recorded weights + held-out slice |

**The measurement that decides where effort goes** is $\tau/s$, and only budget
(c) gives it cleanly:

* $\tau \gg s$ — *knowledge-limited*. The relaxer finds the bottom; we do not know
  which genes have a low one. Invest in the critic; extra restarts are wasted.
* $s \gg \tau$ — *search-limited*. We know which genes are promising and cannot
  realise them. Invest in gauge fixing, lattice priors, basin hopping; a better
  critic buys nothing.

---

## 6. Row 3 — archive procedure partly known

Everything in row 2, plus four corrections that cost **documentation work, not
labelling budget**. This is the best effort-to-payoff cell in the matrix, and it
improves even the archive-only column.

### 6.1 The selection tilt becomes partly identifiable

The archive's inclusion rule is written down, per source:

* **Alexandria R1** submitted to DFT only when CGAT predicted $E_\text{hull} < 50$ meV/atom;
  **Alexandria 2.0** only when Orb-v2/ALIGNN predicted $< 100$ meV/atom.
* **MP** required substitution probability $\ge 10^{-3}$ and strict formal charge neutrality.
* **OQMD** was exhaustive over 88 binary prototypes × 84 elements — *unconditional*
  in binary space, and therefore genuine negative evidence there.

So $P(\text{archive} \mid e)$ is a mixture of explicit, documented, largely
threshold-shaped rules. Where it must still be estimated, Bayes splits it into
one piece each side has:

\[
P(\text{archive}\mid e) = \frac{P(e\mid\text{archive})\;P(\text{archive})}{P(e)} .
\]

$P(e\mid\text{archive})$ comes from the archive; $P(e)$ comes from the
generated-and-relaxed set (budgets b/c); the prevalence $P(\text{archive})$ is
not identifiable but enters as an additive constant in the log-odds and
**cancels for ranking**. It cannot be estimated from the archive alone — that is
the positive-unlabelled trap.

The tilt then enters the novelty term as a coupling, not a feature:

\[
\nu(g) \;\approx\; \mathbb E_e\Big[\exp\big(-M\,p_\theta(g)\,w(e)\big)\Big],
\qquad w(e)=\frac{P(\text{archive}\mid e)}{P(\text{archive})},
\]

the expectation taken over the critic's posterior on $e$, since $e$ is unknown at
decision time. This is the principled resolution of the factorisation problem:
neither assume independence nor abandon the factorisation — write down the one
dependence that exists and integrate it out. It is also far cheaper in degrees of
freedom than fitting $P(\text{metastable}\mid\text{novel})$ on the novel subset
alone.

### 6.2 A novelty prior that needs no lookup

The documented blind spots are computable gene-level features that predict
archive **absence from mechanism** rather than from likelihood. Under axis (ii)
this is the single largest available improvement:

* prototype distance $d_\text{proto}$ against the 88 OQMD + ~2,500 Alexandria scaffolds
* count of continuous Wyckoff DoF — prototype substitution never sampled general $(x,y,z)$
* monoclinic / triclinic indicator ($P2_1/c$, $C2/c$, $P1$, $P\bar 1$ appear almost only as ICSD imports)
* stoichiometric grid distance to the enumerated ratio set
* cell-size band: 20–60 atoms is capped out of Alexandria ($\le 20$–30) and thin in OQMD ($\le 50$)
* chemical arity $\ge 4$; non-oxide intermetallic indicator; electronegative-element fraction

*Measured:* features of exactly this kind — space group, positional DoF, arity,
orbit count, formula-unit size — carry **all** of the 1.27× that a fitted
selector achieves on the current pool. Deleting them drops it to 0.97×;
deleting the trained critic and the likelihood instead costs 0.02× and 0.03×.
The mechanism features are not a supplement to the critic; on this evidence they
are the whole of it.

### 6.3 The threshold $t$ carries known uncertainty

66.6% of reduced formulas hold exactly one calculated structure, so their hull
bound is one attempt from one pipeline. Formula attempt multiplicity $n(X)$ and
the ICSD-backed indicator say how tight the bound is — where an ICSD entry
exists, it is the minimum 55.4% of the time with mean excess 17.3 meV/atom;
elsewhere it is loose.

**This forks the objective, and the fork must be declared.** Beating a *loose*
archive hull is easy and counts for the benchmark; beating the *true* floor is
the discovery claim. Benchmark-optimal scoring uses $t_\text{archive}$ and should
*prefer* sparse hulls; discovery-optimal scoring uses $P(e \le t_\text{true})$
and should discount them. They are different rankings. Forced exploration into
rare chemical systems maximises the divergence, so report enrichment **stratified
by hull density**.

### 6.4 Part of the 60 meV/atom label floor is correctable

Known, documented, systematic label errors:

* ferromagnetic initialisation across all three sources elevates AFM ground
  states by 50–200 meV/atom;
* Hubbard $U$ differs (MP: oxides **and** fluorides; OQMD: oxides only;
  Alexandria: none in PBEsol/SCAN);
* functionals differ (PBE / PBEsol / SCAN);
* elemental reference potentials for Br, I, Ag shift 20–36 meV/atom between
  shallow and deep hulls.

Conditioning the critic on source and functional, and adding a magnetic-ion /
AFM-risk feature, attacks the label floor at its source rather than working
around it. Since $\tau$ bounds the whole enterprise in the knowledge-limited
regime, this is high-value and needs no new labels.

| budget | what row 3 adds over row 2 |
| :--- | :--- |
| **(a)** | all four corrections are available with zero labelling. Materially better one-shot scorer than (2a); the only cell where archive-only is genuinely respectable |
| **(b)** | the tilt and novelty-prior features are installed rather than fitted, so fewer parameters come from the 10k — shift budget from the scorer stratum to the replicate stratum (e.g. 6k × 1 + 200 × 20) |
| **(c)** | same design as (2c); the freed degrees of freedom go into measuring $s(g)$ per stratum and into the hull-density stratification of §6.3 |

---

## 7. Conditional generators

Conditioning and reranking are two routes to the same target,
$p_\theta(g)\,r(g)/Z$. Reranking keeps the generator unconditional and reweights
afterwards: exact, retargetable without retraining, but capped at $1/\mathbb
E[r]$ and confined to the generator's support. Conditioning moves the proposal
itself: amortised, no acceptance cost, and — the one thing reranking cannot do —
able to change the support, and with it the base rate.

**They are substitutes, and they eat each other's headroom.** §2.1's mechanism
applies directly: a better conditional generator raises the base rate, which
lowers the ceiling for any subsequent reranker. Budget the two together, not
independently.

### 7.1 Two kinds of conditioning, with very different risk

**Constraint conditioning** — the condition is a *design variable* we choose and
know exactly: composition ([`csp_mode.md`](csp_mode.md)), allowed element set
([`chemical_system_mode.md`](chemical_system_mode.md)), space group (this
architecture's start token). Acceptance rate is one by construction in both
modes; the proposal density $\pi(c)\,p_\theta(g\mid c)$ is exactly known because
we designed $\pi$; and the condition is a deterministic function of the gene.

**Label conditioning** — the condition is a measured, noisy property: $E_\text{hull}$,
formation energy, `max_force`. It bakes the label's bias irreversibly *into the
generator*, where no post-hoc correction can reach it; the AdaLN
`condition_scale` is an extrapolation knob with measured instability (`wjwmgjag`
diverged on a bad value, per [`dirty_data_conditioning.md`](dirty_data_conditioning.md));
and nothing can be retargeted without retraining. The project has already
retired one instance of it — conditioning on `max_force` was dropped once its
cost was measured ([`unconverged_relaxation_energy.md`](unconverged_relaxation_energy.md)).

> **Rule.** Prefer constraint conditioning. Use label conditioning only with a
> label you would defend as a training target — which, for
> $\min(E\mid\text{gene})$, means the censored likelihood, not a per-row
> $E_\text{hull}$ that rebuilds the popularity-prior confound inside the
> generator.

### 7.2 What conditioning does to the formulas

**The novelty term needs the *unconditional* density.** $\nu = e^{-M p_\theta(g)}$
is an occupancy statement about the archive's marginal gene distribution.
$p_\theta(g\mid c)$ is that distribution tilted to $c$, and under a conditional
sampler every gene is scored under its own conditioning, so the numbers are not
comparable across genes. `score_gene_likelihood` refuses this case outright
rather than returning a plausible wrong number — correctly.

For **constraint** conditioning the repair is exact, because $c$ is a
deterministic function of $g$:

\[
\log p_\theta(g) \;=\; \log p_\theta\big(g \mid c(g)\big) \;+\; \log p\big(c(g)\big),
\]

with $p(c)$ the archive's marginal over conditions. This is precisely the
treatment `gene_likelihood.py` already gives the space group, which is drawn from
the empirical start-token prior and reported as its own `log_p_spacegroup`
column. Extending the same construction to the composition and chemical-system
heads makes conditional likelihoods comparable again and lifts the guard for
those modes.

For **label** conditioning there is no such repair: $E_\text{hull}$ is a property
of a realisation, not of the gene, so $c$ is not a function of $g$ and the
marginal cannot be recovered. A label-conditioned sampler therefore requires a
**separate unconditional model kept solely as a novelty meter**.

**Everything else in §1.1 survives verbatim** with $p_\theta(g)$ replaced by
$\pi(c)\,p_\theta(g\mid c)$, provided $\pi$ is the distribution actually sampled
from. The same caution as for temperature applies: the proposal is what you
sampled, not what you meant to sample.

**Conditioning has the same interior optimum as selection.** Condition hard on
low $E_\text{hull}$ and the generator returns archive-like stable structures and
novelty collapses — the §2.1 curve with a different knob. Alexandria 2.0 is the
extreme case: a hull-distance-conditioned transformer reaching a 99% hit rate
within 100 meV/atom, at which point stability is solved and novelty is the entire
remaining question. **The condition value is a knob to tune, not to maximise.**

### 7.3 What conditioning enables that reranking cannot

**(a) Chemical-system exploration.** Design $\pi(S)$ over systems instead of
inheriting the training frequency. This also promotes the budget problem to two
levels: a bandit over chemical systems (reward = unique-MSUN per relaxation,
prior from hull density), then §1.1's water-filling within a system.

The blocker is documented in
[`chemical_system_sampler.md`](chemical_system_sampler.md): the space group is
the start token, fixed before any element is drawn, so **no element mask can move
it**. `system_prior` reads $p(\text{system}, \text{sg})$ off the training split,
so a system absent from training has no space-group prior at all — and the cost
of getting it wrong is measured: row-weighted mean KL 0.92 nats per system
against the unconditional distribution — reproduced independently at **0.861
nats** on 4,000 held-out systems, of which a naive element-marginal backoff
recovers only **11%** — with Li-Mn-Co-O triclinic and monoclinic
while the unconditional sampler starts a third of its batch in 123 and 216.
Forced exploration into unseen systems, as the machinery stands, spends most of
its budget on wrong start tokens.

> **The enabling piece is a backoff model for $p(\text{sg}\mid S)$ with support
> outside the training empirical distribution** — element-similarity smoothing
> over the 661,816 observed systems, or a small head from the element indicator
> to a space-group distribution. Independently checkable on held-out systems
> (KL against their true space-group distribution), and nothing else in the
> exploration plan depends on anything but it.

**(b) Condition on the archive's documented blind spots.** This is the row-3
payoff, and it is the strongest novelty lever that requires no novelty label.
Every feature in §6.2 is a *design variable*, not just a covariate: monoclinic
and triclinic space groups (essentially never generated by any of the three
theoretical pipelines — they appear almost only as ICSD imports), high continuous
Wyckoff DoF (prototype substitution never sampled general $(x,y,z)$), cell sizes
of 20–60 atoms (capped out of Alexandria at $\le 20$–30, thin in OQMD at
$\le 50$), chemical arity $\ge 4$, non-oxide intermetallics. Conditioning on
these targets unvisited space directly, and because the condition is exactly
known it is constraint conditioning with acceptance rate one.

Caveat, and it is §6.3's fork in its sharpest form: these are regions the archive
never *looked*, so the hull there is artificially elevated. That is easy MSUN and
uncertain discovery. Report enrichment stratified by hull density or the number
means nothing.

**(c) CSP mode** is a different objective — novelty is irrelevant when the
composition is given — but it is the cleanest cell in this whole note: $\nu$ drops
out, $\delta$ is exact, and §1.1's water-filling over restarts is the entire
problem.

### 7.4 The identifiability bonus

Conditioning on the chemical system makes the scoring regression identifiable,
which is a bigger practical win than the extra support. Unconditionally,
$\log p_\theta(g)$ and $\hat m(g)$ are collinear — both partly encode "is this a
popular composition", and `gene_energy.py`'s MSE-on-observed-minimum target makes
it worse by learning a popularity prior in energy costume. With the system fixed
by design, $\log p_\theta(g\mid S)$ becomes within-system structural plausibility
while $E_\text{hull}$ is already a within-composition quantity, so the two
features are near-orthogonal by construction.

### 7.5 Summary of modes

| mode | condition | proposal known | novelty term | use when |
| :--- | :--- | :--- | :--- | :--- |
| unconditional | — | yes | $e^{-M p_\theta(g)}$ | baseline; maximum reranker headroom |
| space group pinned | start token | yes | add $\log p(\text{sg})$ — already done | targeting blind-spot symmetries |
| chemical system | element set | yes, $\pi(S)p_\theta(g\mid S)$ | add $\log p(S)$ (§7.2) | exploration; identifiability |
| CSP | exact composition | yes | not applicable | structure prediction, not discovery |
| $E_\text{hull}$ (AdaLN / guidance) | noisy label | only through the sampler | **unrecoverable** — keep a separate unconditional meter | last resort, censored label only |

Constraint conditioning costs nothing at sampling time, so it belongs at the
front of the cascade, ahead of every scored stage.

---

## 8. The novelty axis

### (i) Lookup permitted

$\nu$ becomes an exact indicator, so it is a **filter**, and among survivors the
ranking is by the metastability factor alone. Three things stop it being simply
$\Phi(z)$:

1. **The lookup is also evidence.** Learning "absent" updates belief about $m$,
   because the archive preferentially absorbed the stable genes. The update is
   gene-dependent: for a low-$p_\theta$ gene absence is expected and carries
   nothing; for a high-$p_\theta$ gene absence is *surprising* and carries a lot —
   and those are exactly the survivors that would otherwise rank at the top. Use
   $P(m \le t \mid g, \text{absent})$, i.e. recalibrate the link on the novel
   subset rather than reusing archive-fitted $\tau$.
2. **$s$ does not vanish** unless row 1 holds. Two genes with equal $z$ and
   different $s$ are not equally good, and the water-filling still applies with
   $\nu \equiv 1$ among survivors.
3. **The lookup is gene-level; the reward is structure-level.** It is an exact
   indicator of a *proxy*.

### (ii) Lookup forbidden

$\nu = \exp(-M\,p_\theta(g)\,w(e))$ per §6.1 — the occupancy model with the
selection tilt. Under row 3, §6.2's mechanism features substitute for much of
what the lookup would have told you.

The occupancy form follows from treating the archive as $M$ draws: a gene is
novel with probability $(1-p_\theta)^M \approx e^{-M p_\theta}$, a sigmoid in
surprisal with half-point at $-\log p_\theta \approx \log M \approx 15.4$ nats
for the `lemat_bulk_fmax1` train split. Caveats: the archive is not an i.i.d.
sample; $p_\theta$ was fitted *on* $D$ so it inflates density near the data and
**under-calls** novelty (conservative, the right direction for a benchmark); and
gene novelty is not structure novelty.

**Calibrating it without deploying the lookup.** Hold out a slice of LeMat,
predict membership in the retained part from $e^{-M p_\theta}$, measure the
calibration curve. This is legitimate — using the archive to *validate* is the
same status as training on it. At budgets (b)/(c) you can go further and fit
$P(\text{novel}\mid\text{surprisal})$ empirically; a fitted monotone scalar
function is a model, not a per-gene lookup, and stays on the right side of the
line.

### Where the line sits

A probability about a deterministic fact is defined relative to an information
set, so the benchmark is well posed only if the information set is declared:

* **weights only** — $p_\theta$, geometry, composition, critic outputs. Reportable
  as the generator's own SUN rate, comparable with prior work.
* **weights + aggregate archive statistics** — hull, chemsys coverage, documented
  procedure features (§6.2). Aggregate, not per-gene; same status as training on
  the archive. Declare it.
* **weights + per-gene archive lookup** — production only.

**Report two numbers**: the raw generator's MSUN (the scientific claim about the
model) and the cascade's unique-MSUN per relaxation (the engineering claim about
the discovery system), with every reference-touching filter confined to the
second.

---

## 9. Summary matrix

Score and policy are set by axis 1; the budget decides what is measured versus
assumed; the lookup axis modifies the novelty factor only.

| | **(a) archive only** | **(b) 10k** | **(c) 100k** |
| :--- | :--- | :--- | :--- |
| **(1) deterministic** | $\nu\Phi(z)$, top-$N$. Conformalise $\tau$ on an archive split. Order transfers, level does not | recalibrate $\tau$ on-distribution; 2 × 5k rounds | 4–5 active-learning rounds, 100k distinct genes, knowledge-gradient |
| **(2) stochastic, procedure unknown** | weakest cell. $s$ not identifiable for your sampler; use it ordinally or fall back to row 1 | 7k × 1 + 150 × 20; get a scorer **or** an $s(\text{DoF})$ law, not both | 5k × 20 replicates + 50k stratified 1-shot; measures $s$, $\tau$, $\tau/s$, hit-curve shape |
| **(3) procedure partly known** | best effort/payoff: tilt + mechanism novelty features + hull-looseness + label-bias corrections, all free | as (2b) with more budget freed for replicates (6k × 1 + 200 × 20) | as (2c), plus hull-density stratification and per-source tilt |

| | **(i) lookup** | **(ii) no lookup** |
| :--- | :--- | :--- |
| novelty factor | exact filter, **plus** a gene-dependent update to $\hat m$ | $\exp(-M p_\theta w(e))$, tilt from §6.1, mechanism features from §6.2 |
| what it cannot fix | gene novelty ≠ structure novelty | same, plus occupancy-model error |

Conditioning (§7) modifies the proposal in every cell rather than selecting one.
Constraint conditioning leaves all of the above intact once $\log p(c)$ is added
back to the likelihood; label conditioning forfeits the axis-(ii) novelty term
and requires a separate unconditional meter. Because conditioning and reranking
are substitutes, a cell's reranking headroom should be re-measured after any
change to the generator, not carried over.

---

## 10. Invariants — true in every cell

1. **Take the top $N$ by $S(g,1)$, one relaxation each.** Free genes make the
   multi-restart allocation moot unless the pool is small. *Measured:* a restart
   returns 62% of what a fresh gene returns per relaxation (best-of-1 yields
   0.199 MetaSUN per relaxation, best-of-3 yields 0.123).
2. **Threshold, do not sample proportionally**, when the goal is hits per
   relaxation. Acceptance $\propto r$ targets $p(g\mid\text{success})$ and
   preserves diversity; thresholding maximises yield. Pick deliberately.
3. **Cascade, do not gate.** Free gene score → NEP89 pre-relax (recall 1.83×
   chance, median regret 0) → ORB/MACE. The gene score's virtue is that it is
   free, so it goes first; each threshold follows from the cost ratio. Fit one
   model per stage on that stage's available features — never mix a later stage's
   features into an earlier stage's model.
4. **Feed the $z$-score, not the probability.** A probit link makes "already
   calibrated" the interpretable null of slope 1, intercept 0. OLS on a
   probability can go negative and breaks rejection sampling outright.
5. **Label with one protocol.** NEP89's 0.5 eV/atom oracle scatter disqualifies
   it as a labeller (it is a selector); the signal sits at the 60 meV/atom floor.
6. **Never feed the label side as a feature**: no post-relaxation quantities, no
   per-gene archive lookups, no `GENE_MIN_FORMATION_ENERGY_COLUMN`, no
   nearest-archive-gene distance (a soft lookup that leaks novelty almost
   perfectly), no provenance columns that do not exist for generated genes, no
   sampling stratum (that is a weight, never a predictor), and not the *measured*
   $s$ from the replicate stratum (it costs 20 relaxations to obtain).
7. **Instrument $t$ rather than letting it leak.** $t$ is archive-derived and
   silently carries "how well-studied is this composition". Add explicit chemsys
   coverage features so the model can attribute the effect, and report with and
   without them.
8. **`add_observed_gene_minimum` takes the minimum over the full dataset by
   design.** Any critic used for a split-based experiment must have its target
   built from train rows only, or $\hat m$ on test rows is partly memorised.
   *Measured on `gene_min_energy_adamw_wsd-20260907-151306`: it does not bite* —
   11.6% of test rows share a gene with train, and their MAE (0.056) is no
   better than the rest (0.051). Keep the invariant; the exposure is real even
   where the effect is not.
9. **Never train the critic on the labels the protocol scores with.** Budgets
   (b)/(c) produce replicated, on-distribution energy labels — better data for
   fitting $m$ and $s$ than the archive, and the direct way to attack the
   60 meV/atom floor at its source rather than working around it. But if the
   protocol scores with ORB and the critic is fitted on ORB labels, selection
   optimises the scorer rather than the physics. Label the critic's training set
   with a *different* MLIP from the one the protocol scores with, or hold out a
   disjoint slice for scoring.
10. **Search-side fixes are budget-free multipliers** and apply in rows 2–3
   regardless of the scoring: quotient out the polar-origin gauge; replace i.i.d.
   PyXtal `from_random` draws with a low-discrepancy sequence over the free
   coordinates (the oracle test says ~0.3 Å precision suffices, so this is a
   coverage-of-a-box problem); fix the lattice prior (28% cell inflation wastes
   the coordinate draw); relax on the gene's manifold with `FixSymmetry` /
   `GeneralizedCoordinates` rather than letting the search wander off-symmetry.
11. **A critic's worth is its error relative to the spread of the pool it
   ranks, and conditioning narrows that spread on purpose.** Measured on one
   checkpoint: Pearson $+0.979$ against archive $e_\text{hull}$ (spread 0.583,
   residual 0.120) and $+0.061$ against the same quantity on an
   $E_\text{hull}$-conditioned pool (spread 0.137, residual 0.198). The critic
   barely changed; the target collapsed below its noise floor. So a critic
   validated on archive-like data says nothing about its value on a conditioned
   pool, conditioning and reranking cannot be budgeted independently (§7), and
   the residual must be quoted against the *deployment* spread, never the
   archive's.
12. **A conditional sampler still needs an unconditional novelty meter.** Under a
   constraint condition it is the same model plus $\log p(c)$; under a label
   condition it is a second model. Never read novelty off the conditional
   likelihood — every gene would then be scored under its own conditioning, which
   is why `score_gene_likelihood` refuses rather than returning a number.

---

## 11. What to measure first — and what came back

Run 2026-09-12; full numbers and method in
[`gene_selection_measurements.md`](gene_selection_measurements.md).

| # | question | answer |
| ---: | :--- | :--- |
| 1 | yield gain of $\hat r$ on an existing pilot | **1.27×** at $N=250$, $p<0.017$; linear model on critic+likelihood alone gives 1.00× |
| 2 | perfect-oracle enrichment on the current generator | **1.75×** at $N=500$; **0.82×** at $N=100$; ceiling 3.16× |
| 3 | is $\hat s$ estimable from genes with ≥20 archive rows | **No.** 93.1% of the 4.83M archive genes are singletons; 195 have ≥20 rows |
| 4 | $\tau/s$ — knowledge- or search-limited | $\tau \approx 0.20$, $s \approx 0.11$ eV/atom → **knowledge-limited** |
| 5 | gene→structure novelty gap among the metastable | 0.626 vs 0.491 — **a gap of 0.135**; the binding anticorrelation is structure-level ($\phi=-0.321$), invisible to any gene feature |
| 6 | occupancy calibration | **Works above the half-point**, badly over-confident below; one fitted constant repairs it (effective $\log M = 13.90$, not 15.20) |
| 7 | held-out-system KL for a $p(\text{sg}\mid S)$ backoff | **11% of 0.861 nats recovered.** The cheap backoff does not unblock chemical-system exploration |

Three follow-ups the measurement pass raised and could not settle:

1. **The same pool from an unconditional generator.** Everything in §2 was
   measured on an $E_\text{hull}$-conditioned model, so "the critic adds
   nothing" and "conditioning already ate the critic's signal" are not
   separated. This is the one experiment that needs a generator run.
2. **A hull-referenced critic.** The existing critic predicts formation energy
   at MAE 0.051 eV/atom and *still* correlates with relaxed $e_\text{hull}$ at
   only $+0.061$, because subtracting the hull cancels what it is good at. Fit
   the target that is actually ranked.
3. **A learned $p(\text{sg}\mid S)$**, over the element set rather than
   element marginals. 11% is a floor, not a ceiling.

---

## 12. Known gaps in this note

* The reward is treated as separable across genes. When two distinct genes relax
  into the same structure it becomes monotone submodular under a cardinality
  constraint: exact maximisation is NP-hard and greedy carries the standard
  $1-1/e$ guarantee. Given the measured duplicate structure, this is the regime
  we are actually in.
* Plugging point estimates of $(\nu, m, s)$ into an argmax is the optimiser's
  curse: selection favours genes whose $\hat m$ is underestimated or whose $\tau$
  is overestimated, and marginal calibration on a validation split guarantees
  nothing on the accepted tail. The exactly-optimal policy under parameter
  uncertainty is adaptive — a Gittins index on each gene, tractable here because
  the per-gene state is low-dimensional — and §4's sequential design is the
  practical version.
* Covariate shift between a pilot drawn unconditionally and a deployment drawn
  under the reweighted sampler is not addressed by any single-round design. One
  refit round (the cross-entropy method's first step) is cheap and is the
  difference between interpolation and extrapolation.
* $p_\theta$ is the *proposal*, which is not the model density alone: the space
  group is drawn from the empirical training distribution rather than predicted,
  generation applies both a learned calibration temperature and a sampling
  temperature, and per-token tempering is not global tempering. Any reweighting
  must evaluate the density actually sampled from.
