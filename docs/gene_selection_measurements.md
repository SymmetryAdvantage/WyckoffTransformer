# What gene-level selection actually buys: measured

> **Scope:** the measurement pass for
> [`gene_selection_optimality_matrix.md`](gene_selection_optimality_matrix.md),
> run 2026-09-12 on artifacts that already existed. Nothing here required
> training a new generative model. Every number is reproducible from the scripts
> named in §8.
>
> **Correction, 2026-09-15 (audit at commit `fba3b24`): the protocol artifact
> this note measures is contaminated, and its conclusions are unsupported until
> they are re-measured.** `protocol_ehull5x-20260904-213346:v1` scores one gene
> cohort's structures under another's indices. Its `wyckoff_genes.json.gz` (md5
> of the decompressed JSON `86990c10…`) replaced the one the first pass drew from
> (`48ac2f0e…`, v0), and the stage logs were resumed on top of it. Of the 2804
> successful relaxations, 412 have their v1 gene's composition and 2392 their v0
> gene's. Of the 998 structures in `structures.csv`, **789 belong to a v0 gene**.
> §0.1's diagnosis is therefore inverted: the ~400 rows it filters out are the
> *correct* ones. Every analysis below that pairs a gene (its features, its
> likelihood, its critic score, its gene-level novelty) with a structural
> outcome pairs mismatched genes for most of the pool.
>
> The clean replacement is **`protocol_ehull5x-20260904-213346:v3`**: a
> from-scratch re-run on 2026-09-15 on zeus (new 1000-gene cohort, `--no-resume`,
> code at `fba3b24`, ORB-v3-conservative-inf, `--relax-timeout 1800`, 8 workers
> on one RTX 6000 Ada). All 2406 draws and 998 structures match their own genes,
> all 2406 relaxations succeeded, and no worker faulted. Its free-readout MetaSUN
> is **0.268** per sampled gene, not 0.316 (novel structures 0.594 vs 0.702,
> metastable 0.576 vs 0.521); fixed-symmetry MetaSUN is 0.185 vs 0.223. The W&B
> run summary now carries v3's numbers. Re-measure on v3 before using anything
> below. See `docs/de_novo_ranking_protocol.md` for the guard that now prevents
> this.
>
> **Headline (unsupported, see above):** on the current generator the whole gene-level reranking
> enterprise is worth **1.27× at 25% acceptance**, against an arithmetic ceiling
> of 3.16× and a perfect-energy-oracle ceiling of 1.75×. All of that 1.27×
> comes from **free structural descriptors of the gene**. The trained energy
> critic and the generator likelihood each contribute nothing measurable — not
> because the critic is bad (it ranks archive hull distance at Pearson +0.979)
> but because the $E_\text{hull}$-conditioned generator has already narrowed the
> pool to below the critic's noise floor.

---

## 0. What was measured, on what

| artifact | what it is | note |
| :--- | :--- | :--- |
| `ehull5x-20260904-213346` | generator, `energy_above_hull`-conditioned, trained on `lemat_bulk_ehull` | protocol MetaSUN **0.316** per sampled gene |
| `protocol_ehull5x-20260904-213346:v1` | 1000 genes → PyXtal → ORB-v3-conservative-inf relaxation, ORB hull | 2404 usable trials |
| `gene_min_energy_adamw_wsd-20260907-151306` | Scalar critic, MSE on `gene_min_formation_energy_per_atom`, trained on `lemat_bulk_fmax1` | scored at `max_force = 0` |
| `lemat_bulk_fmax1_stress` | the current archive | 5,327,342 rows |

Two caveats carried into every conclusion below. The generator and the critic
were trained on **different, superseded** LeMat variants (`lemat_bulk_ehull`,
`lemat_bulk_fmax1`); per `CLAUDE.md` these runs may be analysed but not
extended. And the critic predicts a **PBE** formation energy while the protocol
scores against an **ORB** hull, so any critic-versus-outcome correlation mixes
model error with a cross-functional offset.

### 0.1 A data defect found on the way in

> **Misdiagnosed; see the correction at the top.** The minority rows are not
> misfiled: they are the only trials of the genes in the artifact's own gene
> file. `structures.csv` *is* affected, and the formula filter keeps the wrong
> side.

`relaxations.csv` and `pyxtal.csv` inside the protocol artifact concatenate a
crashed first pass and its retry **without reconciling the `index` column**.
400 of 2804 `status == ok` relaxation rows carry an index belonging to a
different gene: index 130 holds two `Fe3Hf6Re9` trials and one `Cs18I2Se` trial.

Per-gene statistics computed naively from that file are wrong, and wrong in a
way that looks like physics — before filtering, zero-DoF genes appeared to have
a 2.23 eV/atom spread across restarts. **Filter trials to those whose `formula`
matches the gene's formula in `structures.csv`** before grouping; after that the
per-gene minimum reproduces `structures.csv` exactly (max abs difference 0.0).
`structures.csv` itself is unaffected — no gene's recorded best trial is one of
the 400.

---

## 1. The ceiling moved, exactly as predicted, and further

§2.1 of the matrix note predicted that reranking headroom *shrinks* as the
generator improves. Re-running the perfect-oracle enrichment curve on this
generator confirms it and overshoots the prediction.

Ranking all 1000 genes by their **measured** ORB `e_above_hull` and keeping the
best $N$:

| $N$ kept | MetaSUN per kept gene | lift | novel fraction | metastable fraction |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.260 | **0.82×** | 0.260 | 1.000 |
| 250 | 0.420 | 1.33× | 0.420 | 1.000 |
| 500 | 0.554 | **1.75×** | 0.554 | 1.000 |
| 700 | 0.451 | 1.43× | 0.629 | 0.793 |
| 1000 | 0.316 | 1.00× | 0.701 | 0.555 |

Base rate 0.316, arithmetic ceiling $1/0.316 = 3.16\times$.

Two things changed against the `upi73i4k` measurement at base rate 0.116:

- **The peak fell from 3.6× to 1.75×**, and the ceiling it sits under fell from
  8.6× to 3.16×. A perfect energy oracle now captures 55% of the arithmetic
  ceiling where it once captured 42% — the oracle got *relatively* better while
  the absolute prize collapsed.
- **The interior optimum moved from ~20% acceptance to ~50%.** At 10% acceptance
  a perfect energy oracle is now actively *harmful*: 0.82×, worse than not
  selecting at all. Metastability saturates at 100% by $N = 500$; everything
  tighter is paid for purely in novelty.

**Consequence 2 of §2.2 needs revising: tune acceptance near 50%, not 20%.**

---

## 2. What a real selector recovers: 1.27×, all of it from free features

The architecture under discussion — critic + generator likelihood → linear model
→ gene-level rejection sampling — was fitted and evaluated with 10-fold
cross-validation on the 1000 relaxed genes.

| features | model | AUC | $1+\mathrm{CV}^2$ | lift@250 | lift@500 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| critic only | logistic | 0.508 | 1.004 | 0.99× | 1.01× |
| surprisal only | logistic | 0.498 | 1.002 | 0.90× | 1.00× |
| critic + surprisal | logistic | 0.514 | 1.007 | 1.00× | 1.06× |
| + DoF, site count | logistic | 0.517 | 1.018 | 0.90× | 1.04× |
| + gene-novelty lookup | logistic | 0.535 | 1.024 | 1.15× | 1.05× |
| 12 features | **gradient boosting** | **0.574** | 1.438 | **1.27×** | **1.16×** |

**The proposed linear model buys nothing.** A nonlinear model on a wider feature
set buys a real but small amount. A 60-permutation null over the entire
CV-and-fit pipeline puts the gradient-boosted result at $p < 0.017$ for all
three statistics (null means 0.503 / 1.006× / 1.010×, null 95th percentiles
0.535 / 1.165× / 1.095×), so 1.27× is signal, not selection noise.

### 2.1 Which features

Leave-one-group-out and group-only ablation, same protocol:

| feature set | AUC | lift@250 | lift@500 |
| :--- | ---: | ---: | ---: |
| all 12 | 0.574 | 1.27× | 1.16× |
| **geometry only** (space group, positional DoF, arity, orbit count, formula-unit size, site count) | 0.558 | **1.20×** | 1.14× |
| critic only (predicted $E_f$, hull, their difference) | 0.519 | 1.05× | 1.01× |
| likelihood only (surprisal at two conditions, $\log p(\text{sg})$) | 0.479 | 0.92× | 0.97× |
| all minus geometry | 0.490 | 0.97× | 0.99× |
| all minus critic | 0.568 | 1.25× | 1.12× |
| all minus likelihood | 0.579 | 1.24× | 1.18× |

**Every measurable bit of gene-level signal is in free structural descriptors.**
Deleting the trained critic costs 0.02×; deleting the likelihood costs nothing;
deleting geometry destroys the model. This is the direct answer to "which
features should and should not go into $P(\text{novel} \wedge \text{metastable}
\mid \text{features})$": geometry yes, and on this evidence the other two are
not worth their inference cost.

### 2.2 What the pilot can and cannot resolve

A 1000-gene pilot at a 0.316 base rate has a detection floor. Over 10,000
random rankings:

| budget | random lift, 95th pct | 99th pct |
| ---: | ---: | ---: |
| $N=100$ | 1.23× | 1.33× |
| $N=250$ | 1.13× | 1.19× |
| $N=500$ | 1.08× | 1.11× |

Bootstrap CIs over genes for single-feature rankings:

| ranking | lift@250 | lift@500 |
| :--- | :--- | :--- |
| critic score | 0.96× [0.80, 1.11] | 0.99× [0.90, 1.09] |
| surprisal | 0.90× [0.74, 1.06] | 1.00× [0.90, 1.09] |
| gene-novelty lookup | 1.06× [0.87, 1.20] | 1.05× [0.95, 1.13] |
| oracle (measured $e_\text{hull}$) | 1.33× [1.12, 1.49] | 1.75× [1.61, 1.89] |

So "the critic buys nothing" is properly stated as: **at 95% confidence the
critic buys less than 1.09× at $N = 500$**, against an available 1.75×.

### 2.3 $1+\mathrm{CV}^2$ is not the yield gain unless $\hat r$ is calibrated

The gradient-boosted model reports $1+\mathrm{CV}^2 = 1.438$ and delivers 1.27×.

The identity is exact only when the score *is* the success probability.
Reweighting a proposal by an arbitrary score $\hat r$ gives a yield of

\[
\frac{\mathbb E[\hat r\, r]}{\mathbb E[\hat r]\,\mathbb E[r]},
\]

which equals $1 + \mathrm{CV}^2(\hat r)$ only when $\hat r = r$. An
over-dispersed $\hat r$ — which is what a boosted tree ensemble produces
without calibration — inflates the denominator's spread without buying any
extra covariance with the truth. §11 item 1 of the matrix note should read:
**calibrate $\hat r$ first, and report the top-$N$ lift as the operational
number.**

---

## 3. Why the critic fails here: range restriction, not a broken model

The critic is not broken. On its own held-out split it predicts
`gene_min_formation_energy_per_atom` to **MAE 0.051 eV/atom, Pearson 0.993**,
against a predict-the-mean baseline of MAE 0.605 and a target sd of 0.858.

**It is also not leaked.** Invariant 8 warns that
`add_observed_gene_minimum` takes the minimum over the full dataset, so a gene
present in both splits carries an identical target. Because that target is a
per-gene constant, exact float equality identifies the shared genes. Only
**11.6%** of test rows are affected, and their accuracy is no better than the
rest:

| test rows | n | MAE | Pearson |
| :--- | ---: | ---: | ---: |
| gene also in train | 2,318 | 0.0560 | +0.993 |
| gene test-only | 17,682 | 0.0505 | +0.993 |

The invariant is right in principle and does not bite on this checkpoint.

**Nor does the hull subtraction destroy the signal.** The obvious suspicion —
that formation energy is set by composition, that the hull at that composition
is set by the same thing, and that subtracting them cancels what the model is
good at — is testable and turns out to be false. Scoring the same 20,000
archive rows as a *hull distance* rather than a formation energy:

| | archive rows | generated pool |
| :--- | ---: | ---: |
| spread of the target being ranked | $e_\text{hull}$ sd **0.583** | ORB $e_\text{hull}$ sd **0.137** |
| critic residual sd | **0.120** | **0.198** |
| signal-to-noise | 4.9 | **0.69** |
| Pearson (critic score, measured $e_\text{hull}$) | **+0.979** | **+0.061** |
| Spearman | +0.953 | +0.078 |

**The critic ranks hull distance almost perfectly on the archive.** What kills
it on the generated pool is *range restriction*: the thing it is asked to rank
has a 4.3× narrower spread there, while its own error grows by 1.6× (part
genuine off-distribution degradation, part the PBE→ORB functional offset). Its
absolute accuracy barely moves; the target collapses below its noise floor.

That narrowing is not an accident of sampling — **it is what the generator's
$E_\text{hull}$ conditioning did**. The conditioned model emits genes that are
already concentrated near the hull, which is exactly the regime where a
0.12–0.20 eV/atom critic has nothing left to say. This is the matrix note's §7
claim — *conditioning and reranking are substitutes, and they eat each other's
headroom* — measured on the mechanism rather than inferred from the yield.

Two consequences follow, and they point in opposite directions:

1. **Do not conclude the critic is useless.** An unconditional pool should have
   a wider spread than this one, and the critic may well rank it. That is an
   expectation, not a measurement: the pool would also be off-distribution in
   ways the archive is not, and only running it settles the question. It is the
   one measurement in this note that was not possible without training.
2. **Do not expect to stack them.** Whatever a critic buys on an unconditional
   pool is plausibly the same quantity the conditioning already buys. Budget for
   one or the other, and re-measure after any change to the generator.

One caveat on the residual comparison: the archive-side 0.120 is PBE against
PBE, while the pool-side 0.198 is PBE against ORB. The 1.65× growth therefore
mixes genuine off-distribution degradation with the functional offset, and this
data cannot separate them. Fitting the critic on the functional the protocol
scores with — subject to invariant 9 — would remove one of the two.

---

## 4. Search versus knowledge: $\tau/s$, and whether restarts are worth it

### 4.1 $s$, the realisation spread

From the 589 genes with three formula-consistent successful trials
(positional DoF ≥ 3, so the trial schedule gave them three):

- mean(max − min) across restarts **0.163 eV/atom**, median 0.094
- exponential-scale estimator $\hat s$: **mean 0.108, median 0.061 eV/atom**
- 13.1% of genes give three relaxations agreeing to within 1 meV/atom

The censored model in `censored.py` predicts
$\mathbb E[\min_n] = m + s/n$, so best-of-1 minus best-of-3 should be
$\tfrac{2}{3}s = 0.072$ eV/atom with $s = 0.108$. Measured: **0.0721 eV/atom**.
The exponential-excess model is internally consistent on this data.

### 4.2 $\tau$, the critic's uncertainty

Residual sd of the critic against measured ORB `e_above_hull` on the pool:
**0.198 eV/atom** (robust 0.107).

$\tau/s \approx 1.8$ on robust scales, $\approx 3.2$ on means. **The regime is
knowledge-limited, not search-limited.** In the EMG score
$S = \nu[\Phi(z) - e^{a^2/2 - az}\Phi(z-a)]$ with $a = n\tau/s$, a large
$\tau/s$ is the limit where the score collapses towards $\nu\,\Phi(z)$ — the
user's original proposal — and where extra restarts barely move the score.

### 4.3 Restarts lose to fresh genes

On the same 589 genes, expected best-of-$k$ over all $\binom{3}{k}$ subsets:

| | mean $e_\text{hull}$ | median | $P(\text{metastable})$ | MetaSUN |
| :--- | ---: | ---: | ---: | ---: |
| best-of-1 | +0.187 | +0.153 | 0.316 | 0.199 |
| best-of-2 | +0.133 | +0.105 | 0.480 | 0.321 |
| best-of-3 | +0.115 | +0.087 | 0.547 | 0.370 |

Restarts help a lot per *gene* — MetaSUN 0.199 → 0.370. But a third trial costs
exactly what a third gene costs. **Per relaxation spent**: one trial each on
three genes yields 0.199 expected MetaSUN per relaxation; three trials on one
gene yields 0.370/3 = 0.123. **Restarts are worth 62% of a fresh gene.**

This is invariant 1 — take the top $N$ by $S(g,1)$, one relaxation each —
confirmed on data rather than argued from concavity. The schedule
`0:1,2:2,*:3` ran at 2.45 trials per gene, so ~1.45 relaxations per gene went
to restarts worth 0.62 of a fresh draw each.

Two things sharpen rather than weaken this. The measurement is on the DoF ≥ 3
stratum, which is **where restarts help most** — a zero-DoF gene re-realised
gives back nearly the same structure — so the 62% is an upper bound over the
pool. And it holds *at the current acceptance rate*, where genes are free and
plentiful; it inverts only if the pool is ever small enough that the
water-filling allocation of §1.1 of the matrix note actually binds.

---

## 5. Novelty: the gene is a poor proxy, and the occupancy formula works

### 5.1 The gene→structure novelty gap

|  | structure known | structure novel |
| :--- | ---: | ---: |
| **gene known** | 145 | **241** |
| **gene novel** | 154 | 460 |

241 of 386 gene-known draws (62%) relax into a novel structure, and 154 of 614
gene-novel draws (25%) relax into a known one. Restricted to the metastable
realisations — the ones that matter:

$$P(\text{structure novel} \mid \text{gene novel}) = 0.626, \qquad
P(\text{structure novel} \mid \text{gene known}) = 0.491.$$

**A gap of 0.135.** Gene novelty barely predicts structure novelty among the
successes. Worse, the lookup points the wrong way on stability:
$P(\text{metastable} \mid \text{gene novel}) = 0.523$ against
$P(\text{metastable} \mid \text{gene known}) = 0.606$.

The anticorrelation that caps everything is at the *structure* level:
$\phi(\text{metastable}, \text{structure-novel}) = -0.321$, whereas
$\phi(\text{metastable}, \text{gene-novel}) = -0.082$. This is measurement 5 of
the matrix note's §11, and the answer is the pessimistic one — **most of the
gap between the 1.75× energy oracle and the 3.16× arithmetic ceiling lies in structure novelty,
which no gene-level feature can see.** Ranking gene-novel genes by a perfect
energy oracle reaches only 1.93× at $N = 300$.

### 5.2 The occupancy formula is calibrated where it matters

$\nu(g) = (1-p_\theta(g))^M \approx \exp(-M p_\theta(g))$, against the actual
training-set lookup, $M = 4{,}007{,}729$ rows ($\log M = 15.20$):

| surprisal (median of octile) | observed $P(\text{novel})$ | $\exp(-M p_\theta)$ |
| ---: | ---: | ---: |
| 12.01 | 0.080 | 0.000 |
| 13.17 | 0.176 | 0.001 |
| 14.09 | 0.328 | 0.064 |
| 15.26 | 0.552 | 0.393 |
| 16.76 | 0.816 | 0.801 |
| 18.79 | 0.960 | 0.969 |
| 21.53 | 1.000 | 0.997 |
| 27.96 | 1.000 | 1.000 |

**Above the half-point the formula is essentially exact.** Below it, it is
badly over-confident: it says "certainly known" where a third of genes are in
fact novel. Brier score 0.126 against 0.237 for the constant base rate.

One free parameter fixes the lower half: fitting $\nu = \exp(-e^{c - \text{surprisal}})$
gives $\hat c = 13.90$ against the nominal 15.20, an effective
$M \approx 1.1\times10^6$ against 4.0×10⁶ rows. Distinct genes account for only
a small part of that (3,959,797 distinct genes in the reference); the rest is
the conditional-versus-marginal gap of the matrix note's §7.2 — $p_\theta(g \mid e = 0)$ is not
the archive marginal — plus ordinary model error. **Recalibrate $c$
empirically; do not plug in $\log(\text{row count})$.**

Measurement 6 of the matrix note's §11 therefore passes: the axis-(ii) novelty term is usable,
with a fitted constant and a documented failure region.

---

## 6. Two pieces of machinery that the archive cannot support

### 6.1 $s(g)$ is not estimable from the archive

Rows per augmentation-invariant gene in `lemat_bulk_fmax1_stress`
(5,327,342 rows, 4,826,004 distinct genes, mean 1.104 rows per gene):

| rows per gene | genes | % of genes | % of rows |
| ---: | ---: | ---: | ---: |
| ≥ 1 | 4,826,004 | 100% | 100% |
| ≥ 2 | 331,578 | 6.87% | 15.6% |
| ≥ 3 | 89,738 | 1.86% | 6.6% |
| ≥ 5 | 15,358 | 0.32% | 1.9% |
| ≥ 10 | 1,521 | 0.032% | 0.45% |
| ≥ 20 | **195** | **0.004%** | 0.15% |

**93.1% of genes are singletons.** Measurement 3 of the matrix note's §11 asked whether row 2's
per-gene $s$ machinery is usable from the archive alone; it is not, by three
orders of magnitude. Cell (2a) of the summary matrix is correctly described as
the weakest cell, and this quantifies *how* weak.

On the 195 genes that do have ≥20 rows, $\hat s$ is median 0.032, mean
0.059 eV/atom — about half the protocol-side $s$ of §4.1, which is expected:
archive rows for one gene come from different source databases and search
campaigns, not from repeated random realisation of the same manifold.

### 6.2 The $p(\text{sg} \mid S)$ backoff recovers 11%

The one measurement that gates the chemical-system exploration plan. Holding out
4,000 chemical systems with ≥30 rows each (300,141 rows) from 5.3M:

| model | row-weighted KL | recovered |
| :--- | ---: | ---: |
| global marginal $p(\text{sg})$ | 0.861 nats | — |
| element-marginal backoff, geometric pool, $\lambda = 0.5$ | 0.766 nats | **11.0%** |
| element-marginal backoff, arithmetic pool, $\lambda = 1.0$ | 0.783 nats | 9.1% |

A never-seen chemical system costs 0.861 nats if you ignore it — consistent with
the 0.92 nats/system recorded in `chemical_system_sampler.md` — and a naive
element-marginal backoff recovers about a ninth of that. **The gate is mostly
still shut.** Forced exploration into unseen systems will sample space groups
close to the global prior. A learned model over the element set may do better;
this measurement rules out the cheap version, not the idea.

---

## 7. What this changes in the matrix note

Section numbers below are the **matrix note's**.

1. **§2.2 consequence 2**: the interior optimum is at ~50% acceptance on this
   generator, not ~20%. Below 10% acceptance a *perfect* energy oracle loses
   money.
2. **§11 item 1**: report the top-$N$ lift, not $1+\mathrm{CV}^2$, unless
   $\hat r$ has been calibrated first (this note's §2.3).
3. **§11 item 3 is answered: no.** 93% of archive genes are singletons.
4. **§11 item 4 is answered**: $\tau/s \approx 2$–3, knowledge-limited.
5. **§11 item 5 is answered, pessimistically**: the metastable gene→structure
   novelty gap is 0.626 vs 0.491, and structure-level novelty carries the
   anticorrelation ($\phi = -0.321$) that caps the whole scheme.
6. **§11 item 6 is answered: yes, with a fitted constant.**
7. **§11 item 7 is answered: 11% recovered.** Not enough to unblock exploration
   on its own.
8. **Invariant 1 is now measured**, not just argued: restarts return 62% of a
   fresh gene per relaxation.
9. **Invariant 8 does not bite on this checkpoint** — the overlap is 11.6% and
   accuracy is flat across it — but the reason to keep the invariant stands.
10. **A new invariant is needed**, but not the one that looked likely: a critic's
    usefulness is set by its error *relative to the spread of the pool it ranks*,
    and a well-conditioned generator narrows that spread deliberately. Measured:
    the same critic scores Pearson +0.979 on the archive and +0.061 on the
    conditioned pool, with its own error changing by a factor of 1.6 (§3).

## 8. Reproducing

Scripts are in the session scratchpad rather than the repo, because each is a
few dozen lines against artifacts already in W&B. In order:

| what | inputs |
| :--- | :--- |
| oracle enrichment, novelty gap (§1, §5.1) | `protocol_ehull5x-20260904-213346:v1` → `structures.csv` |
| $s$, restarts (§4.1, §4.3) | same artifact → `relaxations.csv`, **filtered by formula** (§0.1) |
| gene features | `score_gene_likelihood` at `energy_above_hull` ∈ {0, 0.05, 0.1}, 32 permutation samples; `wyformer-gene-screen`'s `score_genes` at `max_force = 0` |
| selector, ablation, permutation null (§2) | 10-fold CV, `HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, lr=0.06, l2=1.0)` |
| critic diagnostics (§3) | `cache/lemat_bulk_fmax1/data.pkl.gz` test split, 20k rows, scored both as a formation energy and, via `HullLookup`, as a hull distance |
| archive gene census (§6.1) | `cache/lemat_bulk_fmax1_stress/data.pkl.gz`, `gene_energy._fingerprints` |
| $p(\text{sg} \mid S)$ backoff (§6.2) | same cache, 4000 held-out systems with ≥30 rows |

The generation condition for the pool is not recorded in the protocol manifest;
§2's likelihoods assume `energy_above_hull = 0`. Surprisal at 0 and at 0.1
correlate at 0.924, and the ablation deletes the likelihood group without loss,
so no conclusion here depends on that assumption.
