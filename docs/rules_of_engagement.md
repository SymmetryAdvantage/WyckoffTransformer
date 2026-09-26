# Rules of engagement: WyFormer's inference modes

> **STATUS (2026-09-26).**
> - The composition layer, the four modes and the per-sampled-gene accounting
>   are in `wyckoff_transformer.roe`.
> - The tensor-space novelty and uniqueness checkers are in
>   `evaluation/gene_hash.py`. They are the default and are measured below.
> - **broadside, fire-discipline and fire-control have been compared** at 1000
>   reconstructions each. The backbone is the CFG model at
>   `e_hull = 0.05`, w = 3; results are [below](#results).
> - **torpedo-run has been run** on the chemical-system backbone, with two paired
>   arms from one pool ([below](#torpedo-run-results)).
>   - Scoring each gene against a hull that includes the other candidates beat
>     the DFT-hull ranking: MetaSUN 0.613 against 0.542 (p = 0.0015), at 20% less
>     relaxation time.
>   - SUN did not move (18 against 20).
>   - A per-system residual correction of the energy predictor failed held-out
>     validation, and was not used.

Generating a (M)SUN structure is an attack on the convex hull: a candidate that
lands below it does not merely pass a threshold, it redraws the hull beneath
itself. The four inference modes differ in how much is checked before the
expensive shot is fired, so they are named for rules of engagement.

**Naming rule.** Where a quantity already has a name -- `e_hull`, formation
energy, the convex hull, novelty, uniqueness, validity, metastable, stable, SUN,
MetaSUN, relaxation, gene -- it keeps it, in the docs, the code and the columns.
The naval vocabulary names the one thing that had no name: the modes. A reader
should never have to decode a metaphor to find out what was measured.

## The four modes

| # | Mode | Pipeline | What it adds |
|---|---|---|---|
| 1 | **broadside** | generate → reconstruct | nothing; the baseline |
| 2 | **fire-discipline** | generate → screen → reconstruct | uniqueness and novelty, by fingerprint lookup |
| 3 | **fire-control** | generate → screen → predicted `e_hull` → reconstruct | a predicted formation energy compared with the hull |
| 4 | **torpedo-run** | sample chemical system → generate conditioned on it → predicted `e_hull` → screen → reconstruct | a chosen target |

Ordering is part of a mode, not an implementation detail. `fire-control`
screens before it ranks; `torpedo-run` ranks before it screens. Two modes with
the same components in the other order are different experiments, so the order
is in the mode's definition (`roe/plan.py`) and in its manifest.

`wyformer-roe list -v` prints the same table with each mode's rationale.

**Formal validity is not one of the filters.** A gene naming a Wyckoff letter
its space group does not have is not a cheap reconstruction that a careless mode
would attempt and a careful one would skip -- PyXtal has nothing to place. Every
mode drops it, none is credited for it, and it is reported separately.

### Why each filter is where it is

**`fire-discipline`: the free lookup, first.** The fingerprint lookup against
the reference archive is exact, needs no model, and removes two kinds of wasted
reconstruction: the same gene twice, and a gene LeMat-Bulk already holds. On the
`e9ywwsie` pool it was worth 2.78x MetaSUN at a budget of 250
([generative novelty screen](generative_novelty_screen.md)). It is the first
thing any mode past the baseline should do.

**`fire-control`: screen before you rank.** An energy ranker left to itself
finds low-lying genes partly by finding compositions the archive already holds:
its best-scoring decile was 90% metastable and 83% already-known formulas, and
MetaSUN peaked a decile *lower*
([the DFT screen uplift](archive/e9ywwsie_dft_screen_uplift_report.md)). Removing
the overlap first and ranking the remainder is what recovered the gain, so that
is the order the mode fixes.

**`torpedo-run`: rank before you screen.** Inside a named chemical system most
genes are novel, so the screen has less to remove and less reason to run first;
and the predicted-hull filter is the cheaper of the two whenever the 4.8M
reference fingerprints are not already resident. This ordering is a judgement,
not a measurement -- it is the first thing the mode comparison should check.

## The accounting

A mode filters before the reconstruction, so the protocol's rates -- which are
per gene the protocol was handed -- answer "how good were the genes that got
through". That is not the question a mode is chosen on. `roe/report.py`
restates every rate against the genes the mode *drew*, by feeding the protocol's
own `funnel_structure_metrics` a `GeneScreen` built from the cohort's weights.
MetaSUN must mean exactly what it means in
[the ranking protocol](de_novo_ranking_protocol.md); a second implementation of
those masks would eventually mean something slightly else.

Three quantities, in `engagement.json`:

| | |
|---|---|
| `per_sampled_gene` | the funnel, restated against the drawn cohort |
| `cost.trials_charged` | reconstruction trials the mode is charged for |
| `trials_per_hit` | the first divided by the second: **the number the escalation exists to move** |

**A cohort marks, it never drops.** Filters write verdict columns and a `kept`
mask; no gene leaves the table and no index is reused. That is what keeps the
denominator fixed and the avoided cost recoverable, and a filter that tried to
re-admit a gene an earlier one removed raises rather than being absorbed.

**Duplicates are charged to `broadside` without being relaxed.** It runs no
uniqueness screen, so a faithful campaign at those rules would have reconstructed
every duplicate as though it were new. The protocol nonetheless deduplicates,
because that is what the protocol does, so the cost is recovered from the
cohort's weights instead of spent. This makes `broadside`'s **cost exact and its
yield a lower bound**: the duplicate draws would have been extra trials of the
same gene, and extra trials sometimes find a lower minimum. The bias runs
against the conclusion the other modes want, which is the right direction for it
to run.

## The first comparison: broadside, fire-discipline, fire-control

> **Result (2026-09-24).** At a fixed 1000 reconstructions each, fire-control
> made 646 MetaSUN structures, fire-discipline 549 and broadside 410. The
> ordering holds per trial too (4.2, 5.0 and 6.3 trials per hit). Fire-control
> also quadrupled SUN (34 against 8 and 10). **But per relaxation worker-hour
> it is the worst of the three for MetaSUN** (14.1 against 20.8 and 19.8),
> because the predicted hull favours cells of about 52 atoms against about 30.
> W&B run
> [`roe_cfg_c0p05_w3-20260923`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/roe_cfg_c0p05_w3-20260923),
> artifact `roe_cfg_c0p05_w3`.

Run on iapetus on 2026-09-23/24, at commit `e4ef930` plus the driver change
that lets `scripts/platforms/iapetus/run_rules_of_engagement.sh` pass sampling
flags to the pool (`POOL_GEN_ARGS`, `POOL_OVERSAMPLE`).

- **Backbone:** the [classifier-free guidance](classifier_free_guidance.md)
  model `ehull_adamw_wsd_5x_cfg-20260916-055000`, artifact `:v33` (epoch
  39,999). It samples at `energy_above_hull = 0.05` with guidance scale 3, the
  recommended setting there.
- **Energy predictor:** `min_energy_5x_adamw_wsd-20260921-125850`, artifact
  `:v6` (epoch 2858). It is a scalar MSE model of
  `gene_min_formation_energy_per_atom`, trained on `lemat_bulk_fmax1_stress`.
- **Where the models came from:** both were fetched from W&B into the runs
  store and read from disk.

A first attempt used the unconditional backbone
`unconditional_5x_ehull01-20260915-151250` and the predictor
`min_energy_adamw_wsd-20260912-115957`. It was stopped on 2026-09-21 when the
[Wyckoff augmentation defect](wyckoff_augmentation_audit.md) turned up, with
broadside partly relaxed, and was not resumed. Its files are in
`/mnt/hdd/kna/wyformer/roe/{pool,broadside}`.

**One pool, three selections, one budget.** A pool of 10,000 formally valid
genes was drawn once, from 12,500 draws. All three modes select from that
pool, so they differ only in what they select, not in sampling noise. Each
mode reconstructs exactly **1000 genes**. The expensive half is therefore held
fixed, and what separates the modes is the selection.

| mode | selection | pool consumed |
|---|---|---|
| broadside | the first 1000 genes, duplicates included | 1000 |
| fire-discipline | the first 1000 unique *and* novel genes | 1553 |
| fire-control | the 1000 lowest predicted `e_hull` among every unique, novel gene in the pool | 10,000 (6515 passed the screen) |

**Fire-control selects a budget, not a threshold.**
- On this pool, 258 of the 6515 screened genes (4.0%) were predicted at or
  below the hull, and the 1000th gene was cut at a predicted `e_hull` of
  **0.030 eV/atom**.
- A 300-gene pilot from the unconditional backbone had found 10 of 187 (5.3%)
  at or below the hull.
- Filling 1000 from a threshold at zero would take about 40,000 draws.
- `--energy-select threshold` is still available. It is the right choice when
  the question is how many genes a generator puts below the hull, rather than
  how to spend a budget.

**Reconstruction** is the de novo ranking protocol unchanged:
- ORB `orb_conserv_inf`, the default trial schedule, and both the
  fixed-symmetry and free readouts, scored against `lemat_bulk_fmax1_stress`
  (4,823,981 fingerprints, after the augmentation fix).
- Five workers: two on each K20c and one on the GTX 750 Ti, as
  [the host's usage notes](platforms/iapetus/usage.md) prescribe.
- The modes ran one after another, because the score stage holds the reference
  at about 25 GB and this host has 30 GB.

### Results

The table below uses the free (post-rattle) track, with Wilson 95% intervals.
Most rows are per **reconstructed** gene, the fixed budget, and three rows are
per sampled gene.

| | broadside | fire-discipline | fire-control |
|---|---|---|---|
| gene novelty of what was reconstructed | 0.642 | 1.000 | 1.000 |
| novel structure | 0.653 | 0.878 | 0.846 |
| metastable | 0.647 | 0.564 | 0.678 |
| P(metastable \| novel structure) | 0.628 | 0.625 | **0.764** |
| **MetaSUN** | 0.410 [0.380, 0.441] | 0.549 [0.518, 0.580] | **0.646** [0.616, 0.675] |
| MetaSUN, fixed-symmetry track | 0.302 | 0.430 | **0.531** |
| stable | 0.035 | 0.012 | 0.041 |
| SUN | 0.008 [0.004, 0.016] | 0.010 [0.005, 0.018] | **0.034** [0.024, 0.047] |
| reconstruction trials charged | 2572 | 2747 | 2718 |
| **trials per MetaSUN** | 6.27 | 5.00 | **4.21** |
| trials per SUN | 322 | 275 | **80** |
| relaxation worker-hours | 20.8 | 26.4 | 45.8 |
| mean atoms per trial | 30.2 | 32.9 | 51.9 |
| **MetaSUN per worker-hour** | 19.8 | **20.8** | 14.1 |
| SUN per worker-hour | 0.39 | 0.38 | **0.74** |
| MetaSUN per *sampled* gene | **0.410** | 0.354 | 0.065 |
| wall time on iapetus | 4.8 h | 6.0 h | 12.1 h |
| worker faults (GTX 750 Ti out of memory, retried) | 3 | 0 | 17 |

Differences in MetaSUN (Fisher exact):
- fire-discipline − broadside: +0.139 (p = 6e-10)
- fire-control − fire-discipline: +0.097 (p = 1e-5)
- fire-control − broadside: +0.236 (p = 4e-26)

Differences in SUN:
- fire-control − fire-discipline: +0.024 (p = 3e-4)
- fire-discipline − broadside: +0.002 (n.s.)

The arms share genes from one pool, so they are not independent samples.
Pairing lowers the variance of a difference, so these p-values are
conservative.

**The screen does what it was designed to do.**
- It lifts novel structures from 0.653 to 0.878 per reconstruction, and MetaSUN
  by the same order (+0.139).
- Metastability falls (0.647 → 0.564), because the screen removes known
  genes, and on this backbone known genes relax into metastable structures far
  more often than novel ones.
- P(metastable | novel structure) does not move (0.628 → 0.625).
- In other words, fire-discipline swaps known hits for novel ones and changes
  nothing else.
- It costs 1.55 draws per reconstruction instead of 1. Generation is cheap:
  the whole 12,500-draw pool took 16 minutes on the CPU.

**The energy ranking adds real selection on top of the screen.**
- Fire-control is the only arm that changes how likely a *novel* structure is
  to be metastable: 0.764 against 0.625.
- It is also the only arm that moves SUN, which is 3.4× fire-discipline's rate
  and significant even at n = 1000.
- The protocol's power analysis puts a SUN comparison at about 10,000 genes
  when the rates are similar. A 3–4× gap is resolvable at 1000.

**The price is cell size, and it matters for cost.** The lowest predicted
`e_hull` genes have larger cells, 52 atoms per trial against 30–33.
- Relaxation cost per trial is roughly proportional to the number of atoms,
  so fire-control used 2.2× broadside's worker-hours for about the same
  number of trials.
- Counted in trials, fire-control wins (4.2 trials per hit). Counted in GPU
  time, which is the resource actually spent, fire-discipline wins for
  MetaSUN (20.8 per worker-hour against 14.1), and fire-control wins only for
  SUN (0.74 against 0.38).
- The larger cells also accounted for most of the out-of-memory retries on
  the 2 GB card.
- **So `trials_per_hit` is not the whole cost.** Any comparison of modes that
  select on energy should also report worker-hours.

**Which mode to use:**

| goal | mode |
|---|---|
| MetaSUN per unit of compute | fire-discipline |
| SUN, or MetaSUN per reconstruction slot (e.g. DFT follow-up, where cost per structure matters more than cell size) | fire-control |
| nothing | broadside: it wins only per *sampled* gene, and generation is cheap |

**Compared with the guidance study.** This broadside's MetaSUN is 0.410,
against 0.479 for the same recipe on zeus in
[the CFG study](classifier_free_guidance.md). The difference is significant
(Fisher p = 0.002), and it lies almost entirely in metastability (0.647
against 0.711, p = 0.003). The novelty gap is not significant (novel
structure 0.653 against 0.687, p = 0.12). Checked on 2026-09-24; files are in
`/mnt/hdd/kna/wyformer/roe/cfg-c0p05-w3/{zeus_w3,zeus_rescore,single_point_*.csv}`.

Ruled out:
- **The hardware.** ORB single-point energies of zeus's 979 relaxed structures,
  recomputed on iapetus, match zeus's recorded values to within −0.00008 eV/atom
  on average (5–95%: [−0.00025, +0.00008]; largest 0.0006). The control,
  iapetus's own structures, reproduces to ±0.00002. So there is no offset of
  0.025–0.030 eV/atom between zeus and iapetus on the same structure.
- **The Wyckoff augmentation fix.** The zeus arm was screened against the
  pre-fix reference (4,826,004 fingerprints) and this run against the post-fix
  one. Re-screening and re-scoring zeus's cohort on the post-fix reference
  changes 2 gene-novelty verdicts, no structure-novelty verdict and no energy.
  Its MetaSUN stays at 0.479.
- **Settings.** The trial schedule (`0:1,2:2,*:3`), timeouts, ORB checkpoint
  and hull revision are identical.
- **A biased slice of the pool.** Broadside's first 1000 genes match the rest
  of the pool in space group and cell size.

Not ruled out:
- **Chance between two independent 1000-gene cohorts.** w = 3 was also the
  best of the relaxed guidance arms, so its 0.479 carries some
  winner's-curse bias.
- **Relaxation trajectories.** They could differ even though the energy
  function does not. iapetus lost more trials (32 failed against 11) and had
  OOM retries. The direct test is to relax zeus's own PyXtal draws
  (`pyxtal.extxyz`) on iapetus.

Open:
- **Fire-control confounds energy with cell size.** Its selection is
  partly a selection for larger cells. Ranking at a matched atom count, or
  charging per atom, would separate the two.
- **Selection strength is one point.** The 1000 kept of 6515 is a 15% cut;
  the curve of MetaSUN and SUN against the cut has not been measured.
- **Torpedo-run** is [below](#torpedo-run).

## Torpedo-run

> **Status (2026-09-26).** Run on ASPIRE 2A on 2026-09-25, at commit `6dce0f8`
> (branch `roe-torpedo`), with `scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh`:
> - PBS `25537541` is the full run: 2 h 41 min on one 4×A100 node. W&B run
>   [`roe_torpedo_chemsys_sg_uncond_adanmw_wsd-20260924`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/aweaetcy),
>   artifact of the same name. Outputs are in
>   `$WYFORMER_RUNS/roe/torpedo_chemsys_sg_uncond_adanmw_wsd`.
> - PBS `25537540` is the `--pilot` (3 targets, 10 genes per arm). W&B run
>   [`gx5w6u4p`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/gx5w6u4p).
>
> Results are [below](#torpedo-run-results).

Three things the other modes did not need had to be added before this mode could
be fired.

### (a) A target is a chemical system and all of its subsystems

The chemical-system-conditioned model is asked for "exactly these elements"
([chemical-system mode](chemical_system_mode.md)). Asking it for A-B-C therefore
generates ternaries, but the A-B-C hull is bounded by A-B, A-C and B-C.
- A ternary is only a hit if it is below the binaries.
- The binaries a campaign could find belong on that hull as much as the ones the
  archive holds.

`SubsystemClosureSampler` (`--closure`) handles this:
- It draws the targets once from the backbone's own system prior, as distinct
  ternaries.
- Every plan gives each target an equal share of the rows. Half goes to the
  target itself, and the other half is split evenly over its binaries.
- A binary shared by two targets gets both shares.
- Unaries are not generated. The elemental references are DFT's, and a generated
  element predicted below one would re-base every formation energy in the
  system.

The mechanics are in [the sampler's doc](chemical_system_sampler.md#aiming-at-targets-and-their-subsystems).

### (d) A hull built from the predictions, and what the residuals are for

`fire-control` compares each gene, on its own, with the DFT hull at its
composition. Candidates never compete with each other, which causes two
problems:
- Two hundred polymorphs of one formula predicted below the hull all look like
  hits. This did not happen in the run below: both arms picked almost only
  distinct formulas.
- A ternary predicted below the DFT hull still looks like a hit when a binary in
  the same cohort is predicted lower still.

**The joint hull** (`--energy-hull joint`, `formula_energy/joint_hull.py`) puts
every distinct non-unary candidate into the phase diagram next to the DFT
reference. Each candidate is then scored against **the hull of everything else**:

    predicted e_hull(c) = E(c) - H_{-c}(x_c)

- If `c` is not a vertex of the joint hull, this is its ordinary `e_above_hull`,
  which is >= 0.
- If `c` is a vertex, the score is negative: how far below the reference and
  every other candidate it sits.
- The best polymorph of a formula is therefore scored against the next best, not
  against the archive. The unit tests pin this, along with a ternary losing its
  place to a new binary (`tests/test_torpedo_components.py`).

**The residuals.** For a gene LeMat-Bulk already holds, the regressor's error is
observed, not unknown. `gene_energy_residuals.py` uses that in two ways.

1. **A known gene needs no prediction.** Its DFT gene-minimum energy replaces the
   predicted one (`known_dft_formation_energy`).
   - Such genes are later removed by the screen, but they still shape the joint
     hull before that.
   - This way a spuriously low prediction for a known gene cannot push real
     candidates up.
2. **The error is local.** A per-chemical-system offset shifts every candidate in
   the system against the DFT hull by the same amount, which is exactly what a
   hull comparison is sensitive to.
   - The correction is hierarchical shrinkage: a system's own mean residual,
     shrunk toward the pooled residuals of its subsystems, and those toward the
     global mean.
   - The corrected energy is `prediction - b(system)`.

**Only honest residuals are used.** The target, `gene_min_formation_energy_per_atom`,
is a minimum over the whole fingerprint class *across splits*. So a val or test
gene whose fingerprint also appears in `train` had its target shown to the
model.
- The table keeps only val and test genes absent from `train`.
- The correction is fitted on val and scored on test, over κ ∈ {1, 3, 10, 30,
  100, 300}. The report is in `residual_validation.json` and in the W&B summary
  under `residuals/`.
- **A run uses the correction only if it removes at least 1% of the held-out
  MAE.** κ is picked on the same test set, so a smaller gain cannot be told apart
  from that choice. Otherwise the run falls back to the raw prediction, with the
  known-gene substitution still applied.
- **On the full table the correction failed.** Validation used 169,205 honest
  residuals: 84,584 to fit (val) and 84,621 to test.

  | | test MAE, eV/atom | AUC for DFT `e_hull <= 0.05` |
  |---|---|---|
  | raw prediction | **0.03918** | **0.9652** |
  | global offset only | 0.03924 | 0.9652 |
  | κ = 300 (best) | 0.03927 | 0.9652 |
  | κ = 30 | 0.04022 | 0.9636 |
  | κ = 10 | 0.04253 | 0.9593 |

  Every κ is worse than no correction, and the smaller κ is (the more a system's
  own residuals are trusted), the worse it gets. A system's mean residual on some
  genes does not predict the residual on its other genes: at 50 meV MAE the
  errors are gene-level, not system-level. The run therefore used the raw
  prediction plus the known-gene substitution.
- A CPU smoke test on 10k rows per split had pointed the same way (0.0385 raw
  against 0.0385 at best), and so had the pilot's 50k rows (0.03787 against
  0.03789).

**Predictions average 8 equivalent Wyckoff descriptions** (`--augmentation-samples 8`).
With one description, each call draws a random one. In the smoke test, the same
Er3Hg4 gene scored 0.009 and 0.003 eV/atom against the reference hull in the two
arms. The arms of a paired comparison must not disagree about a gene they share.

### A budget taken after every filter

The mode ranks before it screens. A `top-k` cut in the energy slot would spend
the budget on duplicates and known genes that the screen then removes.
- `--energy-select rank` keeps every scored gene and declares a `rank_column`.
- `Engagement.run` then keeps the best `--target-engaged` survivors **of every
  filter**, and records the cut as a `budget` stage.

**Does the order matter?** For the reference-hull score it no longer changes the
selection: each gene's score is its own, and the budget is taken last. For the
joint score it does, because the energy slot builds the hull before the screen
removes duplicates and known genes. Known genes enter that hull at their DFT
energies, which the reference already holds, so the difference should be small.
Nobody has measured it.

### The run

`scripts/platforms/aspire2a/roe_torpedo_in_pbs.sh` runs on one 4×A100 node:

- **Backbone:** `chemsys_sg_uncond_adanmw_wsd-20260921-220745`. This is
  provisional: no chemical-system-conditioned model exists yet on the CFG /
  `der_tokenizer_v1` recipe.
  - It is conditioned on the chemical system only.
  - It was trained on `lemat_bulk_fmax1_stress_ehull01` with
    `lemat_bulk_fmax1_sg_multiplicity`.
  - Its prior is the `system_prior.npz` in its run directory, not the dataset
    cache's.
- **Energy predictor:** `min_energy_adamw_wsd-20260924-102431`, the der critic
  on `der_tokenizer_v1`.
  - It re-tokenises PyXtal genes itself, so the two tokenisers need not agree.
- **Targets:** 50 ternaries drawn with seed 0, closed under their binaries, for a
  pool of 20,000 draws. The prior follows the corpus: the smoke test's three
  targets were Ho-Er-Hg, Ac-Er-Tl and Nd-Ag-Hg.
- **Arms**, one pool and paired, 1000 reconstructions each:

  | arm | hull | energy |
  |---|---|---|
  | `joint` | joint | corrected (or raw plus known-gene DFT, if the correction fails validation) |
  | `reference` | DFT reference | raw: the selection the earlier modes used |

  The `joint` arm's `cohort.csv` holds all four `predicted_e_hull_<hull>_<energy>`
  columns for every gene in the pool. The reference arm only holds the
  reference-hull ones, because the joint hull is computed only when it is
  selected on.
- **Reconstruction:** the protocol defaults (ORB `orb_conserv_inf`, reference
  `lemat_bulk_fmax1_stress`), 4 workers per GPU.
- **Results:** one W&B run, `roe_torpedo_chemsys_sg_uncond_adanmw_wsd-<date>`.
  Its summary holds each arm's `engagement.json` and the residual validation, and
  one artifact holds the pool, plan, residual table and every arm's outputs.

**Caveat:** the backbone differs from the CFG one that broadside, fire-discipline
and fire-control used, so this run's rates are not directly comparable with those
three. The comparison it supports is between its own two arms.

### Torpedo-run results

> **Result (2026-09-25, commit `6dce0f8`, W&B `aweaetcy`).** At 1000
> reconstructions each from one 20,000-gene pool:
> - **MetaSUN (free track):** the joint-hull arm made 613, the DFT-hull arm 542.
>   That is +0.071, Fisher p = 0.0015.
> - **SUN:** the same in both arms, 18 against 20.
> - **Cost:** the joint arm was 24% more efficient per relaxation worker-hour for
>   MetaSUN (59.6 against 47.9).
>
> What moved is metastability, not novelty. The genes only the DFT-hull ranking
> picked are ones other candidates in the cohort undercut, and after relaxation
> they land well above the hull.

**The pool.**
- 20,000 draws in 27 s on one A100, all formally valid.
- They spread over 50 ternary targets and their binaries.
- After the screen, 14,101 were unique and not in LeMat-Bulk. The other 29.5%
  were duplicates or known genes, in line with the 34% seen in the smoke test.
  So even inside a named system, the screen is not idle.
- On their own against the DFT hull, 707 genes were predicted at or below it.
  Against the joint hull, 400 were.

| | `joint` | `reference` |
|---|---|---|
| selection | joint hull, raw + known-gene DFT | DFT hull, raw |
| budget cut, predicted `e_hull` | 0.030 | 0.024 |
| genes: binary / ternary | 585 / 415 | 513 / 487 |
| distinct formulas | 963 | 948 |
| valid structure | 0.953 | 0.930 |
| novel structure | 0.850 | 0.842 |
| metastable | **0.716** | 0.629 |
| **MetaSUN** (free) | **0.613** [0.582, 0.643] | 0.542 [0.511, 0.573] |
| MetaSUN, fixed-symmetry track | 0.382 | 0.343 (p = 0.077) |
| SUN (free) | 0.018 [0.011, 0.028] | 0.020 [0.013, 0.031] |
| SUN, fixed-symmetry track | 0.012 | 0.015 |
| reconstruction trials | 2702 | 2734 |
| **trials per MetaSUN** | **4.41** | 5.04 |
| trials per SUN | 150 | 137 |
| relaxation worker-hours (A100, 16 workers) | 10.3 | 11.3 |
| mean atoms per trial | 29.2 | 30.7 |
| **MetaSUN per worker-hour** | **59.6** | 47.9 |
| SUN per worker-hour | 1.75 | 1.77 |
| wall time | 79 min | 72 min |

The table gives Wilson 95% intervals. Two caveats on the statistics:
- The arms share 791 of their 1000 genes, so they are not independent. Pairing
  lowers the variance of a difference, so the p-values are conservative.
- The shared genes were relaxed once per arm. Their MetaSUN verdicts agree 84.8%
  of the time (0.593 against 0.580 MetaSUN). That is the noise floor of a single
  1000-gene reconstruction.

**Where the gain comes from: the 209 genes each arm picked alone.**

| | joint-only (209) | reference-only (209) |
|---|---|---|
| MetaSUN | **0.689** | 0.397 |
| metastable | 0.861 | 0.550 |
| valid structure | 0.990 | 0.880 |
| median MLIP `e_above_hull` after relaxation | 0.026 | 0.092 |
| median predicted `e_hull`, DFT hull | 0.027 | 0.010 |
| SUN | 1 | 6 |
| mean atoms | 19.2 | 26.5 |

- **It is not polymorph crowding.** Both arms picked almost only distinct
  formulas, with at most 3 genes per formula. The motivating failure (two hundred
  polymorphs of one formula) did not occur on this backbone.
- **It is compositional competition.** The reference-only genes look best
  against the DFT hull alone. The joint hull scores them a median 0.042 eV/atom
  higher, because other candidates in the cohort, often binaries, sit below them.
  The relaxation agrees: they end up a median 0.092 eV/atom above the MLIP hull.
- **A hypothesis for why that works:** the joint hull cancels the regressor's
  shared local bias, by comparing a prediction with *other predictions* in the
  same chemistry. That is what the residual correction tried and failed to do
  from database residuals. This has not been tested directly.
- **The reference-only picks are also where most of the SUN hits came from**
  (6 against 1), and SUN is flat overall. The joint hull's competition therefore
  costs some of the deepest-below-hull bets. At 18–20 hits per arm this is not
  resolved.

**Scores against outcomes, on the union of 1209 reconstructed genes.**
- Within this already-selected set, the DFT-hull score is *anti*-correlated with
  outcome: Spearman −0.26 against MLIP `e_above_hull`, and AUC 0.39 for MetaSUN.
  In other words, lower predicted `e_hull` against the DFT hull meant *less*
  likely metastable.
- The joint score is flat on the same genes: Spearman −0.02, AUC 0.51.
- These are selection-truncated: every gene here is in the best 5% of the pool.
  They say the DFT-hull ordering is wrong at the top of the list, not that either
  score is useless on the pool.

**What the residuals bought.**
- The per-system correction bought nothing and was not applied (see
  [above](#d-a-hull-built-from-the-predictions-and-what-the-residuals-are-for)).
- The known-gene substitution put the DFT energy in place of the prediction for
  every known gene in the pool. That substitution only shapes the joint hull,
  since the screen removes those genes afterwards.
- Whether it contributed to the joint arm's gain separately from the joint hull
  itself was not measured. The 2×2 would need a joint-raw arm without the
  substitution.

**Against the CFG backbone's modes** (not paired, and a different backbone):

| | MetaSUN | SUN |
|---|---|---|
| this `joint` arm | 0.613 | 0.018 |
| CFG fire-control on iapetus | 0.646 | 0.034 |

The chemical-system backbone reaches nearly the same MetaSUN while choosing what
it aims at. Its SUN is half, and it is not a CFG / `der_tokenizer_v1` model. A
chemical-system model on that recipe is the obvious next backbone.

## The architecture

A mode names **slots**; a run fills them. Swapping an implementation into a slot
does not change what the mode means or what it can be compared against.

| slot | what belongs in it | filled today by |
|---|---|---|
| sampler | which chemical system and space group each structure is for | `SystemPriorSampler`, `PlanFileSampler` |
| source | the genes | `WyFormerGeneSource`; `GeneFileSource` for a shared pool, handed out in order |
| `screen` | uniqueness and novelty | `NoveltyUniquenessScreen`, `python` or `tensor` backend |
| `energy` | predicted formation energy against the hull | `PredictedHullFilter`, by threshold or by budget |
| `surprisal` | the generator's own log-density | `SurprisalBandFilter` |
| reconstructor | genes → relaxed, scored structures | `CrySPRReconstructor` |

Every component in `roe/builtin.py` is an adapter. Nothing there implements a
screen, a generator, an energy model or a relaxation: the screen is
`evaluation.protocol.screen_genes`, the energy filter is
`cli.gene_screen.score_genes`, the reconstructor runs `cli.protocol`'s own
stages, and the sampler is `system_prior.SystemSpaceGroupPrior`. When a
component looks like it is doing real work, that work belongs in the module it
wraps.

Adding one means writing a class with `name`, `slot`, `provides`, `requires`,
`apply(cohort)` and `describe()`, and nothing else: the protocols in
`roe/components.py` are structural, so a component is a component by having the
methods.

An assembly is validated **before anything expensive runs** -- a missing slot, a
filter in the wrong slot, a sampler in a mode that does not aim, a chain whose
input nothing upstream produces, a chemical-system-conditioned checkpoint with
no plan to condition on. All of those raise before the first checkpoint loads.

### Running it

```bash
wyformer-roe list -v

# The baseline.
wyformer-roe run broadside --model-path runs/<run> \
    --output-dir generated/<run>/broadside --n-genes 1000 \
    --condition energy_above_hull=0.05 -- --devices cuda:0,cuda:1

# Screen, then rank on the predicted hull.
wyformer-roe run fire-control --model-path runs/<run> \
    --output-dir generated/<run>/fire-control \
    --regressor-path runs/<gene-energy-run> \
    -- --devices cuda:0,cuda:1

# Choose the targets first. Needs a chemical-system-conditioned checkpoint, whose
# run directory carries the prior it was trained with.
wyformer-roe run torpedo-run --model-path runs/<chemsys-run> \
    --output-dir generated/<run>/torpedo-run \
    --system-prior runs/<chemsys-run>/system_prior.npz \
    --closure --n-targets 50 --regressor-path runs/<gene-energy-run> \
    --energy-select rank --energy-hull joint --target-engaged 1000 \
    -- --devices cuda:0,cuda:1

# Or, for paired arms: draw one pool, then select from it once per arm.
wyformer-roe draw --model-path runs/<chemsys-run> --output-dir pool \
    --system-prior runs/<chemsys-run>/system_prior.npz --closure --n-targets 50 \
    --n-genes 20000
wyformer-roe run torpedo-run --genes pool/wyckoff_genes.json.gz \
    --system-plan pool/system_plan.json ...
```

`--model-path` is used exactly as given. On a host whose runs live outside the
checkout, pass the absolute path.

Everything after a bare `--` goes to `wyformer-protocol` untouched, so the
reconstruction keeps every flag it has -- the MLIP, the devices, the trial
schedule, the timeouts -- without this CLI mirroring any of them.

`--no-reconstruct` stops at the filtered gene file, which is what a run whose
reconstruction happens on another machine wants. `wyformer-roe report <dir>`
rebuilds `engagement.json` from a directory afterwards.

To compare modes rather than run one, draw a pool once and point every arm at it
with `--genes`: each consumes only as much of it as its budget needs, so the arms
are paired and their generation costs stay separately visible.
`scripts/platforms/iapetus/run_rules_of_engagement.sh` is exactly that, for this
host.

### Two component decisions worth knowing about

**A gene whose composition the reference hull does not cover is kept, not
dropped** (`--on-missing-hull keep`, the default). There is no hull to compare
it with, and a novel composition is exactly what a discovery campaign is looking
for; dropping it would make the energy filter select against novel chemistry by
construction, which is the failure
[the generative novelty screen](generative_novelty_screen.md) measured for an
energy ranker left to itself. `drop` is available and says so in the manifest.

**`SurprisalBandFilter` keeps a band, not a tail.** Ranking on surprisal in
either direction was worse than random on `e9ywwsie`; keeping a band and ranking
the survivors on energy was the best lookup-free arm there (2.48x). It is
refused outright for a chemical-system-conditioned generator, by
`gene_likelihood.score_gene_likelihood` itself: each gene would be scored under
its own conditioning and the numbers would not be comparable across genes. So
`torpedo-run` cannot use it.

## The tensor-space screen

Built, measured and default (`--screen-backend tensor`). Gene novelty and
uniqueness are decided on a pair of 64-bit integers per gene instead of a nested
`frozenset`, in `wyckoff_transformer.evaluation.gene_hash`.

### It is exactly equivalent, not an approximation

The fingerprint is a *set of multisets*:

    (space group, { { (element, site symmetry, enumeration) x count } for each
                    equivalent enumeration })

so the key is built the same way round -- hash each variant's **sorted** multiset,
then hash the **sorted, deduplicated** list of those variant digests together
with the space group. Sorting canonicalises a multiset, sorting-and-deduplicating
canonicalises a set, and neither step assumes anything about the augmentations.

That last point is the design decision, and checking it turned up something
else. The obvious canonical form is the *minimum over the augmentation orbit*.
It needs the relabellings to form a group -- and **they do**, in all 230 space
groups; see [the augmentation audit](wyckoff_augmentation_audit.md). But the
group property turns out not to be sufficient, because the representation the
fingerprint is built from is not equivariant: it pairs each site's *original*
site-symmetry symbol with its *relabelled* enumeration index, and in 26 space
groups a relabelling changes that symbol. Min-over-orbit is therefore exactly
equivalent to the fingerprint on the other 204 groups and not on those 26.

Hashing the whole variant set needs neither property, so this key is exactly as
correct as `record_to_augmented_fingerprint` -- which, as the audit shows, is
itself not as correct as it looks.

Collision probability over the 4.8M distinct genes of `lemat_bulk_fmax1_stress`
is about 4e-26 at 128 bits. The encoding is dataset-independent by construction:
an element enters as its atomic number and a site symmetry as its own UTF-8
bytes, never as a tokeniser id, so a table built from one model's cache answers
correctly for a model trained on another.

### What it cost and what it bought

Measured on iapetus, 2026-09-21/22, commit `9145354`, against
`cache/lemat_bulk_fmax1_stress` (5,327,342 rows, all three splits):

| | fingerprint set | key table |
|---|---|---|
| distinct genes found | 4,826,004 | **4,826,004** |
| on disk | 246 MB (`gene_fingerprints.pkl.gz`) | **74 MB** (`gene_keys.npz`) |
| resident | ~17 GB | **77 MB** |
| build | ~11 min | **~7 min** (one-off, cached beside the reference) |
| load, then screen 300 genes | ~2 min | **1.2 s** |

The first row is the strongest evidence available that the two encodings agree:
they find the same number of equivalence classes in the same 5.3M rows, and a key
that merged two fingerprints would show up as a smaller table, one that split a
fingerprint as a larger one.

The win is not only the 200x on memory. At 77 MB the screen can run *inside* a
generation loop rather than after it -- which is what makes the top-up to a fixed
reconstruction budget affordable, since every round re-screens the whole
accumulated cohort.

### How it is tested

`src/wyckoff_transformer/tests/test_gene_hash.py` (24 tests, no cache needed):

- **the partition test**, which is the one that matters: over 3000 random legal
  genes, two genes share a key exactly when they share a fingerprint -- an
  identity of the induced partitions, not a sample of spot checks. A companion
  test asserts the sample actually *contains* repeats, so the merge half of the
  claim is exercised rather than assumed;
- the invariances: site order, and every spelling of an element (`Element Fe`,
  `"Fe"`, `Element("Fe")`) that the caches and a restored processor disagree on;
- the distinctions: space group, elements, which element sits on which orbit
  (checked *against whatever the fingerprint answers*, since the augmentation
  makes that case subtle), and a repeated orbit, which a set would collapse and
  a multiset must not;
- **stability across processes** under three values of `PYTHONHASHSEED`, since
  Python randomises `hash()` per process and a cached table must not move;
- the table: membership against a Python set, an empty table, queries below and
  above every key, a matching low word with a different high word, the refusal
  of a duplicated low word, deduplication, and a save/load round trip including
  the refusal of a table from another encoding version;
- uniqueness: representatives and counts identical to `screen_genes` over a
  cohort seeded with duplicates, first-occurrence ordering, and the empty cohort.

`test_gene_hash_reference.py` (marked `needs_cache`) repeats the decisive checks
against the real archive: the class-count identity above, and that a 20,000-row
sample of the reference is found in the table built from it. Both passed on
2026-09-21.

`roe/tests/test_roe.py` pins the two backends *interchangeable* at the component
level -- same kept set, same `duplicates`, same `gene_novel`, same weights, over
four different archives including the empty one -- so a mode's result cannot
depend on which one ran.

### What is still Python, and why that is fine

| stage | tensor? | |
|---|---|---|
| uniqueness within a cohort | yes | `torch.unique` on the keys |
| gene novelty (candidate set) | yes | `searchsorted` against the sorted table |
| the per-gene key | no | `hashlib` over a canonical byte encoding |
| formal validity | no | still the mappings lookup that raises |
| energy prediction | already was | `build_tokenised_prediction_tensors` |
| hull lookup, formula in the table | partly | cached per chemical system |
| **hull for a novel formula** | no | a pymatgen phase diagram per system |
| **PyXtal draw** | no | rejection sampling in Python |
| **MLIP relaxation** | no | |
| **novelty verdict** | no | `StructureMatcher` on geometry |

The per-gene key is `hashlib` rather than a vectorised 64-bit mix because a
thousand-gene cohort takes milliseconds and the reference build is a one-off: at
this scale a hand-written mix would only be one more thing that has to be right.

The last four rows are where the wall-clock is, and none of them moved. That is
the honest limit of the exercise: it makes the *filters* nearly free, which is
what lets them run in a loop, but it does not make a mode cheaper to reconstruct,
and the fingerprint remains a candidate generator for the matcher, never a
verdict ([the ranking protocol](de_novo_ranking_protocol.md#novelty-and-uniqueness-are-two-stage)).

The tokeniser turned out not to be the constraint at all, which was the open
question. Nothing in the key refers to a tokeniser, so the "all models must share
one" precondition never arose. A model-specific gather table is still the right
answer for the *other* gene stages -- `prediction.filter_supported_tokens` is
still a Python loop over rows -- and is not built here.

## Known limitations

- **Torpedo-run is one pool on a provisional backbone.** The joint hull's gain is
  measured on one chemical-system checkpoint and one set of 50 targets.
  - The rank-before-screen ordering only matters through the joint hull, and was
    not varied.
  - Nor were the target share (0.5) and the number of targets.
- **One backbone, one pool, one budget.** The comparison above is a single
  10,000-gene pool from one guided checkpoint (CFG, `e_hull = 0.05`, w = 3) at
  one reconstruction budget, on one host. It says what these selections do to
  this generator's output, not what they do in general.
- **`broadside`'s yield is a lower bound**, for the reason in the accounting
  section. Its cost is exact.
- **DiffCSP++ is not wired up.** The repository reads DiffCSP++ *output*
  (`evaluation/DiffCSP_to_sites.py`) but has no path that hands it genes and
  gets structures back. `DiffCSPReconstructor` exists so that adding one is a
  component rather than a change to every mode, and refuses loudly meanwhile;
  `--no-reconstruct` writes the gene file for an external run.
- **The protocol re-screens the filtered gene file.** Not wasted -- it is the
  audit, and `score` refuses to mix a screen's verdict with a novelty reference
  other than its own -- but a mode with a screen pays the reference load twice
  unless the fingerprint cache is warm.
- **The `surprisal` slot is API-only.** No built-in mode uses it and the CLI
  does not expose it; a mode that does should be defined rather than patched in.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md) -- the
  reconstruction and scoring every mode ends in, and where MetaSUN is defined
- [Generative novelty screening](generative_novelty_screen.md) -- the surprisal
  lever, and the measurement that fixes `fire-control`'s ordering
- [Composition screening](composition_screening.md) -- the energy floor the
  `energy` slot estimates, and the hull it is compared with
- [The chemical system sampler](chemical_system_sampler.md) -- what
  `torpedo-run` aims with
- [Chemical-system mode](chemical_system_mode.md) -- what a chemical-system
  conditioned checkpoint was told, and what it was not
- [Every `e_hull` in this repository](e_hull_definitions.md) -- the predicted
  `e_hull` of the `energy` slot is not the MLIP `e_above_hull` the protocol
  scores with
