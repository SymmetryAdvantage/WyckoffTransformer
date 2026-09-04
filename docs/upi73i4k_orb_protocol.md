# `upi73i4k` under the ORB ranking protocol

Run 2026-09-04. Genes: `generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz`
(2500 genes, `--condition-value 0.0`). Outputs in
`generated/upi73i4k/protocol/`.

Settings, from `manifest.json`: ORB v3 conservative-inf
(`orb-v3-conservative-inf-omat-20250404`) against its own LeMat-Bulk hull,
1 PyXtal trial × 2 CrySPR stages, `fmax 0.05`, 20 workers on one RTX 6000 Ada.
52 min wall for the 1873 gene-novel genes, 3 min for the remaining 627, 8.6
worker-hours in total.

## The funnel

| step | count | per sampled gene |
|---|---:|---:|
| sampled genes | 2500 | 1.0000 |
| valid gene | 2500 | 1.0000 |
| unique gene | 2500 | 1.0000 |
| produced a structure | 2490 | 0.9960 |
| valid structure | 2271 | 0.9084 |
| unique structure | 2271 | 0.9084 |
| novel structure | 1806 | 0.7224 |
| `e_hull ≤ 0.1` — **MetaSUN** | 290 | **0.1160** |
| `e_hull ≤ 0` — **SUN** | 7 | **0.0028** |

`e_above_hull` over the 2271 valid structures: mean 0.364, median 0.241, sd
0.406, min −0.030, max 3.379 eV/atom; 25.9% ≤ 0.1, 15.2% ≤ 0.05, 1.5% ≤ 0.

Uniqueness is exact and unremarkable: `StructureMatcher` found no duplicate
among the 2271, so every valid structure is distinct.

## By degrees of freedom

Every table here is also reported per positional degree of freedom — Σ `dof`
over the gene's Wyckoff sites, the number of free coordinates PyXtal has to
guess. It is the variable that controls reconstruction difficulty, and the model
emits a wide range of it (median 3, mean 5.3, max 186), so an aggregate hides
two opposing trends.

Rates are per sampled gene *within the bin*.

| Σ dof | genes | structure | valid | valid rate | novel | novel rate | median `e_hull` | MetaSUN | SUN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 493 | 493 | 437 | 0.886 | 155 | 0.314 | 0.131 | 0.0669 | 0.0020 |
| 1–2 | 565 | 563 | 521 | 0.922 | 391 | 0.692 | 0.218 | 0.1239 | 0.0071 |
| 3–5 | 706 | 705 | 635 | 0.899 | 589 | 0.834 | 0.274 | 0.1431 | 0.0014 |
| 6–10 | 446 | 444 | 412 | 0.924 | 405 | 0.908 | 0.306 | 0.1166 | 0.0022 |
| >10 | 290 | 285 | 266 | 0.917 | 266 | 0.917 | 0.281 | 0.1172 | 0.0000 |
| **all** | **2500** | **2490** | **2271** | **0.908** | **1806** | **0.722** | **0.241** | **0.1160** | **0.0028** |

Structure validity is flat — 0.886 to 0.924, with no trend — so the pipeline's
ability to produce a *legal* structure does not depend on complexity. Everything
else does:

- **Novelty rises steeply with dof**, 0.314 to 0.917. Simple genes are mostly
  already in LeMat-Bulk, which is what one would expect: there are only so many
  ways to decorate a high-symmetry cell with no free coordinates.
- **Stability falls with dof**, median `e_hull` 0.131 to ~0.30.
- **MetaSUN is the product of the two**, so it peaks in the middle (0.143 at
  dof 3–5) and is *lowest* where reconstruction is easiest (0.067 at dof 0).
  Reading the aggregate 0.116 alone hides this entirely.

**Read the high-dof rows with the reconstruction caveat below.** At one PyXtal
trial the reconstruction stage recovers a known structure 98.9% of the time at
dof 0 and 0% at dof >10, so the falling `e_hull` trend is part real chemistry
and part search failure, and the two are not separated here. That is the
argument for a dof-aware trial budget rather than a uniform one.

## Relaxing the gene-known genes changes the reading

This run is the first under the revised protocol, which relaxes every unique
gene rather than discarding the 627 whose Wyckoff fingerprint already occurs in
LeMat-Bulk. Splitting the funnel by that screen shows why it matters.

| cohort | genes | structure | valid | novel | mean `e_hull` | median | ≤ 0.1 | ≤ 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gene-novel | 1873 | 1863 | 1688 | 1688 | 0.417 | 0.307 | 0.161 | 0.0041 |
| gene-known | 627 | 627 | 583 | 118 | 0.212 | 0.085 | 0.542 | 0.0463 |
| **all** | 2500 | 2490 | 2271 | 1806 | **0.364** | **0.241** | **0.259** | **0.0150** |

- **The energy distribution was biased.** The gene-known cohort is far more
  stable than the novel one — 54.2% at or below 0.1 eV/atom against 16.1%, and
  a median three and a half times lower. Scoring only the gene-novel genes
  conditions `e_above_hull` on novelty and drops exactly the genes that
  reproduce real materials, so the old mean of 0.417 overstated the model's
  typical output; over all 2500 genes it is 0.364.
- **118 of the discarded genes were not duplicates at all.** They share a
  fingerprint with a LeMat-Bulk entry, but `StructureMatcher` rejects the
  match: same space group and same elements on the same Wyckoff orbits,
  different structure. 18 of them are metastable, which is the whole difference
  between MetaSUN 0.1088 and 0.1160.
- **CrySPR recovery, measured.** Of the 583 gene-known genes that gave a valid
  structure, **465 (79.8%)** relax back to a LeMat-Bulk structure the matcher
  accepts. These genes were extracted from real materials, so this is a direct
  control on the reconstruction stage on inputs whose answer is known, and it
  is consistent with the ceiling the [reconstruction
  study](cryspr_reconstruction_study.md) is designed to measure. Note also that
  100% of them produced a structure at all (627/627) against 99.5% for the
  novel cohort.

  The aggregate is not the useful form. By degrees of freedom, at one trial:

  | Σ dof | genes | valid | recovered | of valid | of genes |
  |---|---:|---:|---:|---:|---:|
  | 0 | 315 | 285 | 282 | **0.989** | 0.895 |
  | 1–2 | 180 | 171 | 130 | 0.760 | 0.722 |
  | 3–5 | 95 | 91 | 46 | 0.505 | 0.484 |
  | 6–10 | 31 | 30 | 7 | 0.233 | 0.226 |
  | >10 | 6 | 6 | 0 | **0.000** | 0.000 |
  | **all** | **627** | **583** | **465** | **0.798** | **0.742** |

  At dof 0 PyXtal has nothing to guess, so one trial is every trial. The
  collapse to zero above dof 10 means best-of-1 is not a constant penalty but a
  **complexity-dependent** one, which does not cancel between arms whose dof
  distributions differ. See [the trial-budget
  question](#the-trial-budget-is-the-open-question).

## Do the recovered structures and energies match LeMat-Bulk?

The 583 gene-known genes that gave a valid structure split cleanly in two, and
the split is much sharper than the 79.8% match rate alone suggests.

### The 465 that match: geometry and energy to ~1 meV/atom

- **Geometry.** All 465 also match with `scale=False`, so the lattice agrees
  and not merely the motif. Volume per atom, ours over theirs: mean 1.0000,
  median 0.9991, sd 0.0156, interquartile range [0.9952, 1.0034].
- **Energy on the same scale.** Our ORB-relaxed E/atom against an ORB
  single-point on the LeMat-Bulk geometry: mean +0.17 meV/atom, median
  −0.08 meV/atom, sd 8.9 meV/atom. **86.5%** agree within 1 meV/atom, 97.8%
  within 10, 99.4% within 50.
- **`e_above_hull` against DFT.** Ours (ORB against the ORB hull) against
  LeMat-Bulk's own DFT value, all 465: mean 0.119 vs 0.121, median 0.058 vs
  0.061, Pearson r **0.988**, Spearman 0.976, mean signed difference
  −0.001 eV/atom (sd 0.028). 98.1% agree within 50 meV/atom.

So where CrySPR recovers the structure, the whole pipeline reproduces
LeMat-Bulk — geometry, ORB energy, and the DFT stability ranking.

### The 118 that do not: mostly a missed basin, occasionally a better one

They are not tolerance artefacts. `StructureMatcher.get_rms_dist` returns
`None` for every one — no site correspondence at all — and only 13 of the 118
match even at CDVAE tolerances (`ltol 0.3, stol 0.5, angle_tol 10`).

Relaxing each reference entry on the same PES with the same optimiser, so the
comparison is like for like (this lowers the reference by a median of only
0.1 meV/atom, so a single point would have said the same):

| outcome | n | share | ours − theirs, eV/atom |
|---|---:|---:|---|
| ours **higher** by >1 meV/atom — missed the reference basin | 102 | 86.4% | median +0.339, 80 of them by >0.1 |
| ours **lower** by >1 meV/atom — a better minimum for the same gene | 15 | 12.7% | median −0.060, min −0.795 |
| within 1 meV/atom | 1 | 0.8% | — |

The 102 are reconstruction failures, and they show up in the stability numbers:
the non-matching cohort has mean `e_above_hull` 0.576 against 0.119 for the
matched one, with 15.3% at or below 0.1 eV/atom against 64.1% and none at all
below 0.

The 15 are the `lower_energy_alternative` case the [reconstruction
study](cryspr_reconstruction_study.md) defines — the gene is genuinely
ambiguous and CrySPR found a better ORB minimum than the one LeMat-Bulk
records. 8 of them are metastable. At 15 of 583 they are 2.6% of the gene-known
set, so they matter for interpreting individual structures rather than for the
aggregate rates.

## Against the MACE/LeMat-GenBench run

`generated/upi73i4k/RESULTS.md` scored the same genes with MACE-MP through
LeMat-GenBench. MetaSUN per generated gene barely moves — 0.1160 here against
0.114 there — but the stability distribution is much harsher under ORB: mean
`e_hull` 0.364 vs 0.308, metastable fraction 0.259 vs 0.303, stable 0.015 vs
0.041.

Three things differ at once, so this is not a model-to-model comparison:

- the potential and its hull (ORB against ORB's, MACE against `mace_mp`'s,
  whose checkpoint is unidentified — see the [protocol
  doc](de_novo_ranking_protocol.md));
- the relaxation budget: 1 trial at `fmax 0.05` with two stages here, 3 trials
  there, and GenBench re-relaxed at `fmax 0.02` before scoring;
- novelty: `StructureMatcher` here, structure-matcher-based novelty in the
  `single_mlip` config there (0.740).

## The trial budget is the open question

One PyXtal trial per gene is the protocol's biggest deliberate approximation,
and the recovery table above shows it is not a uniform one. Two further
measurements, on the gene-known cohort re-run with 3 trials:

**How often do the trials of a gene even disagree?** Spread between the best and
worst trial's converged energy:

| Σ dof | genes | disagree by >10 meV/atom | median spread |
|---|---:|---:|---:|
| 0 | 263 | 0.065 | 0.0001 |
| 1–2 | 149 | 0.443 | 0.0005 |
| 3–5 | 81 | 0.716 | 0.277 |
| 6–10 | 22 | 0.727 | 0.095 |
| >10 | 5 | 1.000 | 0.247 |

Two thirds of these genes get the same answer from every trial, so extra trials
buy nothing; the tail gets a different answer every time. A uniform budget
overspends at dof 0 and underspends above it. (This cohort is half dof-0
genes against a fifth for the full 2500, so read the conditional pattern, not
the marginal.)

**A cheap pre-screen does not work.** If the energy after a few optimisation
steps identified which trial converges lowest, the protocol could generate many
candidates, spend ten steps on each, and fully relax only the best — best-of-N
coverage at nearly best-of-1 cost, since generation is free and relaxation is
not. It does not:

| steps | picks the winning trial | mean regret, eV/atom |
|---:|---:|---:|
| 0 | 0.363 | 0.069 |
| 5 | 0.371 | 0.072 |
| 10 | 0.375 | 0.069 |
| chance (1 of 3) | 0.333 | 0.122 |

The raw PyXtal energy is dominated by atomic overlap and carries almost no
information about which basin the trial will fall into. Ten steps do not fix
that. This route is closed.

What remains, in order of how well the evidence supports it:

1. **Allocate the budget by dof**, sized to equalise recovery probability rather
   than compute — flat recovery across dof is what makes two variants
   comparable when their dof mixes differ. Better still, run trials until two
   agree and cap by dof, which adapts within a bin. Dof is a property of the
   gene, so the allocation costs nothing to decide.
2. **Predict the free coordinates.** The gene fixes symmetry and occupancy but
   not the free coordinates or the lattice, so PyXtal draws them near-uniformly
   and any trial budget is rejection sampling against that. This is the only
   option that raises recovery at dof >10 rather than making it affordable, and
   it can sit downstream of WyFormer, so nothing is retrained. Two measurements
   say what to predict and how hard it is:

   | quantity | mean | median | max |
   |---|---:|---:|---:|
   | positional dof | 5.27 | 3 | 186 |
   | lattice dof | 2.29 | 2 | 6 |

   The lattice is half the nominal search space — 19.7% of genes have no
   positional freedom at all — but almost none of the difficulty: those same
   genes recover 98.9% of the time, because the cell is relaxed by gradient
   descent in both stages and is effectively unimodal. Predicting the lattice,
   volume included, buys optimisation steps, not basins. **The failures are
   entirely positional.**

   And the target is well posed. Of the 627 gene fingerprints in this run that
   occur in LeMat-Bulk, **617 (98.4%) map to exactly one structurally distinct
   entry**; 10 map to two; the mean is 1.016. So the conditional distribution of
   the free coordinates given the gene is very nearly a point mass, and a plain
   regressor — trainable on the 4.2M gene/coordinate pairs already in the
   Wyckoff cache — would start the relaxation in the right basin's neighbourhood
   by construction. A generative sampler is needed for de novo diversity, not
   for reconstruction fidelity. (LeMat-Bulk is curated and may record one
   polymorph per gene, so the potential energy surface can hold more basins than
   the dataset holds entries: CrySPR found a strictly lower ORB minimum for 15
   of 583 known genes.)

   **Before any of that, there is a constant to fix.** PyXtal's cells come out
   systematically too large: against the LeMat-Bulk reference the initial volume
   per atom has a geometric mean ratio of **1.177** (median 1.174), and only 25%
   of trials start within 10% of the right volume. That is not an MLIP/DFT
   disagreement — the ORB-relaxed volume over the DFT reference volume has
   median **1.000** (interquartile range 0.996-1.007), so ORB reproduces the DFT
   cell and the excess is entirely the starting guess.

   It is associated with recovery, for genes with positional freedom:

   | start volume / reference | trials | recovery |
   |---|---:|---:|
   | <0.80 | 57 | 0.368 |
   | 0.80-0.95 | 73 | 0.603 |
   | **0.95-1.05** | 141 | **0.766** |
   | 1.05-1.25 | 339 | 0.652 |
   | 1.25-1.60 | 255 | 0.588 |
   | >1.60 | 71 | 0.549 |
   | all | 936 | 0.623 |

   Stage 1 relaxes positions at *fixed cell*, so the guessed volume is the box
   the positions are arranged in before the cell is ever released -- which is
   why the guess is consequential rather than merely a slow start. Landing in
   the 0.95-1.05 band is worth +0.14 absolute recovery over the current mix.

   Two hard-coded constants in ``cryspr/generator.py`` inflate the cell, neither
   exposed as a parameter: ``from_random`` is never passed ``factor``, so it
   takes PyXtal's default of **1.1**, and ``_DEFAULT_IADM =
   Tol_matrix(prototype="atomic", factor=1.3)`` inflates every minimum
   interatomic distance by 30%, forcing the cell open further. Setting the
   volume factor to ~0.935 would move the median ratio to 1.00 and raise
   "within 25% of correct" from 60% to 77%.

   The table above is an association, not an intervention: genes whose volume
   PyXtal estimates well may be easier for other reasons. The clean test is to
   re-run the same genes with only that constant changed, which is cheap and
   not yet done.

   A related cheap option is a learned re-ranker from the unrelaxed structure to
   the relaxed energy. The physics version of this is dead (see the pre-screen
   table above), but every protocol run generates the supervised data for a
   learned one for free — each trial directory holds the initial CIF and the
   converged energy.
3. **Rattle-and-re-relax as a basin-hopping move** replacing independent
   restarts. Untested, and the weakest of the three: releasing the symmetry
   constraint already moves the energy by >1 meV/atom in only 0.4% of trials, so
   a small kick probably stays in the same basin, and a kick large enough to
   escape may cost as much as a fresh draw. Worth one experiment, not a plan.

### Rejected: capping dof at sampling time

The cheapest response would be to refuse the hard cases — cap dof at sampling
and never generate what CrySPR cannot reconstruct. **Decided against.** It is
recorded here because the numbers are worth knowing and the option keeps
suggesting itself; the reason it fails is visible in the two denominators.

| dof cap | kept | MetaSUN per kept gene | MetaSUN per sampled gene | relax time saved |
|---|---:|---:|---:|---:|
| ≤0 | 493 | 0.0669 | 0.0132 | 97.7% |
| ≤2 | 1058 | 0.0974 | 0.0412 | 88.7% |
| ≤5 | 1764 | 0.1156 | 0.0816 | 68.6% |
| ≤10 | 2210 | 0.1158 | 0.1024 | 47.6% |
| none | 2500 | **0.1160** | **0.1160** | 0% |

Capping at dof ≤10 discards 11.6% of genes, costs **0.0002** in MetaSUN per kept
gene, and saves **47.6%** of the relaxation budget — high-dof genes are a ninth
of the count but nearly half the compute (mean 50.6 s per gene against 1.5 s at
dof 0). As a way to spend less, it is excellent.

As an *evaluation* choice it is circular, whichever denominator is used:

- **Per kept gene**, a model that emits nothing but hard genes scores the same
  as one that emits nothing but easy ones, because the rejected genes vanish
  from both numerator and denominator. The rejection rate then becomes an
  uncontrolled difference between arms.
- **Per sampled gene**, rejection is identical to scoring every high-dof gene as
  a failure — which is precisely the best-of-1 bias, now made explicit rather
  than removed. MetaSUN drops from 0.116 to 0.102 at a dof ≤10 cap, and that
  drop is an artefact of the cap, not of the model.

There is also a discovery cost that the MetaSUN number does not show. Novelty
rises monotonically with dof — 0.314 at dof 0 against 0.917 above 10 — so the
genes a cap removes are the ones least likely to be in LeMat-Bulk already.
Capping trades away exactly the unexplored part of the space, and it does so
without improving the metric, because MetaSUN is nearly flat above dof 1–2
(0.124, 0.143, 0.117, 0.117). The cap buys compute, not quality.

So the protocol generates and scores every gene, whatever its dof, and pays for
the hard ones. If relaxation cost ever becomes the binding constraint, a cap is
the lever with the best ratio — but it is a budget decision to be stated
outright, never a quiet default, and the stratified tables above are what make
its effect legible if it is ever taken.

Whatever the budget, **report MetaSUN stratified by dof**. It is free and it
keeps the reconstruction bias visible instead of confounded with the quantity
being measured.

## What this run changed in the protocol

Two defects surfaced while running it.

**Structure novelty was vacuous.** The score stage hashed each structure with
BAWL and tested membership in `data/unique_fingerprints.parquet`. That file
holds no BAWL hashes: all 4,719,106 entries are LeMat-GenBench *augmented
Wyckoff* fingerprints (`AUG_12_('Ba', '4j', 1):1_...`), which a BAWL hash
(`<md5>_<formula>`) can never equal, so every structure came back novel — the
first ORB funnel reported novelty as exactly 1688/1688. BAWL is now gone;
novelty and uniqueness both use the two-stage filter in
`evaluation/novelty.py`. See the [protocol doc](de_novo_ranking_protocol.md).

**Workers left CUDA contexts on the wrong GPU.** Passing `device="cuda:1"` is
not enough: anything reaching for the *current* device initialises `cuda:0` and
leaves a ~200 MB primary context there, 4 GB across 20 workers, on a card
someone else was training on. `_pin_visible_device` now sets
`CUDA_VISIBLE_DEVICES` per worker before CUDA initialises.

Two environment notes, neither a code defect:

- ORB on CUDA needs `libomp.so.5`, which this machine does not have:
  `/usr/local/magma/lib/libmagma.so`, pulled in by `libtorch_cuda_linalg.so`,
  links it, so every `torch.linalg` call on GPU raises `Error in dlopen`. ORB's
  conservative regressor calls `torch.linalg.det` for the stress, so nothing
  runs. Worked around with a symlink to Intel's ABI-compatible `libiomp5.so`
  (GPU and CPU energies then agree to 0.3 meV on Si-diamond); the real fix is
  to install an LLVM `libomp5` package.
- Worker thread pinning has to happen in the parent. A spawned worker imports
  pandas, and MKL fixes its thread count when it is first loaded, so setting
  `OMP_NUM_THREADS` inside the pool initialiser is too late and every worker
  runs multi-threaded.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md)
- [`upi73i4k` e-hull-conditioning audit](upi73i4k_ehull_conditioning_audit.md)
- [How much does Wyckoff → structure cost us?](cryspr_reconstruction_study.md)
