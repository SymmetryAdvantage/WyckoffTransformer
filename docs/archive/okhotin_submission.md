# Okhotin benchmark submission: which generator, which rule of engagement

> **Decision (2026-10-03): WyFormer-GeoCSP-CFG-v2.3.** The CFG e_hull generator
> `ehull_adamw_wsd_5x_cfg-20260929-150414` sampled at `energy_above_hull = 0.025`,
> guidance scale 6, under **fire-control keeping the top ~50%** of unique novel genes by
> predicted e_hull (`gene_min_ehull_adamw_wsd-20260929-154926`). On a fresh 2000-gene
> confirmation arm it scored **MSUN 0.7465 [0.727, 0.765] per submitted structure** under
> our ORB proxy, against 0.652 for the same generator under fire-discipline and 0.458 for
> the unconditional generator. The production run is in [Production](#production).

## Introduction: how a structure is made

The submissions are produced in two steps: first decide *what* the crystal is, then decide
*where* its atoms sit. Everything that selects which crystals to submit acts on the first,
cheap step.

**1. The Wyckoff gene.**
- A crystal's symmetry is given by its space group, one of 230.
- Within a space group, every atom sits on a *Wyckoff position*: a family of sites related
  by the group's symmetry operations, with a fixed multiplicity and a site symmetry.
- A *Wyckoff gene* is the list of (element, Wyckoff position) pairs plus the space group.
  For example: space group 225, Na on 4a, Cl on 4b is rock salt.
- The gene fixes the composition, the symmetry and how atoms are related to each other. It
  leaves open only a handful of continuous parameters: the lattice lengths and angles, and
  the free coordinates of the less symmetric Wyckoff positions.
- So it is a short, discrete, symmetry-exact description of a crystal. Our genes have a
  median of 4 sites (3–7 for 90% of them), where a full structure has dozens of
  coordinates.

**2. WyFormer: sampling genes.**
- WyFormer (Wyckoff Transformer, ICML 2025) is an autoregressive transformer over genes.
  Given a space group, it emits the gene's sites one at a time; each site is an element
  token, a site-symmetry token and an enumeration token.
- A Wyckoff position's sites form an unordered set, so the model is trained to be
  invariant to the order of sites and uses no positional encoding.
- **Unconditional (UC)** samples genes as they occur in the training set.
- **Conditioned (CON)** is the same model with a scalar input, the DFT energy above the
  convex hull (`e_hull`) of the training structure, so a target stability can be asked
  for at sampling time.
- **Classifier-free guidance (CFG)** trains one model both with and without the condition
  (the condition is dropped for 10% of training rows). At sampling time it extrapolates
  from the unconditional prediction towards the conditional one by a guidance scale w,
  which pushes harder towards the target than conditioning alone.
- Sampling a gene takes about a millisecond on a GPU.

**3. Choosing genes before any structure exists (the "rules of engagement").**
- **Broadside** submits genes as sampled.
- **Fire-discipline** first drops duplicate genes and genes already present in the
  reference dataset (alex-mp-20). This is an exact lookup of a gene fingerprint that is
  invariant to equivalent choices of Wyckoff setting.
- **Fire-control** additionally ranks the remaining genes with a second WyFormer, trained
  as a regressor to predict the lowest `e_hull` achievable by any structure with that
  gene. It keeps the best-ranked fraction (here ~50%).
- These steps cost milliseconds per gene. They decide which genes reach the expensive
  step, so the pipeline can afford to sample several times more genes than it submits.

**4. GeoCSP: from gene to 3D structure.**
- GeoCSP is our crystal-structure model, a heavily customised descendant of the DiffCSP++
  diffusion model. Given a gene, it samples the lattice and the free atomic coordinates.
- It starts from noise and denoises over 1000 steps with a symmetry-aware graph neural
  network.
- Each update is projected onto the degrees of freedom the gene leaves open. Coordinates
  fixed by symmetry stay exactly fixed, so the output has exactly the gene's space group
  and composition.
- This is the expensive step: about 1.5 GPU-seconds per structure.

**5. Rattle and submit.**
- A perfectly symmetric structure sits at a stationary point that a local relaxation
  cannot leave, even when a lower-energy, lower-symmetry structure is nearby.
- Each GeoCSP structure is therefore *rattled*, by random displacements of about 0.05 Å
  and a 1% random cell strain, before submission. This lets the organisers' relaxation
  find such distortions.

**How the choices were made.**
- To compare generators, sampling settings and rules of engagement, every candidate went
  through the same pipeline. Its structures were relaxed with the ORB-v3 machine-learned
  interatomic potential, scored against an ORB convex hull and matched against alex-mp-20
  for novelty.
- The metric is MSUN: the fraction of submitted structures that are **m**etastable
  (e_hull ≤ 0.1 eV/atom), **u**nique and **n**ovel.
- ORB is used only for this evaluation; it never touches the submitted structures.
- All models — WyFormer generators, the gene e_hull regressor and GeoCSP — were trained on
  alex-mp-20 `train.csv` only.

## Names

| label | generator | sampling | rule of engagement | submission directory |
|---|---|---|---|---|
| **WyFormer-GeoCSP-CFG-v2.3** | `ehull_adamw_wsd_5x_cfg-20260929-150414` | `energy_above_hull=0.025`, guidance 6 | fire-control, top ~50% | `submission_cfg_e0p025_w6_cut50` |
| **WyFormer-GeoCSP-CON-v2.3** | `ehull_adamw_wsd_5x-20260929-143848` | `energy_above_hull=0.025`, no guidance | fire-control, top ~50% | `submission_cond_e0p025_cut50` |
| **WyFormer-GeoCSP-UC-v2.3** | `uncond_adamw_wsd_5x-20260929-143845` | T = 1 | fire-control, top ~50% | `submission_uncond_t1_cut50` |

- All three share the gene e_hull predictor `gene_min_ehull_adamw_wsd-20260929-154926`
  and the structure model **GeoCSP** (W&B `symmetry-advantage/diffcsp/ua1g6od4:best`).
- GeoCSP is our own model, a heavily customised descendant of DiffCSP++. Its code lives in
  `/home/kna/DiffCSPNew`, and code identifiers there and here (`diffcsp_bridge`,
  `bench/run_diffusion.py --regime geov2`) keep the historical name.
- Intended use: CFG for maximum MSUN, CON for track 2 (discovery with diversity), UC for
  track 1 (similarity to the dataset).

## The benchmark and what we could measure

- **Rules.** Train on alex-mp-20 `train.csv` only; submit 10,000 structures; the
  organisers relax them with an MLIP, judge novelty against alex-mp-20 and stability
  against a hull they have not disclosed.
- **Decision metric:** MSUN per *submitted* structure. A duplicate, a failed GeoCSP
  start or an invalid relaxed structure counts as a miss.
- **Everything used to make the submission is trained on alex-mp-20 train only.**
  - Generators and the gene e_hull predictor: `production_training: false`, with val
    used only for checkpoint selection.
  - The predictor's gene minima are taken per split (`alex_mp_20_labelled_per_split`).
  - GeoCSP, our own crystal-structure model (a heavily customised descendant of DiffCSP++,
    GeoV2 architecture; `symmetry-advantage/diffcsp/ua1g6od4`, artifact `:best` = v28,
    epoch 140) was trained on alex-mp-20 train, and GeoV2 does not use ORB features.
- **ORB is evaluation only.** It imitates the organisers' relaxation and hull so that a
  model can be chosen. It never pre-relaxes, ranks or filters the submission.

### Evaluation, identical for every arm

genes → GeoCSP (one structure per gene) → CrySPR rattle (0.05 Å, strain 0.01,
seeded per gene) → **one unconstrained ORB-v3 conservative-inf relaxation**
(`--relax-schedule single`, BFGS + FrechetCellFilter, fmax 0.05, ≤1000 steps) → e_hull
on the LeMat-Bulk-MLIP-Hull ORB hull → StructureMatcher uniqueness within the arm and
novelty against alex-mp-20 train+val (`--reference-id-column material_id`).

- **The hull is a proxy.** The LeMat ORB hull is a superset of the alex-mp-20 hull, so it
  is a conservative stand-in for the undisclosed one.
- **The relaxation is a proxy too.** The organisers' MLIP and its settings are unknown.
  Absolute rates will move; the *ordering* of arms is what this study is for.
- **The structures evaluated are the structures submitted:** the same GeoCSP output
  and the same seeded rattle (`scripts/alex_bench/assemble_submission.py` checks this).
- Pinned checkpoints: uncond `best_model:v42`, e_hull `v50`, CFG `v55`, predictor `v37`
  (`scripts/alex_bench/fetch_wandb_runs.py`).
- Code: branch `okhotin-submission`. Driver:
  `scripts/platforms/zeus/run_alex_bench_setting.sh`. Per-arm rows are in
  `$STORE/alex_bench/summary.csv`, where `$STORE` is `/home/kna/.local/share/wyformer`.

All arms have n = 2000 submitted structures; brackets are Wilson 95% intervals. SE ≈ 0.011.

## Phase 1: the sampling setting per generator (fire-discipline)

| generator | setting | MSUN | SUN | metastable | stable |
|---|---|---|---|---|---|
| uncond | T = 1 | 0.458 [0.436, 0.480] | 0.0115 | 984 | 27 |
| e_hull cond | target 0 | 0.437 | 0.0220 | 960 | 51 |
| e_hull cond | **target 0.025** | **0.506** [0.484, 0.528] | 0.0110 | 1082 | 24 |
| e_hull cond | target 0.05 | 0.4875 | 0.0095 | 1021 | 22 |
| CFG | target 0, w 2 / 4 / 6 | 0.4075 / 0.3835 / 0.321 | 0.0445 / **0.0765** / 0.0715 | | |
| CFG | target 0.025, w 2 / 4 / **6** / 8 | 0.5765 / 0.633 / **0.652** / 0.648 | 0.014 / 0.0215 / 0.0215 / 0.027 | | |
| CFG | target 0.05, w 4 / 6 | 0.5475 / 0.532 | 0.0075 / 0.0045 | | |

**Target 0.025 is best for both conditioned generators, and CFG beats plain
conditioning by +0.15 MSUN.**
- On alex-mp-20, 0.025 sits between the hull spike (8% of rows at 0) and the median
  (0.046).
- Guidance at 0.025 plateaus at w = 6–8.
- At target 0, guidance trades MetaSUN for SUN: SUN reaches 7.7% at w = 4, the highest of
  any fire-discipline arm, while MSUN falls. A SUN-optimised submission would sample
  there.

**Deviations from the plan:**
- `cfg_e0p05_w2` was dropped. MSUN rose with w at every target measured, and target 0.05
  already trailed 0.025 by 0.085 at w = 4.
- `cfg_e0p025_w8` was added, under the plan's adaptive rule, because 0.025 still rose at
  w = 6.
- One structure in `cfg_e0_w6` (gene 659) stalled the charge-balance check for over two
  hours. Validity is now time-limited (`--validity-timeout 60`), and that structure is
  counted invalid.

## Phase 2: the 3 × 3

Broadside and fire-control are drawn from one 31,000-gene pool per generator.
Fire-control keeps the top 2000 of ~19–21k unique novel genes, a ~10% cut, as planned.
The fire-discipline cell is the Phase 1 arm at the same setting, from a different pool.

| MSUN (SUN) | broadside | fire-discipline | fire-control, 10% cut |
|---|---|---|---|
| uncond, T 1 | 0.3585 (0.008) | 0.458 (0.0115) | 0.591 (0.0705) |
| e_hull cond, 0.025 | 0.415 (0.0115) | 0.506 (0.011) | 0.6085 (0.081) |
| CFG, 0.025, w 6 | 0.4425 (0.0145) | **0.652** (0.0215) | 0.6485 (**0.1135**) |

- **Broadside is dominated everywhere.** It keeps the genes alex-mp-20 already holds,
  and 33–41% of its relaxed structures are known ones.
- **Fire-control at a 10% cut** adds +0.10–0.13 MSUN for the weaker generators, but
  nothing for CFG.
- For CFG, the cut raises metastability (1458 against 1391) but loses structure novelty
  (1759 against 1810, although every selected gene is gene-novel). Its lowest-predicted
  genes relax onto known alex-mp-20 structures.
- **The 10% cut is a SUN selector:** SUN rises 5–7× in every row.

## Phase 3: the cut, measured and confirmed

Every fire-discipline arm carries a prediction for each gene, so keeping its best
fraction f by predicted e_hull emulates fire-control at cut f on an unselected cohort.

| f kept | CFG 0.025 w 6 | CFG 0.025 w 4 | CFG 0.025 w 8 | cond 0.025 | uncond |
|---|---|---|---|---|---|
| 1.00 | 0.652 | 0.633 | 0.648 | 0.506 | 0.458 |
| 0.75 | 0.721 | 0.710 | 0.702 | 0.581 | 0.524 |
| **0.50** | **0.726** | **0.726** | 0.699 | 0.619 | 0.579 |
| 0.30 | 0.657 | 0.683 | 0.642 | 0.618 | 0.592 |
| 0.20 | 0.637 | 0.650 | — | 0.595 | 0.593 |

- MSUN peaks at f ≈ 0.5–0.75 and falls at tighter cuts; SUN keeps rising.
- The cut was read off the same data, so it was confirmed on a **fresh pool**: CFG 0.025
  w 6, top 2000 of 4207 unique novel genes (47.5%). The arm scored **MSUN 0.7465
  [0.727, 0.765]**, SUN 0.0465. Its 0 GeoCSP failures and 1932 valid structures match
  the other CFG arms.

**The gene e_hull predictor works, moderately.**
- On unselected cohorts its Spearman with ORB e_hull among novel valid structures is
  0.33–0.50.
- Within a 10% selection it is 0.08–0.15, as range restriction predicts.
- Its value is in the ranking: half the CFG genes can be discarded for +0.09 MSUN.

## Production

`scripts/platforms/zeus/make_alex_bench_submission.sh`, with:

```
SETTING_NAME=cfg_e0p025_w6_cut50 GEN_RUN=ehull_adamw_wsd_5x_cfg-20260929-150414 \
GEN_ARGS="--condition energy_above_hull=0.025 --guidance-scale 6" \
POOL_SIZE=35400 BUDGET=10400 N=10000
```

- **Pool and cut.** A 35,400-gene pool gives 20,635 unique novel genes, of which
  fire-control keeps 10,400 (50.4%, inside the measured optimum).
- **Pipeline.** GeoCSP runs on all 10,400 genes. `assemble_submission.py` walks them in
  predicted-e_hull order, skips any failed or rejected start, rattles each structure with
  its evaluation seed, and stops at 10,000.
- **Output** goes to `$STORE/alex_bench/submission_cfg_e0p025_w6_cut50/`: `cifs/`,
  `structures.extxyz` and `manifest.csv` (gene, predicted e_hull, formula), plus
  `manifest.json` with the expected ORB score of exactly the submitted subset.

**Result (2026-10-04 03:00):**
- GeoCSP produced a structure for 10,383 of 10,400 genes. The assembler took the first
  10,000 that passed: 17 genes had no structure and 6 had atoms closer than 0.5 Å.
- **Expected ORB score of exactly the submitted 10,000: MSUN 0.7315 [0.723, 0.740],
  SUN 0.0389 [0.035, 0.043]**, with 9644 valid and 9054 valid novel. All 10,400 scored
  0.732, matching the 0.7465 confirmation arm within its interval.
- Re-checked in a fresh process: 10,000 distinct genes and IDs, and every rattled
  structure passes `check_start`.
- **Cell size.** Cells are GeoCSP’s conventional cells: mean 19.6 atoms, max 87.
  - Counted from the genes, primitive cells average 11.0 atoms, and only 1.1% exceed
    alex-mp-20's 20-atom limit (pool: 0.7%). alex-mp-20 train averages 9.6.
  - spglib on the rattled cells cannot tell this: the 1% strain hides the centring.
- Files: `$STORE/alex_bench/submission_cfg_e0p025_w6_cut50/{cifs/,structures.extxyz,manifest.csv,manifest.json}`.

## Three submissions, one per objective

The benchmark has two tracks:
- **Track 1** scores *similarity to the dataset*. The fraction of structures passing each
  stability test should match the dataset's, and the passing structures should be
  distributed like the dataset's own passing structures.
- **Track 2** scores *discovery*: as many stable structures as possible, and diverse ones.

The CFG run above maximises MSUN. Guidance narrows the sampler, though: 9% of CFG genes
repeat within a 31k pool, against about 1% unguided. So two more submissions were made
with the same pipeline and the same ~50% fire-control cut.

| | **WyFormer-GeoCSP-CFG-v2.3** (CFG, target 0.025, w 6) | **WyFormer-GeoCSP-CON-v2.3** (track 2: conditioned, target 0.025) | **WyFormer-GeoCSP-UC-v2.3** (track 1: unconditional) |
|---|---|---|---|
| run | `cfg_e0p025_w6_cut50` | `cond_e0p025_cut50` | `uncond_t1_cut50` |
| pool → unique novel → kept | 35,400 → 20,635 → 10,400 (50.4%) | 31,200 → 20,822 → 10,400 (49.9%) | 29,900 → 20,812 → 10,400 (50.0%) |
| **expected MSUN**, submitted 10k | **0.7315** [0.723, 0.740] | **0.6551** [0.646, 0.664] | **0.5957** [0.586, 0.605] |
| expected SUN | 0.0389 | 0.0273 | 0.0234 |
| valid / valid novel | 9644 / 9054 | 9526 / 8959 | 9469 / 8996 |
| MSUN: distinct reduced formulas | 6995 | 6488 | 5921 |
| MSUN: distinct chemical systems | 4490 | **5336** | 5312 |
| MSUN: distinct chemical systems per MSUN structure | 0.61 | **0.81** | **0.89** |
| MSUN: space groups | 104 | 107 | 106 |
| space-group JS divergence vs alex-mp-20 train (all / MSUN) | 0.029 / 0.052 | 0.026 / 0.039 | **0.024 / 0.036** |
| elements per system in MSUN, 3 / 4 / 5 (train: 0.48 / 0.45 / 0.002) | 0.35 / 0.52 / 0.056 | 0.39 / 0.52 / 0.020 | 0.42 / 0.50 / 0.014 |

**Reading:**
- **CFG gives the most metastable structures, but fewer distinct chemical systems:** 4490,
  against 5336 for the unguided conditioned model, which has 10% fewer MSUN.
- CFG also over-produces quaternaries and quinaries relative to the dataset.
- **For a discovery score that weighs diversity, the conditioned model is the better
  trade**, which is why it was chosen for track 2.
- **The unconditional model is the closest to the dataset** on every distributional
  measure here, which is what track 1 rewards.
- These diversity and similarity numbers are our own proxies. The benchmark's own
  measures are not specified.

Both track runs (2026-10-04, `scripts/platforms/zeus/make_alex_bench_submission.sh` with
`POOL_SIZE=31200` and `29900`) ran side by side on GPU 1.
- **Wall times:** about 8.7 h of GeoCSP sampling each while sharing the card (about
  4.3 h alone), 1.7–1.9 h of relaxation while sharing CPU and card, and 1.1 h of scoring.
- **Gene-level work:** sampling, screening and ranking took 28–40 s, 2.5–2.7 s and
  22–23 s, as in the cost table below.
- **Track 2 lost one structure after the rattle.** One start (gene 9676) cleared the
  0.5 Å floor before the rattle (0.566 Å) and fell below it after (0.493 Å). The
  assembler now judges the rattled structure too, so that gene was dropped and the next
  one taken. All three submissions pass `check_start` in a fresh process.
- Files: `$STORE/alex_bench/submission_{cond_e0p025_cut50,uncond_t1_cut50}/`.

## Computational cost: genes are essentially free

Every selection in this submission happens at the gene level, before any 3D structure
exists. That is affordable because a Wyckoff gene costs about a millisecond to sample and
a few milliseconds to screen and rank, while turning one gene into a structure with
GeoCSP costs about 1.5 GPU-seconds. Discarding genes costs little; placing atoms is
expensive.

Measured on zeus (one NVIDIA RTX 6000 Ada, Intel Xeon w7-3455), production run of the
CFG submission (`production_cfg_e0p025_w6_cut50`, card not shared except for its first
15 min).

The card runs under a deliberate 250 W power cap (about 1 GHz SM clock under sustained
load). The GPU-bound GeoCSP figures are therefore slower than an uncapped card would
give, which only strengthens the comparison below.

| stage | input → output | wall time | per gene / structure |
|---|---|---|---|
| WyFormer sampling, CFG w=6 (two forward passes per step), GPU | 46,020 draws → 35,400 formally valid genes | 45.7 s | **1.0 ms per draw** |
| Uniqueness + novelty screen vs alex-mp-20 (tensor gene keys), CPU | 35,400 → 20,635 unique, novel | 2.7 s | **0.08 ms per gene** |
| Gene e_hull predictor + top-50% cut, GPU | 20,635 → 10,400 | 23.1 s | **1.1 ms per gene** |
| GeoCSP initialisation (PyXtal + CrystalNN graphs), 20 CPU threads | 10,400 genes | 7 min 3 s | 41 ms per gene |
| **GeoCSP sampling** (1000 steps, batch 128), GPU | 10,400 → 10,383 structures | **4 h 14 min** | **1.47 s per structure** |
| Start checks, rattle and writing the submission, CPU | 10,383 → 10,000 | ≈ 1.3 min | < 10 ms per structure |

- **Generation costs about the same for all three generators.**
  - Unconditional: 38,870 draws in 28.5 s, 0.73 ms each.
  - e_hull-conditioned: 40,560 draws in 39.8 s, 0.98 ms each.
  - CFG: 1.0–1.4 ms per draw.
  - Every run screened in 2.4–2.7 s for about 30k genes, and the predictor took 22–25 s
    for about 20k.
- **Process start-up dominates the gene-level wall clock:** loading models, the 670k-entry
  alex-mp-20 key table and the predictor takes about 30–60 s per process. The whole
  gene-level stage (sample, screen, rank, write) took 2.5 min wall for the 10,400-gene
  production, against 4 h 21 min for GeoCSP.
- **Per 1000 submitted structures:**
  - The CFG recipe generates 4,600 draws, screens 3,540 genes and ranks 2,060.
  - That is about 7 s of gene-level compute, against about 25 min of GeoCSP — **a ratio
    of roughly 1 : 200**.
  - Per gene, sampling a gene is about 1,500× cheaper than reconstructing one.
- **What the gene filters avoid:**
  - 77% of the drawn genes never reach GeoCSP: duplicates, genes already in
    alex-mp-20, and the predicted-unstable half.
  - Reconstructing all 46,020 would have taken about 19 GPU-hours instead of 4.2.
  - Fire-control trades about 45 s of extra gene sampling and ranking for a +0.09 MSUN
    gain over fire-discipline (0.7465 vs 0.652).
- **Evaluation, for scale (not part of making the submission):**
  - ORB relaxation of the 10,400 structures took 52 min on the same card (8 workers,
    0.30 s per structure).
  - Scoring took 53 min, 99% of it in the serial per-structure validity and
    charge-balance check over two readouts.

## Caveats

- **Proxy evaluation:** ORB relaxation and the LeMat ORB hull, not the organisers' MLIP
  and hull.
- **Pool mixing in the 3 × 3:** the fire-discipline cells come from different pools than
  the other two columns of their row (same setting, same distribution).
- **Settings chosen on the evaluation data:** the setting and the cut were picked on the
  same proxy they are scored with. The fresh confirmation arm guards against the
  winner's curse in the cut; the setting choice (w 6 against w 4 or 8) is within noise.
