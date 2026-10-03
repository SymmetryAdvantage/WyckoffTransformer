# Okhotin benchmark submission: which generator, which rule of engagement

> **Decision (2026-10-03).** The CFG e_hull generator
> `ehull_adamw_wsd_5x_cfg-20260929-150414` sampled at `energy_above_hull = 0.025`,
> guidance scale 6, under **fire-control keeping the top ~50%** of unique novel genes by
> predicted e_hull (`gene_min_ehull_adamw_wsd-20260929-154926`). On a fresh 2000-gene
> confirmation arm it scored **MSUN 0.7465 [0.727, 0.765] per submitted structure** under
> our ORB proxy, against 0.652 for the same generator under fire-discipline and 0.458 for
> the unconditional generator. The production run is in [Production](#production).

## The benchmark and what we could measure

- **Rules.** Train on alex-mp-20 `train.csv` only; submit 10,000 structures; the
  organisers relax them with an MLIP, judge novelty against alex-mp-20 and stability
  against a hull they have not disclosed.
- **Decision metric:** MSUN per *submitted* structure. A duplicate, a failed DiffCSP++
  start or an invalid relaxed structure counts as a miss.
- **Everything used to make the submission is trained on alex-mp-20 train only.**
  - Generators and the gene e_hull predictor: `production_training: false`, with val
    used only for checkpoint selection.
  - The predictor's gene minima are taken per split (`alex_mp_20_labelled_per_split`).
  - DiffCSP++ GeoV2 (`symmetry-advantage/diffcsp/ua1g6od4`, artifact `:best` = v28,
    epoch 140) was trained on alex-mp-20 train, and GeoV2 does not use ORB features.
- **ORB is evaluation only.** It imitates the organisers' relaxation and hull so that a
  model can be chosen. It never pre-relaxes, ranks or filters the submission.

### Evaluation, identical for every arm

genes → DiffCSP++ GeoV2 (one structure per gene) → CrySPR rattle (0.05 Å, strain 0.01,
seeded per gene) → **one unconstrained ORB-v3 conservative-inf relaxation**
(`--relax-schedule single`, BFGS + FrechetCellFilter, fmax 0.05, ≤1000 steps) → e_hull
on the LeMat-Bulk-MLIP-Hull ORB hull → StructureMatcher uniqueness within the arm and
novelty against alex-mp-20 train+val (`--reference-id-column material_id`).

- **The hull is a proxy.** The LeMat ORB hull is a superset of the alex-mp-20 hull, so it
  is a conservative stand-in for the undisclosed one.
- **The relaxation is a proxy too.** The organisers' MLIP and its settings are unknown.
  Absolute rates will move; the *ordering* of arms is what this study is for.
- **The structures evaluated are the structures submitted:** the same DiffCSP++ output
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
  [0.727, 0.765]**, SUN 0.0465. Its 0 DiffCSP++ failures and 1932 valid structures match
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
- **Pipeline.** DiffCSP++ runs on all 10,400 genes. `assemble_submission.py` walks them in
  predicted-e_hull order, skips any failed or rejected start, rattles each structure with
  its evaluation seed, and stops at 10,000.
- **Output** goes to `$STORE/alex_bench/submission_cfg_e0p025_w6_cut50/`: `cifs/`,
  `structures.extxyz` and `manifest.csv` (gene, predicted e_hull, formula), plus
  `manifest.json` with the expected ORB score of exactly the submitted subset.

**Result (2026-10-04 03:00):**
- DiffCSP++ produced a structure for 10,383 of 10,400 genes. The assembler took the first
  10,000 that passed: 17 genes had no structure and 6 had atoms closer than 0.5 Å.
- **Expected ORB score of exactly the submitted 10,000: MSUN 0.7315 [0.723, 0.740],
  SUN 0.0389 [0.035, 0.043]**, with 9644 valid and 9054 valid novel. All 10,400 scored
  0.732, matching the 0.7465 confirmation arm within its interval.
- Re-checked in a fresh process: 10,000 distinct genes and IDs, and every rattled
  structure passes `check_start`.
- Cells are DiffCSP++'s conventional cells: mean 19.6 atoms, max 87. Check whether the
  benchmark caps cell size (alex-mp-20 itself is ≤ 20 atoms).
- Files: `$STORE/alex_bench/submission_cfg_e0p025_w6_cut50/{cifs/,structures.extxyz,manifest.csv,manifest.json}`.

## Caveats

- **Proxy evaluation:** ORB relaxation and the LeMat ORB hull, not the organisers' MLIP
  and hull.
- **Pool mixing in the 3 × 3:** the fire-discipline cells come from different pools than
  the other two columns of their row (same setting, same distribution).
- **Settings chosen on the evaluation data:** the setting and the cut were picked on the
  same proxy they are scored with. The fresh confirmation arm guards against the
  winner's curse in the cut; the setting choice (w 6 against w 4 or 8) is within noise.
