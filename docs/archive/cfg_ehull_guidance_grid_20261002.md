# CFG guidance × E_hull target grid, 2026-10-02

The latest e_hull-conditioned CFG WyFormer was sampled at three target
`energy_above_hull` values and guidance scales from 0 upward. The full
[de novo ranking protocol](../de_novo_ranking_protocol.md) was run on 1,000
genes per arm. The original 3 × 6 grid covered w = 0–5; an adaptive extension
tested higher integer scales until a downturn was observed or the user ended
the sweep. **Thirty arms completed and uploaded.**

The best observed **MetaSUN** rate was **53.0%** at target 0.05 eV/atom,
w = 5. At the same target, w = 11 gave 39.1%: stronger guidance kept raising
gene novelty but lost more valid, metastable structures. No target showed
unbounded MetaSUN improvement with w. SUN peaked at different scales, but its
small counts make those peak locations exploratory.

## Provenance and method

- Evaluation completed **2026-10-02 (Asia/Singapore)** on zeus. Checkpoint
  and all protocol artifacts belong to W&B run
  [`ehull_adamw_wsd_5x_cfg-20260924-223159`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x_cfg-20260924-223159).
  The original grid began at Git commit `e9a6694`; the final sweep report
  records `c360bb35e28ca9fdebc1a892ccc8d3d8b51a68c6`. The orchestration
  scripts were locally modified during evaluation to allow two relaxation
  workers per GPU and automatic continuation; their final versions are
  committed with this report. The per-arm manifests are the record of settings
  actually used.
- Generation used temperature 1, target `energy_above_hull` = 0, 0.05, or
  0.10 eV/atom, and integer guidance scale w. Here w = 0 selects the
  unconditional branch and w = 1 is ordinary conditional sampling. The model's
  conditioning field is the **LeMat-Bulk raw PBE** hull energy described by
  [`lemat_bulk_fmax1_stress.yaml`](../../yamls/datasets/lemat_bulk_fmax1_stress.yaml).
  The relaxed-structure hull used for MetaSUN/SUN is **ORB-v3
  conservative-inf**. They are different energy definitions; the target is a
  sampling input, not an ORB hull measurement.
- Each arm kept the first 1,000 formally valid sampled genes after
  oversampling. Formal validity is valid raw draws divided by attempted raw
  draws in the successful generation call. The other rates below divide by
  the 1,000 kept genes. A failed low-oversampling call could be retried at a
  higher factor and is not included in that formal-validity denominator.
- PyXtal used eight CPU processes; relaxation used the
  `0:1,2:2,*:3` trial schedule, ORB-v3 conservative-inf, CUDA devices 0 and 1,
  up to two workers per GPU, and a 300 s trial timeout. The runner was limited
  to ten CPU threads. Reported structure metrics use the **free** track after
  symmetry release and rattling. The LeMat-Bulk reference defines structural
  novelty. MetaSUN counts novel, unique structures at ORB hull energy ≤ 0.1
  eV/atom; SUN uses ≤ 0 eV/atom. `Metastable` and `stable` in the table apply
  before the novelty filter.
- Per-arm W&B artifacts are named
  `protocol_<run-id>.cfg-grid-e<target>-w<scale>` and include generated genes,
  PyXtal starts, relaxation and structure tables, relaxed CIFs, funnel and
  manifest. The final aggregate artifact is
  `cfg_ehull_high_w_<run-id>` (`tables/grid.csv`, `tables/report.json`, and
  `tables/conclusion.md`). The initial 18-arm aggregate is
  `cfg_ehull_grid_<run-id>`. The local output directory is a working copy of
  the generated and scored outputs in those artifacts.

## Full results

Every row has **N = 1,000** sampled genes. `Gene novel` is gene novelty;
`Free valid` is valid relaxed structure. MetaSUN and SUN include the structural
novelty filter. All values are percentages.

| Target E_hull (eV/atom) | w | Formal valid | Gene novel | Free valid | Metastable | Stable | MetaSUN | SUN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 99.5% | 58.7% | 93.7% | 30.7% | 1.4% | 22.3% | 0.2% |
| 0 | 1 | 99.1% | 54.3% | 93.4% | 58.6% | 6.9% | 26.3% | 1.5% |
| 0 | 2 | 91.7% | 55.7% | 92.5% | 61.9% | 10.3% | 27.0% | 2.4% |
| 0 | 3 | 66.8% | 60.7% | 91.2% | 61.0% | 10.3% | 29.0% | 2.3% |
| 0 | 4 | 36.5% | 62.5% | 88.7% | 58.5% | 8.8% | 24.6% | 2.0% |
| 0 | 5 | 20.1% | 68.7% | 86.0% | 54.1% | 9.4% | 26.5% | 3.8% |
| 0 | 6 | 11.5% | 72.8% | 81.5% | 47.7% | 7.7% | 27.5% | 3.7% |
| 0 | 7 | 7.0% | 76.5% | 72.1% | 40.2% | 6.1% | 23.9% | 3.6% |
| 0.05 | 0 | 99.4% | 58.8% | 93.6% | 26.9% | 0.9% | 19.2% | 0.2% |
| 0.05 | 1 | 99.7% | 60.3% | 92.3% | 59.0% | 2.9% | 34.5% | 1.2% |
| 0.05 | 2 | 98.3% | 61.0% | 91.3% | 67.9% | 3.2% | 42.2% | 1.5% |
| 0.05 | 3 | 91.8% | 61.7% | 91.8% | 70.2% | 5.2% | 43.3% | 1.6% |
| 0.05 | 4 | 78.4% | 73.4% | 89.1% | 67.2% | 3.2% | 49.1% | 1.7% |
| 0.05 | 5 | 61.7% | 74.9% | 88.6% | 67.1% | 2.4% | 53.0% | 1.0% |
| 0.05 | 6 | 46.2% | 77.5% | 85.2% | 61.7% | 3.0% | 48.7% | 1.8% |
| 0.05 | 7 | 32.7% | 80.9% | 84.0% | 61.1% | 3.0% | 49.8% | 1.7% |
| 0.05 | 8 | 23.6% | 84.1% | 79.7% | 56.3% | 3.6% | 48.5% | 2.5% |
| 0.05 | 9 | 16.1% | 84.4% | 77.8% | 55.7% | 3.6% | 47.5% | 3.1% |
| 0.05 | 10 | 11.1% | 83.9% | 76.2% | 54.0% | 3.7% | 45.2% | 2.9% |
| 0.05 | 11 | 7.5% | 87.1% | 68.0% | 46.5% | 4.4% | 39.1% | 3.5% |
| 0.10 | 0 | 99.4% | 59.0% | 94.3% | 31.8% | 1.2% | 21.7% | 0.5% |
| 0.10 | 1 | 99.6% | 56.7% | 93.5% | 46.6% | 2.1% | 28.3% | 0.9% |
| 0.10 | 2 | 99.4% | 55.4% | 91.7% | 51.8% | 1.4% | 31.1% | 0.3% |
| 0.10 | 3 | 97.7% | 56.2% | 92.5% | 49.9% | 1.9% | 32.0% | 0.4% |
| 0.10 | 4 | 94.8% | 62.0% | 93.5% | 50.4% | 1.3% | 33.4% | 0.3% |
| 0.10 | 5 | 89.0% | 63.5% | 90.2% | 47.5% | 1.8% | 31.8% | 0.7% |
| 0.10 | 6 | 83.4% | 63.0% | 89.3% | 46.7% | 2.1% | 31.2% | 0.8% |
| 0.10 | 7 | 76.8% | 67.1% | 84.6% | 43.4% | 1.4% | 31.0% | 1.1% |
| 0.10 | 8 | 71.1% | 64.2% | 82.3% | 41.1% | 1.0% | 29.1% | 0.5% |
| 0.10 | 9 | 66.1% | 64.9% | 80.2% | 40.9% | 1.7% | 29.2% | 1.0% |

## Interpretation and stopping

At target **0.05**, ordinary conditioning (w = 1) yielded 34.5% MetaSUN.
Guidance raised it to 53.0% at w = 5, an observed **+18.5 percentage points**.
Past w = 5, the broad trend reversed: 48.7–49.8% at w = 6–7 and 39.1% by
w = 11. Over w = 5–11, gene novelty rose from 74.9% to 87.1%, while the valid
relaxed-structure rate fell from 88.6% to 68.0% and the metastable rate from
67.1% to 46.5%. In these cohorts, novelty gains no longer compensated for
the fall in valid and metastable outcomes. Formal validity of raw
draws also fell from 61.7% to 7.5%, raising generation cost sharply.

At target **0**, MetaSUN's best observed rate was 29.0% at w = 3; at w = 7 it
was 23.9%. Gene novelty climbed from 60.7% to 76.5% over those settings,
while free-structure validity fell from 91.2% to 72.1%. Formal validity was
only 7.0% at w = 7. At target **0.10**, MetaSUN peaked at 33.4% at w = 4 and
ended at 29.2% at w = 9; formal validity remained higher, 66.1% at w = 9.
Thus the target changes both the attainable MetaSUN rate and how quickly raw
sampling validity collapses.

The best observed SUN arm at each target was w = 5 (3.8%, 38/1,000) for 0,
w = 11 (3.5%, 35/1,000) for 0.05, and w = 7 (1.1%, 11/1,000) for 0.10.
Those are **not** reliable estimates of a SUN optimum: the protocol uses
1,000 genes because it is powered for MetaSUN development, while its [sample
size rationale](../de_novo_ranking_protocol.md#how-many-genes) calls for roughly
10,000 genes to resolve SUN shifts. The 3.8% versus 3.5% comparison differs
by only three observed successes. At target 0.05, SUN had not shown a sustained
downturn by w = 11 even though MetaSUN had.

The adaptive extension defined a downturn as two consecutive arms at w ≥ 6
below the earlier best rate, assessed separately for MetaSUN and SUN. This
rule stopped target 0 at w = 7 and target 0.10 at w = 9. For target 0.05,
MetaSUN met the rule at w = 7 but SUN did not. The user ended the experiment
after the already-running w = 11 arm completed and uploaded; no w = 12 arm was
run. This stopping rule and the choice of the largest observed rates are
exploratory, not statistical tests or corrections for scanning many settings.

For this checkpoint and this **MetaSUN** ranking protocol, target 0.05 and
w = 5 is the best observed setting in the measured grid. There is no single
setting established as best for SUN. The earlier
[classifier-free-guidance study](../classifier_free_guidance.md) concerned a
different CFG checkpoint and stopped its relaxed sweep at w = 3; its numbers
should not be pooled with these cohorts.
