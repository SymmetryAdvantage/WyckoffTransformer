# De Novo Ranking Protocol: `ehull_adamw_wsd_5x-20260912-115321` at $e_{\text{hull}} = 0.05$

Evaluation of [`ehull_adamw_wsd_5x-20260912-115321`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x-20260912-115321) under the de novo ranking protocol ([`docs/de_novo_ranking_protocol.md`](file:///home/kna/WyckoffTransformer/docs/de_novo_ranking_protocol.md)) sampled at target **$e_{\text{hull}} = 0.05$ eV/atom** rather than the training default $e_{\text{hull}} = 0.0$.

- **Date**: 2026-09-21
- **Git Commit**: `b4b88e61749c6d80d1ae9ea9cf6bc1bc8123db93`
- **Platform**: `iapetus` (6 CPU cores, 2x Tesla K20c + 1x GeForce GTX 750 Ti)
- **WanDB Run**: [`ehull_adamw_wsd_5x-20260912-115321`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x-20260912-115321) (Run summary was preserved at baseline $e_{\text{hull}} = 0.0$)
- **WanDB Artifacts**:
  - Dedicated target artifact: `protocol_ehull_adamw_wsd_5x-20260912-115321_ehull005:latest` (`:v0`, aliases: `E_hull=0.05`, `ehull005`, `ehull_0.05`, `target_0.05`)
  - Canonical collection version: `protocol_ehull_adamw_wsd_5x-20260912-115321:v3` (aliases: `E_hull=0.05`, `ehull005`, `ehull_0.05`, `target_0.05`)
  - Baseline comparison artifact: `protocol_ehull_adamw_wsd_5x-20260912-115321:v2` (`:latest`, target $e_{\text{hull}} = 0.0$)
- **Outputs on Disk**: [`generated/ehull_adamw_wsd_5x-20260912-115321/protocol/`](file:///home/kna/WyckoffTransformer/generated/ehull_adamw_wsd_5x-20260912-115321/protocol/)

---

## Executive Summary

Sampling the $e_{\text{hull}}$-conditioned model at $e_{\text{hull}} = 0.05$ eV/atom rather than $e_{\text{hull}} = 0.0$ leads to a substantial increase in MetaSUN yield:
- **Free track MetaSUN** rises from **0.236** to **0.360** (+12.4 percentage points, a **+52.5% relative gain**).
- **Fixed-symmetry MetaSUN** rises from **0.163** to **0.243** (+8.0 percentage points, a **+49.1% relative gain**).
- Total metastability ($e_{\text{hull}} \le 0.1$ eV/atom) increases from **55.6%** to **59.2%** in the free track.
- Gene novelty rises from **55.6%** to **60.6%**, and relaxed structure novelty from **59.0%** to **63.0%**.

Conditioning on $e_{\text{hull}} = 0.0$ over-constrains generation to well-explored, crowded regions in LeMat-Bulk where the hull is heavily occupied. Relaxing the target to $0.05$ eV/atom expands exploration into novel compositions and geometries while keeping energies safely below the 0.1 eV/atom metastability threshold.

### Caveat: the baseline was scored on different hardware

The $e_{\text{hull}} = 0.0$ baseline (`:v2`) was run on **aspire2a** (4 GPUs,
device budgets 34376, 60 PyXtal cores); this run was on **iapetus**. The MLIP
checkpoint, hull revision, novelty reference, `fmax`, tolerance factor and trial
schedule are identical, but on the most matched population available
(known gene **and** zero positional DoF, where the energy is close to a pure
calculator readout) iapetus reads ~0.025–0.030 eV/atom *higher* than aspire2a.

That offset works **against** this run, so the conclusion is conservative: the
+0.124 MetaSUN gain would widen, not shrink, if both arms were scored on one
machine. Two qualifications follow:

- The generation-stage gains are unaffected by hardware — gene novelty
  ($0.556 \to 0.606$) is computed from the generator's output against the
  fingerprint reference with no MLIP involved.
- The apparent *loss* of stable/SUN structures ($0.064 \to 0.027$, $0.012 \to
  0.009$) is a threshold-at-zero statistic and is therefore the quantity most
  sensitive to the offset. Do not read it as established until one cohort is
  re-relaxed on the other machine.

Full three-arm comparison against the filtered unconditional model, and what the
sweep implies for the conditioning strategy, in
[negative_data_strategy.md](negative_data_strategy.md).

---

## The Funnel Comparison

All rates are per sampled gene (denominator: 1000).

| Cascade Stage | Metric | Target $e_{\text{hull}} = 0.05$ | Target $e_{\text{hull}} = 0.0$ (Baseline) | Absolute $\Delta$ |
| :--- | :--- | :--- | :--- | :--- |
| **Gene Screen** | Valid gene | 1000 (1.000) | 1000 (1.000) | 0.000 |
| | Unique gene | 998 (0.998) | 999 (0.999) | -0.001 |
| | Gene novel | 605 (0.606) | 555 (0.556) | **+0.050** |
| | Gene known | 393 (0.394) | 444 (0.445) | -0.051 |
| **Fixed Symmetry** | Produced structure | 994 (0.996) | 996 (0.997) | -0.001 |
| *(pre-rattling)* | Valid structure | 915 (0.917) | 938 (0.939) | -0.022 |
| | Unique structure | 915 (0.917) | 938 (0.939) | -0.022 |
| | Novel structure | 604 (0.605) | 580 (0.581) | **+0.024** |
| | Metastable ($e_{\text{hull}} \le 0.1$) | 476 (0.477) | 483 (0.483) | -0.006 |
| | **MetaSUN** ($e_{\text{hull}} \le 0.1$, novel) | **243 (0.243)** | **163 (0.163)** | **+0.080** |
| | Stable ($e_{\text{hull}} \le 0.0$) | 22 (0.022) | 58 (0.058) | -0.036 |
| | **SUN** ($e_{\text{hull}} \le 0.0$, novel) | **6 (0.006)** | **9 (0.009)** | -0.003 |
| **Free Track** | Produced structure | 994 (0.996) | 996 (0.997) | -0.001 |
| *(post-rattling)* | Valid structure | 915 (0.917) | 939 (0.940) | -0.023 |
| | Unique structure | 915 (0.917) | 939 (0.940) | -0.023 |
| | Novel structure | 629 (0.630) | 589 (0.590) | **+0.040** |
| | Metastable ($e_{\text{hull}} \le 0.1$) | 590 (0.592) | 556 (0.556) | **+0.036** |
| | **MetaSUN** ($e_{\text{hull}} \le 0.1$, novel) | **359 (0.360)** | **236 (0.236)** | **+0.124** |
| | Stable ($e_{\text{hull}} \le 0.0$) | 27 (0.027) | 64 (0.064) | -0.037 |
| | **SUN** ($e_{\text{hull}} \le 0.0$, novel) | **9 (0.009)** | **12 (0.012)** | -0.003 |

---

## Energy Distribution

Energy above the ORB convex hull evaluated across valid structures:

| Metric | Free Track (Post-Rattling) | Fixed Symmetry (Pre-Rattling) |
| :--- | :--- | :--- |
| **Mean $e_{\text{above\_hull}}$** | 0.0999 eV/atom | 0.1565 eV/atom |
| **Median $e_{\text{above\_hull}}$** | 0.0700 eV/atom | 0.0952 eV/atom |
| **Standard Deviation** | 0.1117 eV/atom | 0.2077 eV/atom |
| **Minimum** | -0.0239 eV/atom | -0.0239 eV/atom |
| **Maximum** | 1.5725 eV/atom | 2.4164 eV/atom |
| **Fraction $\le 0.0$ eV/atom** | 3.0% (27 structures) | 2.4% (22 structures) |
| **Fraction $\le 0.05$ eV/atom** | 36.9% (338 structures) | 29.3% (268 structures) |
| **Fraction $\le 0.10$ eV/atom** | 64.5% (590 structures) | 52.0% (476 structures) |

Post-rattling relaxation shifts the median energy from 0.0952 down to 0.0700 eV/atom, pulling an extra 114 structures (12.5% of valid structures) into the $\le 0.1$ eV/atom metastable window.

---

## Breakdown by Positional Degrees of Freedom

Positional degrees of freedom ($\sum \text{dof}$ over Wyckoff sites) control how many unconstrained coordinate dimensions PyXtal must solve during structure generation. Rates are per sampled gene within each bin:

| $\sum \text{dof}$ | Sampled Genes | Valid Structure | Valid Rate | Novel Structure | Novelty Rate | Metastable | MetaSUN | MetaSUN Rate | SUN | SUN Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 176 | 162 | 0.920 | 48 | 0.273 | 114 | 22 | **0.125** | 1 | 0.0057 |
| **1–2** | 208 | 191 | 0.918 | 104 | 0.500 | 130 | 55 | **0.264** | 4 | 0.0192 |
| **3–5** | 301 | 279 | 0.927 | 216 | 0.718 | 197 | 130 | **0.432** | 3 | 0.0100 |
| **6–10** | 175 | 157 | 0.897 | 140 | 0.800 | 100 | 76 | **0.434** | 0 | 0.0000 |
| **>10** | 138 | 126 | 0.913 | 121 | 0.877 | 83 | 76 | **0.551** | 1 | 0.0072 |
| **Total** | **998** | **915** | **0.917** | **629** | **0.630** | **590** | **359** | **0.360** | **9** | **0.0090** |

Key trends:
1. **Validity is complexity-invariant**: Structure validity remains steady between 89.7% and 92.7% across all DoF bins.
2. **Novelty scales steeply with DoF**: From 27.3% at 0 DoF to 87.7% at >10 DoF.
3. **MetaSUN yield peaks in complex cells**: Genes with $\ge 3$ DoF achieve MetaSUN rates exceeding 43%, reaching 55.1% for $>10$ DoF genes.

---

## Novelty Transitions

Novelty is evaluated twice: first on the sampled gene fingerprint, and second on the relaxed structure fingerprint detected post-rattling.

- **`gene_known_became_novel`**: 112 structures (11.2% per sampled gene). These were drawn from space groups and compositions already in LeMat-Bulk, but escaped to novel geometries or broken-symmetry configurations after relaxation and rattling.
- **`gene_novel_became_known`**: 25 structures (2.5% per sampled gene). PyXtal placed these on unknown fingerprints, but they converged into known reference configurations during relaxation.
- **`relaxed_fingerprint_changed`**: 446 structures (44.8%). Symmetry release and rattling moved almost half of the cohort off their initial Wyckoff orbit placements.

---

## Protocol Verification & Lineage

Hardware and trial metrics from [`manifest.json`](file:///home/kna/WyckoffTransformer/generated/ehull_adamw_wsd_5x-20260912-115321/protocol/manifest.json):
- **Cohort Generation**: 1150 attempted, 1138 formally valid (98.96% validity); first 1000 kept.
- **PyXtal Drawing**: 2434 trials drawn across 6 CPU cores (`--pyxtal-cores 6`, 2.439 trials/gene). 0 failed, 0 timed out.
- **Relaxation Workers**: 5 worker slots across 3 GPUs:
  - `cuda:0` (Tesla K20c): 896 trials relaxed
  - `cuda:1` (Tesla K20c): 873 trials relaxed
  - `cuda:2` (GeForce GTX 750 Ti): 665 trials relaxed
  - 2419 trials relaxed successfully, 15 failed (due to large cells / memory bounds).
- **Novelty Reference**: Joint LeMat-Bulk `cache/lemat_bulk_fmax1_stress`, train+val+test (4,826,004 reference fingerprints).
- **ORB Calculator**: Custom split CPU-neighbour / GPU-forward implementation (`orb_conserv_inf`, `fmax = 0.05`).
- **WanDB Logging**: Artifact uploaded with explicit metadata `target="E_hull=0.05"`, `condition="energy_above_hull=0.05"`. Run summary was verified unchanged via API post-upload.
