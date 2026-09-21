# Improving Energy Prediction Precision for Wyckoff Gene Critics

This document analyzes the performance of the Wyckoff gene energy critic in W&B run
[`min_energy_adamw_wsd-20260912-115957`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/min_energy_adamw_wsd-20260912-115957),
diagnoses its architectural and physical limitations, and details systematic improvements
to achieve high-precision energy prediction.

---

## 1. Baseline Experiment Analysis: `min_energy_adamw_wsd-20260912-115957`

* **Date:** 2026-09-12 / 2026-09-13
* **Git commit:** [`aa2f070`](file:///home/users/nus/kna/scratch/WyFormer/worktrees/energy-best/docs/gene_selection_measurements.md) (*"Price gene-level selection, and measure what it actually buys"*)
* **Config:** [`yamls/models/lemat/min_energy_adamw_wsd.yaml`](file:///home/users/nus/kna/scratch/WyFormer/worktrees/energy-best/yamls/models/lemat/min_energy_adamw_wsd.yaml)
* **Dataset:** `lemat_bulk_fmax1_stress` (5,127,342 train / 100,000 val / 100,000 test)
* **Target:** `gene_min_formation_energy_per_atom` regressed with MSE (`scalar_loss: mse`)
* **Hardware:** NVIDIA A100-SXM4-40GB (`asp2a-gpu002`)
* **Summary Metrics:**
  * Train MAE: **0.0589 eV/atom** (final batch MSE: 0.00549 eV²/atom²)
  * Validation MAE: **0.0601 eV/atom** (Best Val MAE: **0.0501 eV/atom** at epoch 2750 during WSD decay)
  * Test MAE: **0.0601 eV/atom**

Comparing this run with its sibling run `gene_min_energy_adamw_wsd-20260912-115824` on the exact same dataset and split, the sibling included `condition_feature: max_force` and achieved **0.0589 eV/atom** Test MAE, demonstrating that even a single physical condition immediately buys precision.

---

## 2. Diagnosis & Identified Bottlenecks

### 2.1 Issue 1: Unweighted Sequence Pooling Violates Intensive Physical Scale
In [`yamls/models/lemat/min_energy_adamw_wsd.yaml`](file:///home/users/nus/kna/scratch/WyFormer/worktrees/energy-best/yamls/models/lemat/min_energy_adamw_wsd.yaml), sequence aggregation is set to:
```yaml
CascadeTransformer_args:
  token_aggregation: mean
  include_start_in_aggregation: true
```
* **Physical contradiction:** Formation energy per atom ($E_{\text{form}} / N_{\text{atoms}}$) is an intensive property. In a crystal structure, each Wyckoff site has a geometric site multiplicity $m_i \in \{1, 2, 3, 4, 6, 8, 12, 16, 24, 48, \dots\}$. A general position with multiplicity 24 represents 24 atoms in the unit cell, whereas a special position with multiplicity 1 represents only 1 atom. Unweighted mean (`token_aggregation: mean`) weights every token equally ($1/N_{\text{sites}}$), effectively giving an atom on a multiplicity 1 site 24 times more influence on per-atom energy than an atom on a multiplicity 24 site.
* **Start token contamination:** Setting `include_start_in_aggregation: true` includes token index 0 (`spacegroup_number`) in the site mean. The space group number represents global symmetry, not an atomic site; blending its embedding into the atomic mean distorts the pooled representation in a sequence-length-dependent manner.
* **The fix:** Include `multiplicity` in the cascade order as a weighting scalar, set `token_aggregation: weighted_mean`, `aggregation_weight: 3` (or the index of `multiplicity`), and `include_start_in_aggregation: false`. This computes the physically correct weighted average $\sum_i m_i \mathbf{h}_i / \sum_i m_i$.

### 2.2 Issue 2: Severe Underparameterization (<50k Parameters for 5.1M Rows)
* Embedding dimensions: `elements: 16`, `site_symmetries: 16`, `sites_enumeration: 8` $\to d_{\text{model}} = 40$.
* Encoder: 3 layers, 4 attention heads (head dimension 10), feedforward dimension 128.
* Total trainable parameters are under 50,000. Across 5.1M diverse materials containing elements from H to U, $d_{\text{model}} = 40$ cannot capture subtle electronic and steric interactions.
* The training curves show zero overfitting (Train MAE 0.0589 vs Val MAE 0.0601), indicating the network is heavily capacity-constrained.

### 2.3 Issue 3: Missing Relaxation Quality / DFT Convergence Conditioning
* Structures in `lemat_bulk_fmax1_stress` have residual forces up to 1.0 eV/Å (and Materials Project calculations often converged on energy criteria with residual forces ~0.086 eV/Å).
* Without `condition_feature: max_force`, the model must fit a single energy target across structures with varying degrees of unfinished relaxation, introducing irreducible label noise.
* Restoring `condition_feature: max_force` (via AdaLN with `log1p` transform and scale 0.05) allows the model to learn $E(\text{gene}, F_{\text{max}})$ and query the clean-relaxation limit ($F_{\text{max}} = 0$) at screening time.

### 2.4 Issue 4: MSE on Observed Minima vs. Censored Likelihood
* A Wyckoff gene fixes the space group and site symmetries, but leaves continuous atomic coordinates and unit cell parameters free, specifying a manifold of possible structures.
* Every entry in an archive is merely an *upper bound* on what the gene can attain ($m(g) = \min(E \mid g)$).
* Regressing observed minima with MSE fits $\mathbb{E}[E \mid g]$, which is biased upward by poor initial relaxations or undersampled manifolds (93.1% of archive genes are singletons).
* The censored model ([`CensoredMinLoss`](file:///home/users/nus/kna/scratch/WyFormer/worktrees/energy-best/src/wyckoff_transformer/censored.py#L40-L100)) fits observations as $E_i = m(g) + \text{Exponential}(s) + \mathcal{N}(0, \sigma)$, recovering the true attainable floor $m(g)$.

### 2.5 Issue 5: Truncated Training Horizon
* The 2,859-epoch cutoff was explicitly documented as a "proof of concept" horizon to fit within a single job link, cutting short the planned 20,000-epoch schedule.
* Under Warmup-Stable-Decay (WSD), the majority of optimization gains occur during the decay phase. Chaining the run to complete 10,000–20,000 epochs allows the optimizer to converge to significantly lower loss.

### 2.6 Issue 6: Range Restriction — Composition Dominates Formation Energy
* As shown in [`docs/gene_selection_measurements.md`](file:///home/users/nus/kna/scratch/WyFormer/worktrees/energy-best/docs/gene_selection_measurements.md), formation energy is dominated by elemental composition.
* When ranking candidate genes on convex hull distance ($E_{\text{hull}}$) or comparing polymorphs of the same composition, subtracting the composition hull cancels the elemental baseline, leaving an error floor of ~0.12–0.20 eV/atom.
* **Why not train on hull-derived targets?** `energy_above_hull` and `delta_e_polymorph` are *database-dependent quantities* — they shift whenever an entry is added to or removed from the convex hull. Formation energy per atom is a genuine physical property of a structure, invariant to the state of any database. Training on hull distance would couple the model to a snapshot of the database rather than to physics, making predictions non-portable and semantically fragile.
* **Implication:** The model should continue to predict formation energy (a physical quantity), but downstream evaluation and screening should subtract composition-level baselines (e.g. hull interpolation) to isolate the structural ranking signal. Reducing the 0.06 eV/atom MAE on formation energy is the path to sharper hull ranking, not changing the target.

---

## 3. Systematic Roadmap for Precision Improvement

| Priority | Area | Action | Expected Gain |
|---|---|---|---|
| **P0** | **Physical Pooling** | Use `token_aggregation: weighted_mean`, `aggregation_weight: multiplicity`, `include_start_in_aggregation: false` | Eliminates multiplicity weighting distortion across sites |
| **P0** | **Convergence Conditioning** | Add `condition_feature: max_force` (`log1p`, scale 0.05); query at 0 force | Filters incomplete DFT relaxation noise (~1.2 meV MAE reduction demonstrated) |
| **P1** | **Model Capacity** | Scale $d_{\text{model}}$ to $\ge 96$–128 (embeddings 32/32/16), 4–6 layers, FFN 256–512 | Resolves capacity bottleneck on 5.1M dataset |
| **P1** | **Training Horizon** | Run full WSD schedule (10,000–20,000 epochs) across chained PBS jobs | Allows proper convergence during WSD decay |
| **P2** | **Relational Bias** | Enable [`RelationalAttentionBias`](file:///home/users/nus/kna/scratch/WyFormer/worktrees/energy-best/src/wyckoff_transformer/cascade/relational.py#L30-L60) (electronegativity differences, radius ratios) | Direct chemical bonding priors in attention logits |
| **P2** | **Censored Loss** | Use `scalar_loss: censored` to predict $\min(E \mid g)$ and excess scale $s(g)$ | Unbiases predictions from upper-bound DFT sampling |
| **P3** | **Ensembling** | Train 5–10 seeds (Wren protocol) | Reduces epistemic error $\tau(g)$ and provides calibrated uncertainty |
