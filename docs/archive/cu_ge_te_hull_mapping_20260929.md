# Convex Hull Mapping of the Cu-Ge-Te Chemical System

> **STATUS: MEASUREMENT, 2026-09-29.**
> Code at `54c955b` (branch `main`, includes `perf/nep89-relaxation`).
> Orchestrator: `scripts/map_cu_ge_te_hull.py`. Publication plot: `scripts/plot_publication_hull.py`.
> W&B Run: [`pglsoqms`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/pglsoqms) (`cu_ge_te_hull_mapping_20260929`).
> W&B Artifact: [`cu_ge_te_hull_campaign:v0`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/artifacts/hull_mapping_results/cu_ge_te_hull_campaign/v0).

---

## 1. Overview and Objectives

This study performs an end-to-end de novo mapping of the convex hull for the ternary copper-germanium-tellurium ($\text{Cu-Ge-Te}$) system, including all constituent binary subsystems ($\text{Cu-Ge}$, $\text{Ge-Te}$, $\text{Cu-Te}$) and the ternary space.

The primary goals were:
1. **Unbiased Generative Exploration:** Sample 100,000 candidate Wyckoff representations across all compositions using our best chemical-system conditioned generative model (`chemsys_sg_uncond_adanmw_wsd-20260925-030937`).
2. **Critic-Guided Screening:** Score unique deduplicated genes with our state-of-the-art formation energy regressor (`min_energy_adamw_wsd-20260924-102431`) to identify the top 10,000 most thermodynamically promising genes.
3. **High-Throughput Reconstructive Relaxation:** Reconstruct 50 PyXtal structures per gene (500,000 attempts), pre-relax and screen with NEP 89 under symmetry constraints on 20 CPU cores, select up to 5 lowest-energy polymorphs per gene, and perform full cell and position MLIP relaxation using ORB-v3 (`orb_conserv_inf`) across 4 GPU workers.
4. **Thermodynamic Hull Resolution:** Merge relaxed structures with 42 published ORB reference phases, identify stable ground states (SUN) and metastable phases (MetaSUN), and construct publication-quality ternary phase diagrams and pseudo-binary cuts.
5. **Experimental Validation:** Query the de novo dataset for the phase reported in *Chem. Mater.* 2016, 28, 9, 3111–3118 (`acs.chemmater.6c00801.pdf`), $\text{Cu}_5\text{Ge}_2\text{Te}_7$, to determine its thermodynamic distance to the convex hull and explain the non-equilibrium synthesis requirements.

---

## 2. Methodology & Execution Funnel

```
+------------------------------------------------------------------------+
| 1. Generative Backbone Sampling                                       |
|    Model: chemsys_sg_uncond_adanmw_wsd-20260925-030937                 |
|    100,000 draws -> 98,538 valid Wyckoff genes (98.5% valid)          |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 2. Deduplication                                                       |
|    Canonical 128-bit augmented Wyckoff gene hash keys                  |
|    29,659 unique composition-spacegroup Wyckoff genes                  |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 3. Energy Regressor Scoring                                            |
|    Model: min_energy_adamw_wsd-20260924-102431                         |
|    Top 10,000 genes selected (Delta E_hull,pred in [-0.155, 0.122] eV) |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 4a. PyXtal Structure Realisation (20 CPU cores)                        |
|    50 random trial geometries per gene (500,000 trials total)          |
|    489,371 valid initial structures produced (97.9% success rate)      |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 4b. NEP 89 Pre-relaxation & Selection (20 CPU cores)                   |
|    489,371 structures pre-relaxed with NEP 89 (fast BFGS, FixSymmetry) |
|    Select up to 5 lowest-energy distinct polymorphs per gene           |
|    48,660 candidate structures retained                                |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 4c. ORB-v3 MLIP Relaxation (4 GPU workers on 2x RTX 6000 Ada)          |
|    Model: orb_conserv_inf (cell + atomic positions, fmax=0.01 eV/A)    |
|    48,629 successfully relaxed structures (99.94% convergence rate)    |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 5. Convex Hull Construction (pymatgen PhaseDiagram)                    |
|    48,629 de novo relaxed structures + 42 published ORB references     |
|    Total entries: 48,671                                               |
+------------------------------------------------------------------------+
```

### Computational Performance & NEP 89 Acceleration
Mid-campaign, the branch `perf/nep89-relaxation` was merged into `main` (`d644904`, `e9a6694`). It eliminated redundant `FixSymmetry` constraint copies and introduced a positive-definite BFGS solve with line-search clipping:
- **Pre-relaxation throughput:** Increased from ~0.67 structures/sec/core to ~2.22 structures/sec/core.
- **Speedup:** **3.3x wall-clock acceleration**, enabling 489k structure relaxations to complete comfortably within the compute budget on `zeus`.

---

## 3. Convex Hull Results

### Phase Diagram Summary
- **Total entries evaluated:** 48,671
- **Ground-state hull vertices:** 9 phases
  - 4 Elemental reference vertices: $\text{Cu}$ ($Fm\bar{3}m$), $\text{Ge}$ ($Fd\bar{3}m$), $\text{Te}$ ($P3_1 21$)
  - 5 De novo discovered ground-state compounds (SUN phases):
    1. $\text{Cu}_3\text{GeTe}_4$ ($P\bar{4}2m$, space group 111, $E_f = -0.1271$ eV/atom)
    2. $\text{GeTe}$ ($R3m$, space group 160, $E_f = -0.1260$ eV/atom)
    3. $\text{CuTe}$ ($Pmmn$, space group 59, $E_f = -0.0792$ eV/atom)
    4. $\text{Cu}_3\text{Te}_2$ ($P\bar{4}2_1m$, space group 113, $E_f = -0.0784$ eV/atom)
    5. $\text{Cu}_5\text{Ge}$ ($P6_3/mmc$, space group 194, $E_f = -0.0223$ eV/atom)

### Metastable Landscape (MetaSUN)
- **$e_\text{above\_hull} \le 50\text{ meV/atom}$:** 3,979 distinct relaxed polymorphs.
- **$e_\text{above\_hull} \le 100\text{ meV/atom}$:** 17,781 distinct relaxed polymorphs.

This demonstrates rich structural polymorphism in the ternary and pseudobinary telluride systems, with numerous layered and defect-ordered tetrahedral phases lying closely above the ground-state convex hull.

---

## 4. Analysis of the Experimental Material from *Chem. Mater.* 2016 (`acs.chemmater.6c00801.pdf`)

The material described in *Chem. Mater.* 2016, 28, 9, 3111–3118 is monoclinic $\text{Cu}_5\text{Ge}_2\text{Te}_7$ (space group $C2$, No. 5).

### De Novo Search Results:
- In our generated and relaxed dataset, **85 distinct relaxed structures** of $\text{Cu}_5\text{Ge}_2\text{Te}_7$ were discovered.
- **Lowest-energy relaxed polymorph:**
  - Structure ID: `gen_8667_trial_18` (`Cu15Ge6Te21_kept.cif`)
  - Space Group: $C2$ (No. 5), cell parameters matching the experimental monoclinic lattice
  - Total Energy per atom: $-3.4357\text{ eV/atom}$
  - Formation Energy $E_f$: $-0.0373\text{ eV/atom}$
  - Energy above convex hull $e_\text{above\_hull}$: **$89.7\text{ meV/atom}$**
- **Thermodynamic Decomposition:**
  At $0\text{ K}$, the phase lies $89.7\text{ meV/atom}$ above the convex hull tie-triangle spanned by:
  $$\text{Cu}_5\text{Ge}_2\text{Te}_7 \longrightarrow 0.5\,\text{Cu}_3\text{GeTe}_4 + 3.5\,\text{CuTe} + 1.5\,\text{GeTe}$$

### Physical and Experimental Implications:
In the cited paper, $\text{Cu}_5\text{Ge}_2\text{Te}_7$ was synthesized using **Direct Joule Heating (DJH)**, an ultra-rapid non-equilibrium synthesis technique ($>100\text{ K/s}$ ramp followed by rapid quenching).
Our finding that $\text{Cu}_5\text{Ge}_2\text{Te}_7$ lies $89.7\text{ meV/atom}$ above the ground-state convex hull provides an immediate physical explanation:
- Standard high-temperature equilibrium solid-state annealing causes decomposition into the stable ternary $\text{Cu}_3\text{GeTe}_4$ and binary tellurides.
- Rapid Joule flash heating bypasses competitive nucleation of the thermodynamic ground-state phases, and rapid quenching kinetically traps the metastable monoclinic $C2$ network.

---

## 5. Artifacts and Provenance

All campaign outputs are tracked in W&B and backed up in the local project store:

| Item | Local Path / Reference | Description |
|---|---|---|
| **W&B Run** | [`pglsoqms`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/pglsoqms) | Logged metrics, tags, parameters, and interactive plots |
| **W&B Artifact** | `cu_ge_te_hull_campaign:v0` | Full versioned dataset bundle (21 files, ~70 MB) |
| **Publication Figure** | `artifacts/cu_ge_te_hull/cu_ge_te_hull.png` | 2-panel figure: Gibbs ternary triangle + pseudobinary cut |
| **Hull Database** | `artifacts/cu_ge_te_hull/cu_ge_te_hull_entries.csv` | 48,671 entries with energies, formulas, and hull distances |
| **Selected Genes** | `artifacts/cu_ge_te_hull/selected_10000_genes.json.gz` | Top 10k Wyckoff genes chosen by the energy critic |
| **NEP 89 Pre-screen** | `artifacts/cu_ge_te_hull/reconstruction/prescreen_selection.csv` | 48,660 candidate structures selected after NEP 89 |
| **ORB Relaxations** | `artifacts/cu_ge_te_hull/reconstruction/relaxations.csv` | 48,629 ORB MLIP relaxations |
| **All Relaxed CIFs** | `artifacts/cu_ge_te_hull/reconstruction/cifs.tar.gz` | Archive of all 48,629 relaxed CIF structures |
| **Key Ground-State CIFs**| `artifacts/cu_ge_te_hull/key_phases/` | Ground-state CIFs for $\text{Cu}_3\text{GeTe}_4$, $\text{GeTe}$, $\text{CuTe}$, $\text{Cu}_3\text{Te}_2$, $\text{Cu}_5\text{Ge}$, and $\text{Cu}_5\text{Ge}_2\text{Te}_7$ |

### Reproduction
To inspect the entries or plot:
```bash
# Plotting
.venv/bin/python scripts/plot_publication_hull.py

# Download W&B artifact
.venv/bin/python -c "
import wandb
run = wandb.init(dir='~/.local/share/wyformer', project='WyckoffTransformer')
artifact = run.use_artifact('symmetry-advantage/WyckoffTransformer/cu_ge_te_hull_campaign:latest')
artifact_dir = artifact.download()
print('Downloaded to:', artifact_dir)
"
```
