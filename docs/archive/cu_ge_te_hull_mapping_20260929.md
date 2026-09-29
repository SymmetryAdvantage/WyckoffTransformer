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

### Phase Diagram & Ground-State Vertices (SUN Phases)
- **Total entries evaluated:** 48,671 (42 published ORB reference entries + 48,629 de novo relaxed candidates).
- **Ground-state hull vertices:** 9 phases defining the 0 K convex hull:
  - 4 Elemental reference vertices: $\text{Cu}$ ($Fm\bar{3}m$), $\text{Ge}$ ($Fd\bar{3}m$), $\text{Te}$ ($P3_1 21$)
  - 5 De novo discovered ground-state compounds (SUN phases):
    1. $\text{Cu}_3\text{GeTe}_4$ ($P\bar{4}2m$, space group 111, $E_f = -0.1271$ eV/atom)
    2. $\text{GeTe}$ ($R3m$, space group 160, $E_f = -0.1260$ eV/atom)
    3. $\text{CuTe}$ ($Pmmn$, space group 59, $E_f = -0.0792$ eV/atom)
    4. $\text{Cu}_3\text{Te}_2$ ($P\bar{4}2_1m$, space group 113, $E_f = -0.0784$ eV/atom)
    5. $\text{Cu}_5\text{Ge}$ ($P6_3/mmc$, space group 194, $E_f = -0.0223$ eV/atom)

#### Ground-State Stability & Polymorph Findings:
1. **Ternary Stoichiometry Stability:** No new ternary compositions entered the ground-state convex hull. $\text{Cu}_3\text{GeTe}_4$ remains the **sole stable ternary composition** in both DFT and ORB. All other ternary phases (including $\text{Cu}_2\text{GeTe}_3$ and the experimental $\text{Cu}_5\text{Ge}_2\text{Te}_7$) are metastable.
2. **Binary Ground-State Discovery:** In the Cu-Ge binary system, hexagonal $\text{Cu}_5\text{Ge}$ ($P6_3/mmc$, space group 194, $E_f = -0.0223\text{ eV/atom}$) emerged as a **new ground-state vertex on the ORB convex hull**, displacing $\text{Cu}_8\text{Ge}$ and $\text{Cu}_3\text{Ge}$ from the published reference hull. (In LeMat-Bulk DFT, $\text{Cu}_5\text{Ge}$ only had monoclinic/orthorhombic entries with positive formation energy).
3. **Hull Lowering via De Novo Polymorphs:** For all four previously known ground-state compositions, our generated structures uncovered lower-energy polymorphs that lowered the published reference hull envelope:
   - $\text{Cu}_3\text{GeTe}_4$: lowered by **$2.9\text{ meV/atom}$** (from $-3.6488$ to $-3.6517\text{ eV/atom}$)
   - $\text{GeTe}$: lowered by **$2.7\text{ meV/atom}$** (from $-3.9280$ to $-3.9307\text{ eV/atom}$)
   - $\text{CuTe}$: lowered by **$11.0\text{ meV/atom}$** (from $-3.4994$ to $-3.5104\text{ eV/atom}$)
   - $\text{Cu}_3\text{Te}_2$: lowered by **$12.3\text{ meV/atom}$** (from $-3.5595$ to $-3.5718\text{ eV/atom}$)

### Metastable Landscape (MetaSUN) & Discovery Breadth
- **$e_\text{above\_hull} \le 50\text{ meV/atom}$:** 3,932 generated structures (3,979 total including reference).
- **$e_\text{above\_hull} \le 100\text{ meV/atom}$:** 17,734 generated structures (17,781 total including reference) spanning **834 distinct compositions**.

#### Compositional Breakdown of Generated Metastable Phases:
| Subsystem | Total Generated | Metastable $\le 50\text{ meV}$ | Metastable $\le 100\text{ meV}$ | Distinct Metastable Formulas |
|---|---|---|---|---|
| **Ternary ($\text{Cu-Ge-Te}$)** | 22,014 | 65 (0.3%) | 2,580 (11.7%) | **344** |
| **Binary ($\text{Cu-Ge}, \text{Ge-Te}, \text{Cu-Te}$)** | 26,610 | 3,862 (14.5%) | 15,149 (56.9%) | **489** |
| **Elemental ($\text{Cu}, \text{Ge}, \text{Te}$)** | 5 | 5 (100.0%) | 5 (100.0%) | 1 |
| **Total Campaign** | **48,629** | **3,932** (8.1%) | **17,734** (36.5%) | **834** |

#### Novelty Relative to Historical Databases (LeMat-Bulk):
- **Ternary Space Expansion:** LeMat-Bulk contained only **21 metastable ternary structures** across **9 compositions**. Our campaign discovered **2,580 metastable ternary structures** spanning **344 distinct compositions**—a **>120× expansion** in ternary metastable structures.
- **Brand-New Ternary Stoichiometries:** Of the 344 ternary compositions discovered within the $\le 100\text{ meV/atom}$ window, **324 compositions (accounting for 1,595 structures)** are completely new stoichiometries that never existed in LeMat-Bulk (e.g., $\text{CuGeTe}_2$, $\text{CuGe}_2\text{Te}_5$, $\text{Cu}_4\text{GeTe}_4$, $\text{Cu}_5\text{Ge}_2\text{Te}_7$).
- **Full System Expansion:** Relative to LeMat-Bulk’s historical total of 195 metastable entries in $\text{Cu-Ge-Te}$, the campaign’s 17,734 entries represent a **>90× expansion** of the accessible metastable crystal landscape.

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

## 5. Energy Provenance & Correspondence with PBE DFT (LeMat-Bulk)

### Energy Provenance Across Pipeline Stages
The study utilizes multiple energy models according to the project's multi-fidelity screening protocol:
1. **Generative Backbone (`chemsys_sg_uncond_adanmw_wsd-20260925-030937`):** Unconditioned on energy; guided solely by composition constraints and empirical space group priors.
2. **Regressor Critic (`min_energy_adamw_wsd-20260924-102431`):** Trained on **LeMat-Bulk PBE DFT** ground-state formation energies (`gene_min_formation_energy_per_atom` / `energy_above_hull`). It scored all 29.6k unique genes on the DFT scale to select the top 10k candidates.
3. **Pre-relaxation (`NEP 89`):** Empirical machine-learned interatomic potential used to relax 489k random PyXtal geometric starts under fixed symmetry on 20 CPU cores, filtering out unphysical packings.
4. **Final Relaxation & Convex Hull Mapping (`ORB-v3`, `orb_conserv_inf`):** All 48,629 retained candidate structures underwent full cell and coordinate relaxation using ORB-v3. The convex hull, formation energies ($E_f$), and distance to the hull ($e_\text{above\_hull}$) were calculated against the published LeMat-Bulk ORB-v3 hull (`LeMaterial/LeMat-Bulk-MLIP-Hull`).

Per project standards (`docs/energy_fields.md` and `AGENTS.md`), MLIP energies are never mixed across potentials because each model carries its own reference zero. Every MLIP evaluation is strictly *self-consistent* against its own published hull.

### Comparison with LeMat-Bulk PBE DFT
In the primary reference dataset [`lemat_bulk_fmax1_stress`](../../yamls/datasets/lemat_bulk_fmax1_stress.yaml), there are **1,265 Cu-Ge-Te structures** (69 ternary phases, 1,196 binary/elemental entries) calculated with **PBE DFT** (Materials Project, Alexandria, and OQMD settings).

#### Baseline LeMat-Bulk (`lemat_bulk_fmax1_stress`) Distribution:
| Category | Total Entries | Stable ($e_\text{hull} \le 1\text{ meV}$) | Metastable ($e_\text{hull} \le 50\text{ meV}$) | Metastable ($e_\text{hull} \le 100\text{ meV}$) | Unstable ($> 100\text{ meV}$) |
|---|---|---|---|---|---|
| **Ternary only ($\text{Cu-Ge-Te}$)** | **69** | 1 (1.4%) | **8** (11.6%) | **21** (30.4%) | 47 (68.1%) |
| **Binary ($\text{Cu-Ge}, \text{Ge-Te}, \text{Cu-Te}$)** | **956** | 5 (0.5%) | **59** (6.2%) | **147** (15.4%) | 804 (84.1%) |
| **Elemental ($\text{Cu}, \text{Ge}, \text{Te}$)** | **240** | 5 (2.1%) | **20** (8.3%) | **27** (11.2%) | 208 (86.7%) |
| **Full System Total** | **1,265** | 11 (0.9%) | **87** (6.9%) | **195** (15.4%) | 1,059 (83.7%) |

- The 69 ternary structures in LeMat-Bulk cover only **25 distinct stoichiometries** in total.
- The single stable ternary phase is $\text{Cu}_3\text{GeTe}_4$ (`agm003555275`, $E_f = -0.1007\text{ eV/atom}$).
- All 8 ternary structures with $e_\text{hull} \le 50\text{ meV/atom}$ are polymorphs of a single composition: $\text{Cu}_2\text{GeTe}_3$ ($e_\text{hull} \in [3.0, 20.1]\text{ meV/atom}$).

In the published LeMat-Bulk MLIP hull benchmark dataset (`LeMaterial/LeMat-Bulk-MLIP-Hull`), **46 structures** in this exact chemical system have **both** their PBE DFT energy (`true_energy`) and their ORB-v3 energy (`orb_conserv_inf_energy`) calculated on the exact same atomic configurations.

#### 1. Absolute Energy Scale vs Relative Agreement
- **Absolute scale offset:** ORB-v3 was trained on `OMat24` (`PBE_OMat24`), which uses different pseudopotentials and isolated-atom reference choices than Materials Project PBE. This introduces a systematic mean offset of **$+0.094\text{ eV/atom}$** (with pure elemental $\text{Cu}$ shifted by $\approx 0.35\text{ eV/atom}$).
- **Relative rank correlation:** The Pearson correlation between DFT and ORB per-atom energies is **$r = 0.926$**, confirming that relative energetic ordering is strongly preserved across the ternary landscape.

#### 2. Convex Hull Vertices (Ground States)
Constructing the 0 K convex hull from the exact same structures using pure DFT vs pure ORB yields remarkable topological agreement:

| Composition | DFT $E_f$ (eV/atom) | DFT $e_\text{hull}$ (meV/atom) | ORB $E_f$ (eV/atom) | ORB $e_\text{hull}$ (meV/atom) | Agreement / Note |
|---|---|---|---|---|---|
| **$\text{Cu}$** | $0.0000$ | $0.0$ | $0.0000$ | $0.0$ | Ground state in both |
| **$\text{Ge}$** | $0.0000$ | $0.0$ | $0.0000$ | $0.0$ | Ground state in both |
| **$\text{Te}$** | $0.0000$ | $0.0$ | $0.0000$ | $0.0$ | Ground state in both |
| **$\text{Cu}_3\text{GeTe}_4$** | $-0.1007$ | **$0.0$** | $-0.1242$ | **$0.0$** | **Identical ground state:** Sole stable ternary compound on both hulls |
| **$\text{GeTe}$** | $-0.0914$ | **$0.0$** | $-0.1233$ | **$0.0$** | **Identical ground state:** Primary binary sink in both |
| **$\text{CuTe}$** | $-0.0715$ | **$0.0$** | $-0.0681$ | **$0.0$** | **Identical ground state:** $\Delta E_f = 3.4\text{ meV/atom}$ |
| **$\text{Cu}_3\text{Ge}$** | $-0.0061$ | **$0.0$** | $-0.0084$ | **$0.0$** | **Identical ground state:** $\Delta E_f = 2.3\text{ meV/atom}$ |
| **$\text{Cu}_2\text{Te}$** | $-0.0509$ | **$0.0$** | $-0.0585$ | $1.0$ | Near-degenerate: on hull in DFT, $1.0\text{ meV/atom}$ above hull in ORB |
| **$\text{Cu}_3\text{Te}_2$** | $-0.0483$ | $10.8$ | $-0.0661$ | **$0.0$** | Near-degenerate: on hull in ORB, $10.8\text{ meV/atom}$ above hull in DFT |

**Key Hull Insights:**
- **Identical Thermodynamic Backbone:** Both DFT and ORB agree on $\text{Cu}_3\text{GeTe}_4$ as the *only* stable ternary ground state, and both identify $\text{GeTe}$, $\text{CuTe}$, and $\text{Cu}_3\text{Ge}$ as stable binary vertices.
- **Copper Telluride Near-Degeneracy:** The only minor difference is between $\text{Cu}_2\text{Te}$ and $\text{Cu}_3\text{Te}_2$. ORB places $\text{Cu}_2\text{Te}$ just **$1.0\text{ meV/atom}$** above the hull, well within the numerical noise threshold of DFT k-point grids and pseudopotentials.

#### 3. Metastable Phase Comparison
- **$\text{Cu}_2\text{GeTe}_3$:**
  - In LeMat-Bulk DFT: Lowest $E_f = -0.0967\text{ eV/atom}$, sitting **$3.0\text{ meV/atom}$** above the DFT convex hull.
  - In our ORB study: Lowest $E_f = -0.1119\text{ eV/atom}$, sitting **$15.1\text{ meV/atom}$** above the ORB convex hull.
  - Both methods consistently identify $\text{Cu}_2\text{GeTe}_3$ as an ultra-low-energy metastable phase immediately above the ground-state tie-line.
- **$\text{Cu}_5\text{Ge}_2\text{Te}_7$ (*Chem. Mater.* 2016):**
  - **Completely absent from LeMat-Bulk** (neither calculated in MP, Alexandria, nor OQMD).
  - Our de novo pipeline generated 85 polymorphs and resolved its ground-state polymorph at $e_\text{hull} = 89.7\text{ meV/atom}$, explaining the physical necessity of non-equilibrium Direct Joule Heating and rapid quenching observed in experiments.

---

## 6. Artifacts and Provenance

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
