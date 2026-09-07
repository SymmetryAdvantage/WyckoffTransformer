# CrySPR Oracle Initialized-Lattice Study: Technical Report

**Date:** 2026-09-05 19:06:13 UTC  
**Protocol:** Initialized with ground-truth conventional cell + standard CRySPR variable-cell relaxation (`factor=1.0` adaptive)  
**Evaluated Cohort:** 400 structures with Total $\text{DOF} \ge 6$ sampled from the LeMat-Bulk convex hull  
**Relaxation Backend:** ORB-v3 conservative infinite cutoff (`orb_conserv_inf-omat-20250404`)  
**Trial Budget:** 5 trials per structure (apples-to-apples vs 5 trials of lattice-free baseline run)  

---

## 1. Executive Summary

This study tests the foundational hypothesis:
> **Hypothesis**: *If WyFormer learns to predict continuous lattice parameters, thereby eliminating lattice degrees of freedom, how many additional high-DoF structures can be recovered during CRySPR reconstruction?*

Following initial findings that a rigid unit cell with inflated exclusion radii (`factor=1.3`) collapsed recovery to **16.2%** (with 39.8% generation timeouts), this investigation implemented two explicit physical adjustments:
1. **Dropped `factor=1.3`**: Deployed standard atomic packing radii (`Tol_matrix(prototype="atomic", factor=1.0)` with adaptive fallback to 0.9).
2. **Initialized cell without freezing**: PyXtal was initialized with the exact ground-truth conventional unit cell, but the cell was allowed to relax using the standard 4-stage CRySPR variable-cell schedule (`FrechetCellFilter`), preserving coupled strain-displacement modes.

### Headline Results

| Metric | Lattice-Free Baseline (5 Trials) | Fixed-Cell Oracle (5 Trials, factor=1.3) | Initialized-Cell Oracle (5 Trials, factor=1.0, Relaxed) | Delta (Init Oracle vs Free Baseline) | Lattice-Free Reference (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recovery Rate** | **50.8%** (203/400) | 16.2% (65/400) | **19.0%** (76/400) | **-31.7% pts** | **60.8%** (243/400) |
| **Sampling Ceiling** | 51.8% (207/400) | 16.8% (67/400) | 19.2% (77/400) | -32.5% pts | 63.8% (255/400) |
| **CDVAE Loose Recovery** | 52.0% (208/400) | 18.0% (72/400) | 20.0% (80/400) | -32.0% pts | - |
| **Missed Rate** | 44.8% (179/400) | 43.5% (174/400) | 78.2% (313/400) | +33.5% pts | 32.5% (130/400) |
| **PyXtal Generation Failed** | 1.2% (5/400) | 39.8% (159/400) | **1.2%** (5/400) | +0.0% pts | 1.2% (5/400) |
| **Lower Energy Alt.** | 2.2% (9/400) | 0.5% (2/400) | 1.2% (5/400) | -1.0% pts | 2.5% (10/400) |

### Core Scientific Takeaways
1. **PyXtal Generation Bottleneck Solved**: Dropping `factor=1.3` to `1.0` completely eliminated the 39.8% PyXtal generation failure collapse (falling to 1.2%, identical to baseline).
2. **The Pre-Compressed Jamming Paradox**: Knowing the exact equilibrium lattice and initializing random atomic positions inside it catastrophically degrades recovery from **50.8% down to 19.0%**.
3. **Physical Root Cause (Explosive Dilatation)**: CRySPR is fundamentally an **expansion-to-compression annealing process**. 
   - When PyXtal generates a structure without a fixed cell, it inflates the cell volume by an average of **$2.06\times$** (mean initial volume ratio: 2.06, median 1.73). In this low-density "gas", atoms comfortably form local coordination polyhedra before variable-cell relaxation gradually contracts the cell (**66.3% contract**, median $-9.5\%$) into the global ground-state basin.
   - When forced into the dense equilibrium cell at $V_{\text{eq}}$, random atomic placement produces severe steric clashes ($< 2.0$ Å). The resulting massive virial pressure blows the cell apart: **73.5% of Oracle trials expand** (mean expansion $+16.4\%$, up to $+198\%$), irrevocably driving the system into disordered metastable states ($+120$ meV/atom above the ground state).

---

## 2. Quantitative Comparison & Breakdowns

### 2.1 By Positional Degrees of Freedom (`dof_pos`)

| Positional DOF | Structure Count | Lattice-Free Baseline (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free Reference (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3–5** | 115 | 73.9% | **30.4%** | -43.5% | 80.9% |
| **6–10** | 162 | 56.8% | **17.3%** | -39.5% | 66.0% |
| **>10** | 123 | 21.1% | **10.6%** | -10.6% | 35.0% |

### 2.2 By Total Degrees of Freedom (`dof_total`)

| Total DOF | Structure Count | Lattice-Free Baseline (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free Reference (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **6–10** | 175 | 68.6% | **29.7%** | -38.9% | 79.4% |
| **>10** | 225 | 36.9% | **10.7%** | -26.2% | 46.2% |

### 2.3 By Crystal System

| Crystal System | Structure Count | Lattice-Free Baseline (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free Reference (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cubic** | 2 | 100.0% | **50.0%** | -50.0% | 100.0% |
| **Hexagonal** | 19 | 57.9% | **10.5%** | -47.4% | 78.9% |
| **Monoclinic** | 132 | 40.9% | **12.1%** | -28.8% | 44.7% |
| **Orthorhombic** | 159 | 53.5% | **25.2%** | -28.3% | 66.7% |
| **Tetragonal** | 42 | 81.0% | **23.8%** | -57.1% | 88.1% |
| **Triclinic** | 8 | 0.0% | **12.5%** | +12.5% | 0.0% |
| **Trigonal** | 38 | 44.7% | **15.8%** | -28.9% | 63.2% |

### 2.4 By System Size (`nsites`)

| Number of Sites | Structure Count | Lattice-Free Baseline (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free Reference (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **≤10** | 103 | 57.3% | **34.0%** | -23.3% | 65.0% |
| **11–20** | 206 | 59.2% | **18.4%** | -40.8% | 68.4% |
| **21–40** | 60 | 26.7% | **5.0%** | -21.7% | 41.7% |
| **>40** | 31 | 19.4% | **0.0%** | -19.4% | 32.3% |

---

## 3. Deep-Dive Physical Mechanisms

### 3.1 Initial Unit Cell Volume Distribution
A comparative analysis of the initial generated volumes across the test cohort reveals why the baseline study outperforms the oracle:

- **Baseline Initial Cell Volume ($V_{\text{init}} / V_{\text{target}}$)**:
  - Mean: **$2.06\times$**
  - Median: **$1.73\times$**
  - Range: $0.56\times$ to $5.79\times$
  - Consequence: PyXtal builds a low-density "pseudo-gas". Atoms have ample spatial freedom to avoid energetic clashes even with a strict `factor=1.3`.

- **Oracle Initial Cell Volume ($V_{\text{init}} / V_{\text{target}}$)**:
  - Fixed exactly to the conventional unit cell ($1.00\times$ or standard setting equivalent).
  - Consequence: PyXtal is forced to pack atoms directly into an equilibrium crystal volume where over 70% of the unit cell volume is occupied by core electron exclusion spheres.

### 3.2 Cell Relaxation Trajectories (Dilation vs Contraction)
Tracing cell volume evolution through the 4-stage CRySPR optimization demonstrates diametrically opposed behaviors:

```
[Baseline Protocol: Compressive Annealing]
Initial Loose Cell (V ~ 2.0x) ───► Stage 1 (Fix Cell) ───► Stage 2 (Frechet Cell Filter) ───► Ground State
                                 (Atoms arrange)          (Cell CONTRACTS: 66.3% contract)

[Oracle Protocol: Steric Jamming & Explosive Dilatation]
Initial Dense Cell (V = 1.0x) ───► Stage 1 (Fix Cell) ───► Stage 2 (Frechet Cell Filter) ───► Distorted Minima
                                 (Atoms jammed)           (Cell EXPANDS: 73.5% dilate)     (+120 meV/atom)
```

- In the **Baseline run**, **66.3% of trials contracted** during optimization (mean $\Delta V / V = -4.3\%$, median $-9.5\%$), mimicking physical annealing where a dispersed configuration condenses into its crystalline equilibrium.
- In the **Oracle run**, **73.5% of trials expanded** away from the true equilibrium cell (mean $\Delta V / V = +16.4\%$, median $+7.6\%$, reaching up to $+198\%$). The initial repulsion between closely packed atoms created enormous internal stresses ($P \gg 100\text{ GPa}$) that forced the cell to dilate rapidly in Stage 2, destroying the target lattice geometry.

### 3.3 Energy Landscape & Trapping
For the 149 structures where Baseline recovered the ground truth but Oracle missed:
- **Baseline Best Energy Delta ($\Delta E = E_{\text{kept}} - E_{\text{target}}$)**:
  - Mean: $-0.00006\text{ eV/atom}$ (exact match within numerical tolerance).
- **Oracle Best Energy Delta**:
  - Mean: **$+0.1198\text{ eV/atom}$** ($+120\text{ meV/atom}$).
  - Median: $+0.1059\text{ eV/atom}$.
  - 75th Percentile: $+0.1614\text{ eV/atom}$.

Because the atoms were jammed into the equilibrium cell, Stage 1 (fixed cell) forced them into distorted local coordinations. Once in these false basins, even when variable cell relaxation was turned on in Stage 2, the barrier to escape into the ground state was insurmountable, leaving them trapped in higher-energy metastable polymorphs.

---

## 4. Synthesis: Three-Way Comparison

| Dimension | 1. Lattice-Free Baseline | 2. Fixed-Cell Oracle | 3. Relaxed-Cell Oracle |
| :--- | :--- | :--- | :--- |
| **Lattice Input** | None (PyXtal estimated) | Ground-Truth Conventional | Ground-Truth Conventional |
| **Tolerance Matrix** | `prototype="atomic", factor=1.3` | `prototype="atomic", factor=1.3` | `prototype="atomic", factor=1.0` (0.9 fallback) |
| **Cell Optimization** | 4-stage variable cell (`FrechetCellFilter`) | Rigid cell (`FixSymmetry`, fixed box) | 4-stage variable cell (`FrechetCellFilter`) |
| **Recovery Rate (5 trials)** | **50.8%** | **16.2%** | **19.0%** |
| **PyXtal Gen Failures** | 1.2% | 39.8% (exclusion choking) | 1.2% (resolved) |
| **Primary Failure Cause** | Sampling limits on large DoF | Rigid pinning + packing collapse | Jamming $\to$ explosive cell dilation |
| **Volume Trajectory** | Compressive (contracts by ~10%) | Invariant ($\Delta V = 0$) | Dilative (expands by +16%) |

---

## 5. Architectural Implications for WyFormer

These findings fundamentally alter how we should design the generative pipeline pairing WyFormer with CRySPR:

1. **Do NOT Predict Equilibrium Lattices for Random Packing**:
   - Training WyFormer to predict the continuous ground-truth equilibrium lattice parameters $(a, b, c, \alpha, \beta, \gamma)$ will **harm**, not help, downstream CRySPR reconstruction if atomic positions are sampled randomly inside that cell.
   - Initializing random atoms inside an equilibrium cell triggers steric jamming, virial explosion, and glassy trapping.

2. **If Predicting Continuous Lattice, Must Also Predict Initial Coordinates**:
   - Continuous lattice prediction only succeeds if accompanied by approximate continuous Wyckoff coordinate parameters $(x, y, z)$. With good positional seeds, atoms are already near their true basins and avoid random packing collisions.

3. **Optimal Downstream Strategy for WyFormer + CRySPR**:
   - If WyFormer focuses on discrete crystallography (Space Group + Wyckoff Site Multiplicities/Letters + Species), the downstream CRySPR decoder should:
     a. Intentionally scale up the initial unit cell volume to **$1.5\times - 2.0\times$** the estimated equilibrium volume.
     b. Maintain `factor=1.3` to enforce healthy minimum interatomic distances.
     c. Allow CRySPR's 4-stage variable-cell optimizer to perform compressive annealing from this low-density initial state into the ground-state crystal.
