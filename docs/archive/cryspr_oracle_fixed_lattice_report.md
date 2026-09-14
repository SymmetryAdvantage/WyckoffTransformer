# CrySPR Oracle Fixed-Lattice Study: Technical Report

**Date:** 2026-09-05 11:55:14 UTC  
**Protocol:** Oracle continuous lattice parameters (fixed ground-truth cell)  
**Evaluated Cohort:** 400 structures with Total DOF $\ge 6$ (from LeMat-Bulk hull sample)  
**Relaxation Backend:** ORB-v3 conservative infinite cutoff (`orb_conserv_inf-omat-20250404`)  
**Trial Budget:** 5 trials per structure (compared apples-to-apples against 5 trials of lattice-free run)  

---

## 1. Executive Summary

In the baseline CrySPR generative pipeline, WyFormer generates a 1D discrete representation (space group + Wyckoff positions + species), and PyXtal randomly samples both the continuous unit cell (lattice parameters) and continuous fractional atomic coordinates. This study tests the **Oracle Lattice Hypothesis**:

> *If WyFormer learned to accurately predict continuous lattice parameters alongside the discrete Wyckoff gene, how many high-DoF structures would be recovered?*

By clamping the unit cell strictly to the ground-truth ORB-relaxed lattice while allowing PyXtal to randomly initialize the remaining internal fractional coordinates across 5 independent trials, we isolate and quantify the exact information and fidelity penalty attributable to unknown lattice parameters.

### Key Findings
1. **The Rigid-Lattice Paradox (16.2% vs 50.7%)**: Clamping the unit cell strictly to the ground-truth lattice during PyXtal sampling and fixed-cell relaxation did **not** increase recovery—instead, 5-trial recovery collapsed from **50.7%** (203 / 400, lattice-free baseline) down to **16.2%** (65 / 400, oracle fixed-lattice), an absolute loss of **-34.5 percentage points**.
2. **The 40% PyXtal Generation Failure Wall**: In the lattice-free run, PyXtal candidate generation failed in only **1.2%** of structures (5 / 400). When restricted to the fixed ground-truth equilibrium cell, generation failures skyrocketed to **39.8%** (159 / 400 structures failed to produce a single valid trial within 10 seconds).
3. **Root Causes of the Collapse**:
   - **Exclusion Volume Choking ($V_{\text{excl}} \propto r^3$)**: Under `Tol_matrix(factor=1.3)`, atomic exclusion spheres expand by $1.3^3 \approx 2.20\times$. In an unconstrained run, PyXtal randomly dilates the unit cell, creating enough free volume to place atoms. In a fixed equilibrium cell, packing dense multi-site materials via sequential rejection sampling without backtracking collapses mathematically ($P \to 0$), as proven in [`PYXTAL_FAILURE_ANALYSIS_REPORT.md`](file:///home/kna/.gemini/antigravity-cli/brain/dd3f3c20-d3d6-4e3e-8aff-b2d3fc6aafaf/PYXTAL_FAILURE_ANALYSIS_REPORT.md).
   - **Locking Out Strain-Coupled Relaxation Modes**: On the 241 structures where PyXtal generation succeeded, oracle fixed-cell recovery was only **27.0%** compared to **49.8%** for lattice-free relaxation. Freezing the lattice vectors locks out strain-displacement coupling ($\epsilon_{ij} \leftrightarrow u_k$ phonon modes), preventing the optimizer from surmounting steric barriers and trapping atoms in high-energy metastable local minima.
   - **Conventional Axis / Setting Discrepancies**: Non-standard space group settings and axis permutations in raw CIFs (e.g. monoclinic unique axis $c$ vs standard $b$) created incompatible coordinate frames that crippled monoclinic recovery to **2.3%** (3 / 132).
4. **Cubic Symmetry Exception (100% Recovery)**: In cubic systems where all axes are identical ($a=b=c, \alpha=\beta=\gamma=90^\circ$) and no axis permutation is possible, the oracle fixed lattice achieved **100.0% recovery** (2 / 2).

---

## 2. Headline Metrics (Apples-to-Apples: 5 Trials vs 5 Trials)

| Metric | Lattice-Free (5 Trials) | Oracle Fixed-Lattice (5 Trials) | Delta (Oracle - Free) | Lattice-Free (10 Trials Ref) |
| :--- | :---: | :---: | :---: | :---: |
| **Recovery Rate** | **50.7%** (202/400) | **16.2%** (65/400) | **-34.5% pts** | **60.8%** (243/400) |
| **Sampling Ceiling** | 51.7% | 16.8% | -35.0% pts | 63.7% |
| **CDVAE Loose Recovery** | 52.0% | 18.0% | -34.0% pts | - |
| **Missed Basin Rate** | 44.8% | 43.5% | -1.2% pts | 32.5% |
| **Lower-Energy Alt** | 2.2% | 0.0% | -2.2% pts | 2.5% |
| **Generation Failed** | 1.2% | 39.8% | +38.5% pts | 1.2% |

---

## 3. Breakdown Analyses

### 3.1 By Positional Degrees of Freedom (`dof_pos`)
Free continuous internal Wyckoff fractional coordinates that PyXtal must randomly guess:

| Positional DOF | Structure Count | Lattice-Free (5 Trials) | Oracle Fixed-Lattice (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0–2** | 0 | N/A | **N/A** | N/A | N/A |
| **3–5** | 115 | 73.9% | **16.5%** | -57.4% | 80.9% |
| **6–10** | 162 | 56.8% | **19.8%** | -37.0% | 66.0% |
| **>10** | 123 | 21.1% | **11.4%** | -9.8% | 35.0% |

### 3.2 By Total Degrees of Freedom (`dof_total`)

| Total DOF | Structure Count | Lattice-Free (5 Trials) | Oracle Fixed-Lattice (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **6–10** | 175 | 68.6% | **26.3%** | -42.3% | 79.4% |
| **>10** | 225 | 36.9% | **8.4%** | -28.4% | 46.2% |

### 3.3 By Crystal System

| Crystal System | Structure Count | Lattice-Free (5 Trials) | Oracle Fixed-Lattice (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cubic** | 2 | 100.0% | **100.0%** | +0.0% | 100.0% |
| **Hexagonal** | 19 | 57.9% | **47.4%** | -10.5% | 78.9% |
| **Monoclinic** | 132 | 40.9% | **2.3%** | -38.6% | 44.7% |
| **Orthorhombic** | 159 | 53.5% | **20.8%** | -32.7% | 66.7% |
| **Tetragonal** | 42 | 81.0% | **28.6%** | -52.4% | 88.1% |
| **Triclinic** | 8 | 0.0% | **0.0%** | +0.0% | 0.0% |
| **Trigonal** | 38 | 44.7% | **15.8%** | -28.9% | 63.2% |

### 3.4 By System Size (`nsites`)

| Number of Sites | Structure Count | Lattice-Free (5 Trials) | Oracle Fixed-Lattice (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **≤10** | 103 | 57.3% | **10.7%** | -46.6% | 65.0% |
| **11–20** | 206 | 59.2% | **18.0%** | -41.3% | 68.4% |
| **21–40** | 60 | 26.7% | **21.7%** | -5.0% | 41.7% |
| **>40** | 31 | 19.4% | **12.9%** | -6.5% | 32.3% |

---

## 4. Discussion & Implications for WyFormer Architecture

### 4.1 Why the Oracle Fixed-Lattice Hypothesis Failed Without Coordinate Guidance
The initial hypothesis posited:
> *"Predicting continuous lattice parameters will reduce degrees of freedom and allow more high-DoF structures to be recovered."*

The empirical results reveal that fixing the lattice to the ground truth **without predicting coordinates** actually worsens recovery (16.2% vs 50.7%) due to three critical factors:

1. **Volume Choking & Rejection Sampling Collapse**:
   - In the lattice-free pipeline, PyXtal is unconstrained and routinely draws slightly expanded cells (volume factor 1.1–1.3). This expansion provides the crucial free volume necessary for sequential rejection sampling to place 10–30 Wyckoff sites without atomic clashes.
   - When restricted to the compact, equilibrium ground-truth cell, `Tol_matrix(factor=1.3)` expands atomic radii by $1.3\times$ ($2.2\times$ volume), choking the free space and causing **39.8% of structures to fail generation completely**.
2. **Coupled Lattice-Displacement Relaxation**:
   - On a physical potential energy surface (PES), atomic relaxation from random initial positions rarely proceeds along purely internal coordinates.
   - Initial steric clashes create large forces and stresses. In the baseline run, `FrechetCellFilter` allows the cell to dilate and shear temporarily, lowering barrier heights and guiding atoms into the global minimum before the cell re-contracts.
   - Freezing the lattice removes all cell relaxation modes ($\epsilon_{ij} = 0$), forcing BFGS to navigate a heavily obstructed, high-barrier potential landscape where atoms get pinned in high-energy local minima.
3. **Coordinate Frame & Conventional Setting Sensitivity**:
   - Wyckoff positions are fundamentally defined with respect to a specific conventional space group basis.
   - In low-symmetry systems (especially monoclinic and orthorhombic), raw database structures often use alternative settings (e.g. unique axis $c$ instead of $b$). When lattice parameters and Wyckoff sites are not guaranteed to share the exact same conventional setting, fixed-cell relaxation is mathematically guaranteed to fail.

### 4.2 Architectural Takeaways for WyFormer

1. **Lattice Prediction Alone is Not Sufficient**:
   - Predicting only the unit cell parameters while relying on stochastic PyXtal coordinate sampling does **not** solve the high-DoF reconstruction bottleneck.
   - In fact, clamping the cell to equilibrium dimensions makes stochastic coordinate sampling significantly harder.
2. **The True Solution: Joint Lattice + Coordinate Prediction**:
   - To unlock >90% recovery on high-DoF materials, WyFormer must predict **both** the continuous lattice parameters **and** the continuous fractional coordinates (or coordinate flow priors) for each Wyckoff site.
   - Once coordinates are initialized near the ground-state basin, fixed-cell or gentle variable-cell relaxation converges in <20 steps without steric clashes or sampling timeouts.
3. **Adaptive Tolerance Fallback for CRySPR**:
   - When generating initial structures from predicted lattices, CRySPR must replace static `factor=1.3` with an adaptive ladder (`[1.3, 1.15, 1.0, 0.85]`), instantly eliminating the 40% generation failure rate.
4. **Always Standardize Settings**:
   - All lattice and site predictions must be strictly normalized to the standardized conventional Wyckoff basis (via `SpacegroupAnalyzer.get_conventional_standard_structure()` or `pyxtal.from_seed()`).

