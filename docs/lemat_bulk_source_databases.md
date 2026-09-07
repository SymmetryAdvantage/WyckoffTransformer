# Candidate Selection Procedures and Convex Hull Biases in LeMat-Bulk Source Databases

> **Purpose:** To support modeling efforts targeting the LeMat-Bulk thermodynamic convex hull by rigorously distinguishing:
> 1. **Where the database authors did not look** (structural, stoichiometric, and chemical unvisited space, where the convex hull is an artificially elevated ceiling).
> 2. **Where the database authors looked and did not find stable structures** (sampled high-energy / unstable configurations, which serve as genuine negative evidence).
>
> In accordance with project instructions, every section explicitly separates **(1) Factual Observations** (documented literature, algorithms, parameters, datasets, and codebase measurements) from **(2) Analysis** (inferential interpretations, sampling biases, physical blind spots, and attack strategies).

---

## 1. Materials Project (MP)

### 1.1 Candidate Selection & Structural Generation

#### Factual Observations
1. **Experimental Ingestion (ICSD)**:
   * MP ingested ~100,000 raw crystal records from the Inorganic Crystal Structure Database (ICSD, FIZ Karlsruhe) (*Jain et al., APL Mater. 2013*).
   * **Handling of Partial Occupancies**: Disordered experimental structures were not computed directly. Instead, `pymatgen.transformations.advanced_transformations.OrderDisorderedStructureTransformation` was applied, generating ordered supercells up to a maximum size (typically $\le 200$ atoms) and ranking them by **electrostatic Ewald summation**. Only the single lowest Ewald-energy ordering was submitted to DFT. Disordered structures requiring supercells larger than the threshold were discarded.
   * **Experimental Ratio**: Across the canonical Materials Project database (~154,700 entries in v2022/v2023), **~52,000–56,000 structures (~34–36%)** map directly to an ICSD record (`theoretical: False` / `database_IDs.icsd`). The remaining **~64–66%** are theoretical/hypothetical calculations. In LeMat-Bulk's 138,931 `mp-` rows, 45,274 are ICSD-backed (32.6%).
2. **Theoretical Candidate Engine — Hautier et al. (2011) Ionic Substitution**:
   * Reference: *Hautier, Fischer, Ehrlacher, Jain, & Ceder, Inorg. Chem. 50, 656–663 (2011)* (`pymatgen.analysis.structure_prediction.substitution_probability`).
   * **Mathematical Formulation**: Models pairwise ion exchange likelihood as a Boltzmann-like distribution:
     $$p(s_1 \leftrightarrow s_2) = \frac{1}{Z}\exp(\lambda_{s_1, s_2})$$
     where $Z = \sum_{s_1, s_2} \exp(\lambda_{s_1, s_2})$ is the partition function over all species pairs, and $\lambda_{s_1, s_2}$ is a symmetric coupling parameter learned via maximum likelihood estimation from historical crystallographic data.
   * **Training Data**: Trained on **1,883 experimental crystal structures** from the ICSD covering **4,286 compounds across 183 structural prototypes** and **260 ionic species** across formal oxidation states from $-3$ to $+8$. For species pairs unseen in the training set, a default penalty weight $\lambda = -5.0$ was applied.
   * **Screening Rules**:
     * **Strict Charge Neutrality**: Proposals must satisfy formal valence neutrality $\sum_i q_i x_i = 0$ within $\pm 10^{-9}$.
     * **Probability Threshold**: Candidate substitutions were retained only if the compound substitution probability exceeded a threshold $p \ge 10^{-3}$ (or $10^{-4}$ in exploratory sweeps).
     * **Volume Scaling (`RLSVolumePredictor`)**: Lattice parameters of the substituted template were scaled using Shannon ionic radii:
       $$V_{\text{target}} = V_{\text{ref}} \left( \frac{\sum_i r_i n_i^{1/3}}{\sum_j r_{\text{ref}, j} (n_{\text{ref}, j})^{1/3}} \right)^3$$
3. **Prototype Substitution Campaigns**:
   * Systematic decoration of canonical mineral prototypes: Perovskites ($ABX_3$) and double perovskites ($A_2BB'X_6$), normal/inverse spinels ($AB_2X_4$), olivines ($LiMPO_4$), tavorites ($LiMPO_4F$), Heusler and half-Heusler alloys ($X_2YZ$, $XYZ$), delafossites ($AMO_2$), Chevrel phases, garnets, pyrochlores, rutiles, rocksalts, and fluorites.
4. **Domain-Specific Screening Campaigns**:
   * **Battery Intercalation (JCESR / Persson / Ceder)**: Systematic topotactic delithiation/desodiation of known Li/Na hosts to calculate voltage profiles; thousands of non-equilibrium intermediate states were relaxed and stored.
   * **Multivalent Consortium (`mvc-`)**: Intercalation host exploration for $Mg^{2+}, Ca^{2+}, Zn^{2+}, Al^{3+}$.
   * **Functional Scans**: High-throughput scans for transparent conducting oxides (TCOs), thermoelectrics (BoltzTraP transport calculations on ~48,000 materials), and piezoelectric/elastic tensors.
5. **Pre-DFT Physical Filters**:
   * Minimum distance check: rejected any structure where interatomic distance $d_{ij} < 0.6 \times (r_i^{\text{cov}} + r_j^{\text{cov}})$.
   * `StructureMatcher` deduplication (`ltol = 0.2`, `stol = 0.3`, `angle_tol = 5.0°`).

#### Analysis
* **Where MP Looked and Found High Energy (True Unstable)**:
  * Battery delithiation intermediates: Topotactically stripped frameworks (e.g. $Li_{1-x}MO_2$ at intermediate $x$) frequently sit $0.1–0.5\text{ eV/atom}$ above the hull. These are true sampled non-ground states.
  * Size-mismatched prototype substitutions: Perovskite and spinel substitutions that violate the Goldschmidt tolerance factor ($t < 0.82$ or $t > 1.05$) were computed and found to relax into high-energy, distorted, or unstable configurations.
* **Where MP Did NOT Look (Unvisited Space)**:
  * **The Oxygen Bias**: Oxygen appears in **>52.9%** of the entire Materials Project (81,887+ structures), followed by Li, Mg, P, S, Mn, Fe, Na, Si, F, and Co.
  * **Intermetallic Neglect**: Non-oxide, non-halide intermetallics (e.g. transition-metal aluminides, silicides, borides, refractory binary/ternary alloys) were largely ignored unless they overlapped with thermoelectric or topological insulator searches.
  * **Chemical Arity Ceiling**: Quinaries and higher ($N_{\text{elements}} \ge 5$) make up **<0.5%** of MP. Quaternaries were searched almost exclusively within restricted prototype families (double perovskites, quaternary Heuslers, oxyhalides).
  * **Elemental Voids**: Actinides beyond Uranium ($Th: 1,059; U: 2,439; Pu: 463$) drop to near zero ($Ac: 304; Np: 410$; all transuranics $Am, Cm, \dots$ are 0). Radioactive elements ($Po, At, Rn, Fr, Ra$) are completely absent. Noble gases ($He, Ne, Ar, Kr, Xe$) are virtually absent except for a few high-pressure xenon phases or clathrates.
  * **Symmetry & Wyckoff Bias**: Theoretical candidates are overwhelmingly cubic, hexagonal, or tetragonal. Triclinic ($P1, P\bar{1}$) and monoclinic ($P2_1/c$) space groups were essentially never generated by the theoretical substitution pipeline; they appear in MP almost exclusively as ICSD imports.

---

### 1.2 MP DFT Protocol & Energy Calibration

#### Factual Observations
1. **DFT Setup (`MPRelaxSet`)**:
   * Code: VASP (PAW potentials, PBE functional).
   * Plane-wave cutoff: $ENCUT = 520\text{ eV}$ across all elements.
   * Brillouin zone sampling: $\Gamma$-centered or Monkhorst-Pack mesh with `reciprocal_density = 64` (~1,000 kppa).
   * Geometry optimization: Double relaxation (`ISIF = 3`, `IBRION = 2`) followed by a static run with Blöchl tetrahedron integration (`ISMEAR = -5`).
2. **Hubbard U Policy**:
   * Dudarev rotationally invariant approach (`LDAUTYPE = 2`) with $U_{\text{eff}} = U - J$.
   * **Restricted to Transition Metal Oxides and Fluorides**:
     * $Co: 3.32\text{ eV}, Cr: 3.7\text{ eV}, Fe: 5.3\text{ eV}, Mn: 3.9\text{ eV}, Mo: 4.38\text{ eV}, Ni: 6.2\text{ eV}, V: 3.25\text{ eV}, W: 6.2\text{ eV}$.
     * Pure elemental metals, sulfides, nitrides, phosphides, and intermetallics use standard GGA ($U = 0$).
3. **Magnetism**:
   * Initialized spin-polarized (`ISPIN = 2`) with **ferromagnetic (FM) high-spin default moments** (`MAGMOM = 5.0 µ_B` for Fe, Mn, Cr; `0.6 µ_B` for Co, Ni). Antiferromagnetic (AFM) or non-collinear configurations were not systematically sampled.
4. **Forces & Convergence in LeMat-Bulk**:
   * MP production historically converged to energy tolerances (`EDIFF = 5e-5 * N_atoms`) rather than strict force thresholds.
   * Codebase measurement (`docs/composition_screening.md`): Median residual force in MP is **0.028 eV/Å**. Only **35.5%** of ICSD-backed MP entries pass a tight $f_{\text{max}} \le 0.02\text{ eV/Å}$ cut.

#### Analysis
* **Convex Hull Inhomogeneity**: The selective application of Hubbard $U$ only to oxides and fluorides creates a piecewise energy landscape. MP applies empirical mixing schemes (`MaterialsProject2020Compatibility`) to reconcile GGA and GGA+U energies, but subtle chemical potential mismatches persist at the interface between oxides and non-oxides.
* **Magnetic Artificial Elevation**: In systems where the true ground state is antiferromagnetic (e.g. transition-metal monoxides $NiO, CoO, FeO, MnO$, or layered halides), MP's default FM initialization often converges to an artificially elevated metastable state (typically $50–200\text{ meV/atom}$ above the true ground state), artificially elevating the convex hull.

---

## 2. Open Quantum Materials Database (OQMD)

### 2.1 Candidate Selection & Structural Generation

#### Factual Observations
1. **Experimental Ingestion (ICSD)**:
   * Reference: *Saal et al., JOM 65, 1501–1509 (2013)*; *Kirklin et al., npj Comput. Mater. 1, 15010 (2015)*.
   * **Strict Exclusion of Disorder**: Unlike MP, OQMD **completely discarded all experimental entries with partial occupancies, fractional site probabilities, or substitutional disorder**.
   * **Unit Cell Size Limit**: Restricted to primitive cells containing **$\le 50$ atoms** (in early phases $\le 40$ atoms).
   * **Volume Reduction & Deduplication**: Primitive cells identified via Niggli reduction; duplicates consolidated under canonical entry IDs. Yielded **~30,000–45,000 unique ordered experimental compounds**.
2. **Exhaustive Binary Combinatorics (~88 Prototypes)**:
   * Identified ~88 binary structure archetypes most frequently observed in the ICSD.
   * Selected an elemental pool of **84 elements** (H through Bi, plus Th and U; excluding noble gases).
   * **Exhaustive Combinatorial Decoration**: Every pair of elements $(A, B)$ was substituted into all ~88 binary prototypes.
     * For asymmetric prototypes (e.g. $NiAs, CaF_2, WC$): both permutations $(A, B)$ and $(B, A)$ were calculated ($84 \times 83 = 6,972$ combinations per prototype).
     * For symmetric prototypes (e.g. $B1$ NaCl, $B2$ CsCl): $84 \times 83 / 2 = 3,486$ combinations were calculated.
     * Total binary production: **>300,000 binary DFT calculations**.
3. **Selective Ternary Prototype Families**:
   * Exhaustive ternary decoration across 84 elements ($84 \times 83 \times 82 \approx 571,000$ per prototype) was computationally impossible. OQMD therefore restricted combinatorial substitution to targeted, high-value ternary families:
     * **Heusler & Half-Heusler Alloys (>200,000 calculations)**: Full Heusler ($L2_1, X_2YZ, Fm\bar{3}m$), Half-Heusler ($C1_b, XYZ, F\bar{4}3m$), and Inverse Heusler ($X_2YZ, F\bar{4}3m$).
     * **Perovskites ($ABX_3$)**: Ideal cubic ($Pm\bar{3}m$) and distorted tilted variants ($Pnma$ and $R\bar{3}c$), where $X = \text{O, F, N, Cl}$.
     * **Spinels ($AB_2X_4$)**: Normal and inverse spinels ($Fd\bar{3}m$, $X = \text{O, S, Se, Te}$).
     * **Chalcopyrites ($ABC_2$)**: $E1_1$ ($I\bar{4}2d$, $C = \text{S, Se, Te}$).
     * **Delafossites ($ABO_2$) & Garnets ($A_3B_2C_3O_{12}$)**.
4. **Quaternaries and Higher**:
   * **Zero systematic combinatorial prototype enumeration**. Higher-order materials exist exclusively through ICSD experimental imports and targeted sub-projects (e.g. kesterite photovoltaics $Cu_2ZnSnS_4$, double perovskites).
5. **Geometry Pre-Processing**:
   * **Volume Scaling**: Candidate cell volumes were initialized by summing tabulated STP atomic volumes:
     $$V_{\text{init}} = \sum_i N_i V_{\text{atom}, i}$$
   * **Clash Rejection**: Pairwise distances checked; rejected if $d_{ij} < 0.6 \times (r_i + r_j)$.

#### Analysis
* **Exhaustive Binary Space vs. Sparse Ternary/Quaternary Space**:
  * In **binary space**, if a compound in any of the ~88 prototypes does not appear on the convex hull, it is because it was **calculated and found thermodynamically unstable or high in energy**. Binary absence in standard prototypes is strong negative evidence.
  * In **ternary space**, if a compound does not belong to the Heusler, spinel, perovskite, or chalcopyrite families, it was **never calculated** unless it was an ordered ICSD compound. Ternary absence is almost pure unvisited space.
  * In **quaternary space**, virtually the entire phase space is unvisited.
* **Symmetry & Wyckoff Blind Spots**:
  * OQMD's prototype library is overwhelmingly dominated by high-symmetry cubic, hexagonal, and tetragonal space groups with low-multiplicity Wyckoff positions having fixed fractional coordinates ($(0,0,0), (1/4,1/4,1/4), (1/2,1/2,1/2)$).
  * Monoclinic and triclinic space groups, complex polyhedral distortions, van der Waals layered materials, open frameworks, and general Wyckoff positions $(x, y, z)$ are entirely missing from the theoretical pool.

---

### 2.2 OQMD DFT Protocol & Energy Calibration

#### Factual Observations
1. **DFT Setup (`qmpy`)**:
   * Code: VASP (PAW potentials, PBE functional).
   * Plane-wave cutoff: $ENCUT = 520\text{ eV}$ uniform across all elements.
   * k-point sampling: $\Gamma$-centered Monkhorst-Pack grids (4,000–8,000 KPRA).
   * Multi-stage relaxation: `coarse_relax` $\to$ `fine_relax` (`ISIF = 3`, `IBRION = 2`) $\to$ `standard` static run with Blöchl tetrahedron integration (`ISMEAR = -5`). Force tolerance: `EDIFFG = -0.01` to `-0.02 eV/Å`.
2. **Hubbard U Policy**:
   * Dudarev rotationally invariant approach (`LDAUTYPE = 2`).
   * **Applied strictly to transition metal Oxides**:
     * $V: 3.25\text{ eV}, Cr: 3.70\text{ eV}, Mn: 3.90\text{ eV}, Fe: 5.30\text{ eV}, Co: 3.32\text{ eV}, Ni: 6.20\text{ eV}$.
     * **Critical Difference from MP**: OQMD did **not** apply $U$ to fluorides, sulfides, or nitrides.
3. **Reference State Energy Fitting (Kirklin et al. 2015)**:
   * To reconcile GGA overbinding of gas-phase molecules ($O_2, N_2, F_2, Cl_2$) and the GGA vs. GGA+U discontinuity, OQMD fitted an empirical set of reference chemical potentials $\mu_i$ by minimizing the root-mean-square error against experimental standard enthalpies of formation ($\Delta H_f^{\text{expt}}$) across hundreds of binary and ternary compounds.
4. **Magnetism**:
   * Collinear spin-polarized calculations (`ISPIN = 2`) initialized with ferromagnetic (FM) positive moments (`MAGMOM = 5.0 µ_B`). Antiferromagnetic or non-collinear orderings were not systematically sampled.

#### Analysis
* **Cross-Database Discrepancies**:
  * A fluoride calculation in MP has a Hubbard $U$ applied, whereas in OQMD it has $U = 0$. Ingesting both into LeMat-Bulk without functional/parameter alignment creates an artificial offset in formation energy.
  * OQMD's uniform FM initialization artificially raises the calculated formation energy of antiferromagnetic oxides and sulfides relative to the experimental convex hull.

---

## 3. Alexandria Database

### 3.1 Candidate Selection & Structural Generation

#### Factual Observations
1. **Evolution Across Four Generations**:
   * **Precursor Seed (2021–2022)**: *Schmidt et al., Sci. Data 9, 84 (2022)* (arXiv:2109.15246). Re-relaxed ~175k crystalline materials from Materials Project and AFLOW using PBEsol and performed single-point SCAN calculations. Cleaned AFLOW prototypes by discarding unphysical `_DEVIL_PROTOTYPES_` and combinations causing SCF divergence.
   * **Alexandria Round 1 (2022–2023)**: *Schmidt et al., Adv. Mater. 35, 2210788 (2023)* (arXiv:2210.00579). Harvested **~2,500 binary and ternary crystal prototypes** from AFLOW and ICSD. Combinatorially substituted up to **89 elements** with formal charge neutrality and Pauling electronegativity filters ($\sim 10^9$ candidate combinations). Screened with **Crystal Graph Attention Networks (CGAT)**; candidates with predicted **$E_{\text{hull}} < 50\text{ meV/atom}$** were submitted to VASP for full PBEsol relaxation, yielding 19,512 new ground states directly on the hull and ~150,000 within 50 meV/atom.
   * **Alexandria Rounds 2 & 3, 2D & 1D (2023–2024)**: *Schmidt et al., Mater. Today Phys. 48, 101560 (2024)* & *Wang et al., 2D Mater. 10, 035007 (2023)*. Retrained CGAT ensemble models on relaxed volume and hull distance; expanded into quaternaries (`cgat_comp/quaternaries`); generated 2D materials via a Wyckoff combinatorial engine across 2D layer groups (yielding ~6,500 stable 2D materials); generated 1D nanowire prototypes.
   * **Alexandria 2.0 / AI-Driven Expansion (2025–2026)**: *Cavignac et al., J. Phys. Mater. (2026)* (arXiv:2512.09169) & *De Breuck et al., arXiv:2501.16051*. Replaced prototype substitution with the **Matra-Genoa** autoregressive transformer (using an invertible Wyckoff representation conditioned on convex hull distance). Sampled **119 million candidates**, pre-relaxed them with **Orb-v2** uMLIP, predicted energies with **ALIGNN** / Orb-v2, and submitted structures with predicted **$E_{\text{hull}} < 100\text{ meV/atom}$** to VASP. Added **1.3 million DFT-validated compounds** (99% hit rate within 100 meV/atom of the hull; 74,000 new ground states), expanding the total database to **5.8 million structures**.
2. **Stoichiometric Scope**:
   * Explored >30 systematic stoichiometries:
     * Binaries: $AB, AB_2, AB_3, AB_4, AB_5, A_2B_3, A_2B_5, A_3B, A_3B_2, A_3B_4, A_3B_5, A_4B_3, A_5B_2$.
     * Ternaries: $ABC, AB_2C, ABC_2, A_2BC, AB_2C_2, ABC_3, A_2BC_3, AB_2C_3, ABC_4, A_2BC_4, A_3BC_4$, perovskites ($ABX_3$), spinels ($AB_2X_4$), Heusler alloys ($X_2YZ$, $XYZ$), antiperovskites, delafossites, chalcopyrites, kesterites.
3. **Space Boundaries & Excluded Scope**:
   * **Unit Cell Size Limit**: Capped at small cells ($\le 20$ or $\le 30$ atoms in earlier rounds; very few structures $> 40$ atoms).
   * **Elemental Exclusions**: Noble gases ($He, Ne, Ar, Kr, Xe, Rn$), transuranics ($Np, Pu, Am, \dots$), and short-lived radioactives ($Tc, Pm$) were completely excluded from combinatorial rounds.
   * **Disordered Systems**: Zero partial occupancies or SQS configurations; strictly stoichiometric, integer-occupancy crystals.

#### Analysis
* **The "Zero-Prototype" Void**: Rounds 1–3 were fundamentally anchored to 2,500 known prototype scaffolds. Any composition whose physical ground state requires a coordination geometry or packing arrangement not present in those 2,500 prototypes could never be generated.
* **Active Learning Censoring**: Because Alexandria submitted candidates to DFT only if an ML surrogate (CGAT or Orb-v2/ALIGNN) predicted $E_{\text{hull}} < 50–100\text{ meV/atom}$, any composition where the ML surrogate had a large positive error (falsely predicting high energy) was **censored upstream of DFT**.
* **Impact of Alexandria on the Hull**: In the WyckoffTransformer codebase analysis (`docs/composition_screening.md`), adding Alexandria to MP and OQMD lowered the formula minimum by $>50\text{ meV/atom}$ for **36.3%** of previously computed formulas, but for only **3.3%** of formulas with an ICSD-backed entry. This confirms that theoretical hulls prior to Alexandria were extremely loose, whereas experimental hulls were relatively tight.

---

### 3.2 Alexandria DFT Protocol & Energy Calibration

#### Factual Observations
1. **DFT Setup**:
   * Code: VASP 5.4.
   * Functional: **PBEsol** used for geometry optimization; single-point **SCAN** performed on PBEsol geometry; standard PBE run for Materials Project compatibility.
   * Pseudopotentials: PAW PBE 5.4. A **custom `Cs_sv` PAW** was used to resolve negative core density artifacts in standard VASP Cs.
   * Plane-wave cutoff: $ENCUT = 520\text{ eV}$ (raised to 600–700 eV if needed).
   * k-point sampling: $\Gamma$-centered Monkhorst-Pack grids (2,000 kppa for relaxation, 8,000 kppa for static single-point).
2. **Hubbard U Policy**:
   * Adopted MP's selective +U on transition metal oxides/fluorides for PBE calculations.
   * **PBEsol and SCAN calculations omit Hubbard U entirely (pure GGA / meta-GGA)**.
   * *Schmidt et al. ("Better without U", 2026)* documented that selective Hubbard U introduces discontinuous potential energy surfaces, recommending uncorrected PBEsol/SCAN for training machine learning potentials.
3. **Forces & Convergence in LeMat-Bulk**:
   * Enforced strict ionic force convergence: `EDIFFG = -0.02 eV/Å`.
   * In LeMat-Bulk, **95.6% of Alexandria entries pass $f_{\text{max}} \le 0.02\text{ eV/Å}$**.

#### Analysis
* **Force Filter Trap**: Applying a naive filter of `max_force <= 0.02` to LeMat-Bulk (as done in `scripts/pipeline_lemat_20wyckoffs.py`) preserves 95.6% of Alexandria entries but discards 64.5% of Materials Project's ICSD-backed entries. This is an accidental provenance filter that strips out the scarcest, highest-quality experimental labels simply because MP reported looser convergence tolerances.
* **Elemental Reference Shifts**: Elemental reference chemical potentials for heavy halides (Br, I) and noble metals (Ag) shift by 20–36 meV/atom between shallow (MP/OQMD) and deep (Alexandria) hulls, causing localized elevation in unvisited ternary halide spaces.

---

## 4. Comprehensive Cross-Database Comparison Matrix

### 4.1 Factual Parameter Comparison

| Dimension | Materials Project (MP) | Open Quantum Materials Database (OQMD) | Alexandria Database |
| :--- | :--- | :--- | :--- |
| **Primary Identifier Prefix** | `mp-` / `mvc-` | `oqmd-` | `agm-` |
| **Experimental Ingestion** | ICSD (~55k entries; disordered resolved via Ewald) | ICSD (~35k entries; **disordered discarded**) | Indirectly via MP and AFLOW experimental subsets |
| **Theoretical Generation Engine** | **Hautier et al. (2011)** data-mined substitution + battery insertion/delithiation | **Combinatorial binary grid** (~88 prototypes $\times$ 84 elements) + selective ternary families | **2,500 AFLOW/ICSD prototypes** (Rounds 1–3) + **Matra-Genoa** transformer (Alexandria 2.0) |
| **ML Pre-Screening Filter** | None (pure crystal chemistry) | None (volume scaling + clash check) | **CGAT** ($E_{\text{hull}} < 50\text{ meV}$) in Rd 1; **Orb-v2 / ALIGNN** ($E_{\text{hull}} < 100\text{ meV}$) in 2.0 |
| **Binary Search Strategy** | Targeted (dominated by oxides/halides) | **Exhaustive brute-force** (all $84 \times 83$ element pairs in 88 prototypes) | Combinatorial across >13 binary stoichiometries |
| **Ternary Search Strategy** | Targeted (battery cathodes, perovskites, spinels) | **Selective only**: Heuslers ($>200\text{k}$), spinels, perovskites, chalcopyrites | High across >17 ternary stoichiometries filtered by CGAT/Orb-v2 |
| **Quaternary / Quinary Scope** | Quaternaries in select prototypes; **quinaries <0.5%** | **Zero systematic enumeration** (ICSD only) | Sampled via active learning / Matra-Genoa; no combinatorial grid |
| **Unit Cell Size Limit** | $\le 200$ atoms (>95% $\le 50$) | **$\le 50$ atoms** | **$\le 20$–$30$ atoms** (extended $\le 40$) |
| **Elemental Scope** | 89 elements; **>52.9% contain Oxygen**; noble gases/transuranics 0 | 84 elements; noble gases/transuranics 0 | 89 elements; noble gases/transuranics 0 |
| **Symmetry Bias** | High-symmetry cubic/tetragonal/hexagonal; $P1$ absent | Overwhelmingly cubic/hexagonal; **monoclinic & triclinic missing** | High-symmetry prototype bias; fixed Wyckoff positions |
| **Exchange-Correlation** | PBE | PBE | **PBEsol** (relaxations), **SCAN** (static), **PBE** |
| **Hubbard U Formulation** | **+U on Oxides AND Fluorides** (V, Cr, Mn, Fe, Co, Ni, Mo, W) | **+U on Oxides ONLY** (none on fluorides, sulfides, nitrides) | **Selective +U in PBE; NO +U in PBEsol or SCAN** |
| **Magnetic Initialization** | Ferromagnetic default (`ISPIN = 2`) | Ferromagnetic default (`ISPIN = 2`) | Ferromagnetic default (`ISPIN = 2`) |
| **Force Convergence ($f_{\text{max}}$)** | Median ~0.028 eV/Å (35.5% $\le 0.02$) | `EDIFFG = -0.01` to `-0.02` eV/Å | `EDIFFG = -0.02` eV/Å (**95.6% $\le 0.02$**) |

### 4.2 Comparative Analysis: LeMat-Bulk Aggregation Effects
* **De-duplication via BAWL**: LeMat-Bulk unifies these three databases using the Bonding Algorithm Weisfeiler-Lehman (BAWL) fingerprinting scheme on Niggli cells and Effective Coordination Number (ECoN) graphs. Where duplicate structures collide across databases, LeMat-Bulk retains the lowest-energy entry.
* **The Single-Entry Trap**: In LeMat-Bulk, **66.6% of reduced formulas have exactly one calculated structure**. For these singletons, the upper bound $E_{\text{hull}}(X)$ is defined by a single attempt from a single database's specific generation pipeline, making the bound exceptionally loose unless backed by an ICSD experiment.

---

## 5. Attacking the LeMat-Bulk Hull: Modeling "Unvisited Space" vs. "Unstable Space"

### 5.1 Factual Baseline in the Project Codebase
From measurements recorded in `docs/composition_screening.md` and `docs/composition_screening_results.md`:
* **Archive Size**: 4,745,121 usable structures across 2,329,360 reduced formulas.
* **ICSD-Backed Formulas**: Only **37,490 formulas (1.61%)** hold an ICSD-backed entry.
* **Hull-Defining Formulas**: Only **3.67%** of formulas define the hull, and only about an eighth of those are experimentally observed.
* **Experimental Floor Sharpness**: On formulas holding an ICSD-backed entry, the experimental structure is the minimum 55.4% of the time, with a mean excess of only 17.3 meV/atom, and only 4.4% are beaten by >50 meV/atom. The bound is tight where experimental data exists, and loose everywhere else.

---

### 5.2 Analysis: Concrete Attack Vectors on the LeMat-Bulk Hull

The mathematical objective is to estimate where the true physical floor $f^*(X)$ is substantially lower than the current archive hull $E_{\text{hull}}(X)$:

$$\Delta E(X) = E_{\text{hull}}(X) - f^*(X) \gg 0$$

Based on the documented candidate generation procedures, five major structural and chemical regimes represent artificially elevated hulls rather than genuine thermodynamic instability:

#### Attack Vector 1: Low-Symmetry & Continuous Wyckoff Voids
* **Mechanism**: Theoretical structures in all three databases were generated by decorating prototype scaffolds whose atoms sit on fixed fractional Wyckoff sites ($(0,0,0), (1/2,1/2,1/2), (1/4,1/4,1/4)$).
* **The Blind Spot**: Low-symmetry monoclinic ($P2_1/c, C2/c$) and triclinic ($P1, P\bar{1}$) space groups, as well as general Wyckoff positions $(x, y, z)$ requiring continuous coordinate optimization, were never combinatorially sampled.
* **Attack Strategy**: Identify chemical compositions in the database that relaxed to high-energy cubic or tetragonal phases with large residual stresses or soft phonon modes (e.g. perovskite/spinel stoichiometries with Goldschmidt tolerance factor $t < 0.85$). Generating symmetry-broken monoclinic/triclinic distorted structures can undercut the local hull by $>100\text{ meV/atom}$.

#### Attack Vector 2: Non-Oxide Intermetallics and Chalcogenides
* **Mechanism**: Materials Project allocated over half its compute to oxides, fluorides, and battery cathode hosts. OQMD exhaustively sampled binaries and ternary Heuslers, but skipped higher-order intermetallics. Alexandria required an existing prototype.
* **The Blind Spot**: Ternary and quaternary transition-metal main-group systems ($M_1 - M_2 - X$, where $X = \text{B, C, Si, P, Ge, As, Se}$) outside standard Heusler or chalcopyrite stoichiometries.
* **Attack Strategy**: Target ternary transition-metal silicides, phosphides, and borides. Because existing databases evaluated only binary end-members and a few Heusler ratios, the convex hull facet bridging them is flat and artificially high, allowing novel ternary phases to readily lower the hull.

#### Attack Vector 3: Non-Standard Stoichiometric Ratios
* **Mechanism**: OQMD and Alexandria only enumerated candidate structures along rigid stoichiometric lines:
  * Binaries: $1:1, 1:2, 1:3, 1:4, 2:3, 1:5, 1:12$.
  * Ternaries: $1:1:1, 1:1:2, 1:1:3, 1:2:4, 2:1:1$.
* **The Blind Spot**: Intermediate stoichiometries:
  * Binaries: $A_3B_4, A_3B_5, A_2B_5, A_4B_5, A_5B_8$.
  * Ternaries: $A_2B_3C_4, A_3B_2C_5, AB_4C_2, A_3BC_3, A_5B_3C_8$.
* **Attack Strategy**: Construct phase diagrams and locate tie-lines between adjacent sampled ratios. The hull facet between $AB$ and $AB_2$ is often defined by a simple linear tie-line because no intermediate $A_3B_4$ or $A_2B_3$ structure was ever calculated. Generating candidates directly at these intermediate stoichiometries will break the linear convex hull.

#### Attack Vector 4: Intermediate Cell Sizes ($N_{\text{atoms}} \in [20, 60]$) and Layered vdW Materials
* **Mechanism**: Alexandria strictly capped primitive cells at $\le 20$–$30$ atoms; OQMD capped at $\le 50$ atoms. Layered materials with van der Waals gaps or long-period superstructures were omitted from combinatorial rounds.
* **The Blind Spot**: Complex packing arrangements, modulated structures, and van der Waals heterostructures requiring 24–60 atoms per primitive cell.
* **Attack Strategy**: Use WyckoffTransformer to generate moderate-sized unit cells (30–60 atoms) in layered chalcogenides, halogen-substituted chalcohalides, and complex polyanion networks.

#### Attack Vector 5: Antiferromagnetic (AFM) Ground States
* **Mechanism**: All three databases initialized collinear calculations in a uniform ferromagnetic (FM) state (`ISPIN = 2`, high-spin parallel moments).
* **The Blind Spot**: For many transition metal compounds (especially oxides, sulfides, and fluorides of Mn, Fe, Co, Ni, and Cr), the true physical ground state is AFM (e.g. Type-G or Type-A antiferromagnetism), which typically lies **50–200 meV/atom lower** in energy than the forced FM state.
* **Attack Strategy**: Compounds containing magnetic ions that currently sit slightly above the hull ($0 < E_{\text{hull}} \le 100\text{ meV/atom}$) in LeMat-Bulk may be false non-ground states. Relaxing them with proper AFM sublattices can drop their energy below the hull.

---

### 5.3 Actionable Feature Engineering: Sampling Intensity vs. Thermodynamic Floor

Instead of treating database provenance as a categorical token (`mp-`, `agm-`, `oqmd-`), construct continuous physical covariates representing search intensity and structural novelty:

1. **Prototype Distance Metric**:
   $$d_{\text{proto}}(\text{gene}) = \min_{P \in \mathcal{P}_{\text{known}}} \text{Distance}(\text{gene}, P)$$
   where $\mathcal{P}_{\text{known}}$ is the union of the 88 OQMD binary prototypes and 2,500 Alexandria prototypes. High $d_{\text{proto}}$ signifies structural space the database authors could not have visited.
2. **Formula Attempt Multiplicity $n(X)$**:
   The total number of relaxation attempts recorded in LeMat-Bulk for reduced formula $X$. For $n(X) = 1$, the upper bound is loose; for $n(X) \gg 10$, the bound is tight.
3. **Stoichiometric Grid Distance**:
   The Euclidean distance between the formula's composition vector and the nearest vector in the discrete set of enumerated prototype ratios $\mathcal{R}_{\text{sampled}}$.
4. **Electronegative Element Fraction**:
   Fraction of oxygen and halogen atoms in the formula:
   $$x_{\text{O,X}} = \frac{N_{\text{O}} + N_{\text{F}} + N_{\text{Cl}}}{N_{\text{total}}}$$
   quantifying distance from Materials Project's primary search basin.
5. **Continuous Wyckoff Degrees of Freedom**:
   Count of unconstrained internal coordinates $(x, y, z)$ across the populated Wyckoff positions. A high count indicates structural complexity that standard fixed-coordinate prototype substitution algorithms systematically skipped.
