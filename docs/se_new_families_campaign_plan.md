# Plan: generative search for new Li solid-electrolyte families

> **Status: plan, 2026-10-25. Nothing here has been run.** It follows
> [the families campaign](solid_electrolyte_campaign.md), which only searched systems where
> Li solid electrolytes (SEs) are already known. Every count below that describes data
> was measured on hera on 2026-10-25 from the `lemat_bulk_fixed_fmax1_stress` cache. Cost
> figures are either taken from the families campaign or quoted from the papers cited.
> Thresholds marked **gate** are fixed here, before any result exists.

## The question, and why the families campaign cannot answer it

The families campaign generated crystals only in the 126 chemical systems where Li SEs are
already known. Inside those systems, "it might be an electrolyte" was assumed rather than
tested. To find a new family, the search has to leave those systems. That removes the
assumption, so something else has to supply the evidence: a transport screen that knows
no families. Without one, the same pipeline returns stable Li insulators, and most of those
conduct Li poorly.

This plan therefore has three parts:
1. a test of whether a family-blind transport screen can rediscover known families it was not
   tuned on;
2. the database first, then generation outside the 126 systems;
3. the transport stage the families campaign left out: PES descriptors and MLIP-driven MD.

That stage already exists in the ICML pipeline
([archive/101_Generative_Pipeline_for_Di.pdf](archive/101_Generative_Pipeline_for_Di.pdf),
§3.4) and is reused here.

**What counts as a new family.** Take a relaxed structure's framework: the structure with
Li removed and every other species anonymised, as Jun et al. 2022 define it. A candidate is
a **new framework** if that framework matches no known SE. Known SEs are the OBELiX CIFs,
the LeMat-Bulk entries at an OBELiX formula, and the families campaign's named prototypes.
New frameworks that match each other form a **candidate family**. The report is organised
by candidate family, not by composition. A known framework in new chemistry is reported
separately as "known family, new chemistry".

## What the literature offers, and what a gene can carry

The descriptor survey and its gene-versus-structure classification are summarised here; the
sources are under [References](#references).

| class | examples | where they can be used |
|---|---|---|
| exact from the gene | Li content, anion identity and polarizability, Li site symmetry, number of Li orbits, composition-only classifiers | gene screen |
| exact from the gene + energy model | electrochemical window (Richards 2016: grand potential; anodic limit set by the least stable Li binary) | gene screen (window only reported, not screened on) |
| structure, coarse | corner-sharing or isolated framework (Jun 2022), anion framework bcc/fcc/hcp (Wang 2015), volume per anion, Sendek 2017's five-feature model, Li polyhedral distortion | after relaxation; gene-level surrogates if [Phase 3](#phase-3--gene-level-surrogates) passes |
| structure + energy model | Ξ from frozen-framework PES (Maevskiy 2025), empty-framework percolation barrier (ICML §3.4), BVSE barrier, phonon band centres (Muy 2019), lattice stiffness (Kraft 2017; Rodin 2025) | after relaxation |
| beyond one ordered structure | Li-sublattice frustration (Kozinsky 2016), high-entropy site-energy overlap (Zeng 2022), doping | Li-stuffed variants in [Phase 7](#phase-7--dynamics) only |

A gene almost never fixes the anion framework. Among the 49,148 Li compounds in
`lemat_bulk_fixed_fmax1_stress` within 0.1 eV/atom that contain O, S, Se, Te, F, Cl, Br, I
or N:
- every anion sits on a site with no free coordinate in only 2.1% of them (0.3% cubic);
- every Li does in 38%;
- the median gene has 8 free coordinates.

So the gene screen can carry composition, symmetry, window and gap. Anything about the Li
path needs either a structure or a trained surrogate.

## Resources

- **hera:** Xeon w7-3455 (24 cores) and 2× RTX 6000 Ada. This is exactly the machine
  Maevskiy et al. 2025 used, so their timings carry over for their MLIP: 5,997 PES scans in
  7 days with M3GNet, about 1.7 min per structure. ORB ran at 72k–157k relaxed atoms/h in
  the families campaign. hera runs generation, the band-gap model, ORB relaxation, the PES
  scans and MD.
- **iapetus:** 6 cores, 2× K20c, about 15k ORB atoms/h. It takes the small protocol runs
  under its existing launcher (≤ 60,000 atoms), as in the families campaign. It is also the
  CPU lane for the batch structure descriptors (BVSE, ChemEnv/CSM, framework matching),
  which are embarrassingly parallel and need no GPU.

## Phase 0 — Data and code

| item | source | use |
|---|---|---|
| OBELiX CIFs (~320 of 599 entries) | NRC-Mila/OBELiX at the pinned commit `4eaac88`; the store has only `all.csv` so far | experimental labels |
| Kahle 2020 AIMD diffusivities | Materials Cloud archive | simulated labels (Maevskiy's `KAHLE2020`) |
| LiTraj `BVEL13k` | `litraj` (PyPI) | checks our BVSE numbers against published ones |
| Maevskiy's published Ξ for 5,997 MP structures | constructor.app `pes_fingerprint` project | checks our Ξ; 21% of the palette compounds below have MP ids |
| LeMat-Bulk-fixed structures | `data/lemat-bulk/lemat_pbe.csv.gz` CIFs, joined on `immutable_id` | benchmark and mining set |

**Code to ask the ICML co-authors for before writing any:**
- the Ξ PES scan (Maevskiy);
- the empty-framework percolation barrier (Dembitskiy, who also wrote `bvlain`);
- the MD protocol of ICML §3.4.

Reimplementing them is the fallback. The published Ξ values and `BVEL13k` are what a
reimplementation is checked against.

**New code:**
- `scripts/se_new_families_campaign.py`, built on `scripts/solid_electrolyte_campaign.py`,
  with stages `benchmark`, `mine`, `surrogate`, `systems`, `generate`, `bandgap`, `select`,
  `relax`, `transport`, `md`, `report`;
- `yamls/campaigns/se_new_families_hera_<date>.yaml`;
- runs store and W&B run `se-new-families-hera-<date>`.

**Two screens replace their family-bound versions:**
- *Anion roles.* `FAMILY_ANIONS` is defined per family, so it cannot be used outside the
  families. The replacement has two stages:
  - Gene stage: halogens must be negative. N, S, Se and Te may be positive only when O or F
    is present (sulfates and selenites are allowed; LiGa(SeO3)2 is Jun 2022's hit). This
    excludes the N–Cl and S–I molecular solids and the halogen oxoanions the families
    campaign's report had to remove.
  - After relaxation: bond-valence oxidation states (`BVAnalyzer`), which can tell a
    phosphate P(+5) from a phosphide P(−3).
- *Known SEs.* The formula-level OBELiX check stays. A framework match against the known-SE
  set is added after relaxation.

## Phase 1 — Benchmark the transport screen (decision gate)

**Set.** 12,383 of the 49,148 compounds have every element in the palette below; together
they have 0.39M atoms in their conventional cells, with a median of 24. Put that set and the
benchmark structures through one ORB relaxation (`orb_conserv_inf`). That makes them the
same kind of structure the generated candidates will be.

**Labels.**
- *Experimental, primary label at the framework level.* An OBELiX framework group is fast if
  any member has σ_RT ≥ 1e-4 S/cm, and poor if every member has σ_RT ≤ 1e-7 S/cm. A
  palette compound inherits the label of the OBELiX framework its own framework matches;
  unmatched compounds stay unlabelled. Framework-level labels are needed because garnets,
  NASICONs and LLTO conduct only when doped. Their ordered stoichiometric parents in LeMat
  conduct poorly, and a formula-level label would mark their frameworks negative.
- *Simulated.* Kahle 2020 at 1000 K: positive at σ ≥ 0.1 S/cm, negative at ≤ 0.01 S/cm.
  These are Maevskiy's thresholds.

**Descriptors.**
- **Gene-level baselines:** Li fraction, anion class, Li site-symmetry statistics, and a
  composition-only classifier fitted leave-one-family-out. These measure how much the gene
  alone knows.
- **Coarse structure descriptors:** corner-sharing/isolated framework; anion framework
  bcc/fcc/hcp; volume per anion; Sendek's logistic model with its published coefficients,
  not refitted; Li polyhedral distortion; topologically isolated anion groups.
- **Energy-model descriptors:**
  - Ξ, computed twice: with the original MLIP, to check it reproduces the published values,
    and with ORB;
  - the empty-framework percolation barrier;
  - the BVSE 1D/2D/3D percolation barrier;
  - exploratory: ρ·v_s³ from ORB elastic constants. Rodin 2025 derive a drag ∝ 1/(ρ·v_s³)
    and Kraft 2017 measured opposite effects of softening on barrier and prefactor; together
    they predict an optimum stiffness rather than "softer is better".

**Cost.** Ξ for all 12,383 compounds would take about 15 days at M3GNet's rate. So:
1. BVSE and the coarse descriptors run on all 12,383;
2. Ξ runs on the labelled structures, the top 3,000 by BVSE barrier, and a random control
   sample of 500.

The ORB rate for Ξ is measured on the first 100 structures before the full batch is sized.

**Metrics.**
- Partial ROC AUC over the lowest 10% of false-positive rate, as Maevskiy et al. use, with
  bootstrap intervals.
- *Rediscovery.* For each OBELiX family with at least five matched fast members, the
  percentile ranks of those members among the 12,383 compounds.
- Any combination with fitted weights is fitted leave-one-family-out.

**Calibration of the later stages, run here.**
- *Framework matcher.* Relax 20 known-SE genes from their genes with the protocol, then
  check that the matcher maps each back to its family. The families pilot found that
  structure matching misses known phases after this relaxation (three of four controls were
  scored novel), so the tolerances must be set on this test, not assumed.
- *MD.* Run the [Phase 7](#phase-7--dynamics) protocol on about 20 reference compounds:
  - fast: Li6PS5Cl, Li10GeP2S12, Li7P3S11, Li3YCl6, Li3InCl6, β-Li3PS4, LiTaOCl4, Li3N,
    LiBH4;
  - poor: LiF, LiCl, Li2O, Li3PO4, Li4SiO4, LiAlO2, Li2SO4, LiAlCl4, and the ordered
    parents of the garnet and NASICON families.

**Gates.**
- **G1, transport screen.** A descriptor, or a leave-family-out combination, enters the
  post-relaxation screen only if:
  - its pAUC interval excludes random on both label sets, *and*
  - the median fast member lands in the top 10% for at least half of the eligible families.

  If nothing passes, stop and write up the negative result. Generation would then only
  produce more stable Li insulators.
- **G1b, framework matcher.** At least 18 of the 20 relaxed controls are matched to their
  family.
- **G1c, MD.** ORB MD separates the fast references from the poor ones with AUROC ≥ 0.8. If
  not, switch the MD potential to TensorNet-MatPES, as ICML §3.4 did, and re-run the
  calibration.

## Phase 2 — Mine the database

Take the 10,725 palette compounds outside the 126 campaign systems; they span 2,464 systems.
1. Keep those with a gap ≥ 0.5 eV: MP's GGA gap where the entry has one, `3t16iys0`'s
   prediction otherwise.
2. Apply the generic anion roles and the coordination check of
   [Phase 6](#phase-6--structure-screens).
3. Remove known-SE frameworks.
4. Rank by the G1 descriptors and cluster the survivors by framework.
5. The top clusters go to [Phase 7](#phase-7--dynamics) as they are.

These compounds already exist, so no reconstruction can fail here. LeMat's ICSD provenance
(`lemat_bulk_fixed_fmax1_stress_icsd`) separates the experimentally reported ones, which Jun
2022 also required. The clusters also seed the generation arm of
[Phase 4](#phase-4--generation).

## Phase 3 — Gene-level surrogates

Label the 49,148 ionic Li compounds with the coarse descriptors that passed G1:
- the corner-sharing/isolated framework, a binary label;
- the anion framework class;
- volume per anion.

Also label the 12,383 palette compounds with the BVSE barrier. Then train gene-level models
like `3t16iys0`, with train and test split by chemical system.

**Gate G2.** A surrogate is used as a gene screen only if:
- its system-grouped test score is AUROC ≥ 0.80 for a binary label or Spearman ≥ 0.6 for a
  continuous one, *and*
- at its threshold it keeps ≥ 80% of the fast-framework members.

Otherwise the gene screen carries no transport term. The surrogates describe the gene's
symmetric structure. 1,247 of the families campaign's 2,852 shortlisted free relaxations
ended in P1, so a surrogate can only ever pre-screen. The post-relaxation descriptors decide.

## Phase 4 — Generation

**Palette.** The ICML paper's 42 elements plus Er and Lu, which the halide family needs:
H, Li, Be, B, C, N, O, F, Mg, Al, Si, P, S, Cl, Ca, Sc, Ti, Cu, Zn, Ga, Ge, As, Se, Br, Sr,
Y, Zr, Nb, Ag, Cd, In, Sn, Sb, Te, I, Ba, La, Hf, Ta, W, Pb, Bi, Er, Lu.

**Model.** The families campaign's: WyFormer-LeMat-Chemsys at revision `5ebcbaf`, one exact
system per request.

**Systems.** All systems contain Li and have 3–5 elements. Excluded are the 126 campaign
systems and every exact element set of an OBELiX entry.
- *Seeded arm, about 150 systems:* the systems of the top Phase 2 clusters, plus each one with
  a single non-Li element swapped for a likely substitute within the palette (pymatgen's
  substitution probabilities).
- *Open arm, about 350 systems:* drawn with `wyformer-system-prior sample --required Li
  --allowed <palette> --min-arity 3 --max-arity 5`, from the prior built on the model's
  own training cache. It uses the empirical p(system) and the default novel-system fraction.
  The conditioning study found MetaSUN highest in crowded systems (2.6× at ≥ 317 entries),
  and the open arm needs that competence more than it needs adventure.
- If Phase 2 gives fewer than 30 surviving clusters, the open arm takes the rest of the
  budget.

**Draws.** 5,000 per system, 2.5M in all, the same as the families campaign's 2.52M. At the
families campaign's yields, 5,000 draws give about 1,600 unique genes in the exact system
and about 170 that pass charge balance and the gap screen. That is enough for the selection
below.

**Gene screens, in order:**
1. exact system;
2. gene novelty against `lemat_bulk_fmax1_stress`'s key table, the protocol's own novelty
   reference;
3. OBELiX formula;
4. single-valence charge balance with the generic anion roles;
5. gap ≥ 0.5 eV (`3t16iys0`);
6. G2 surrogates, if any passed.

The window is computed by Richards' method and reported. Its useful range depends on whether
the material is meant as a bulk electrolyte, an anolyte or a catholyte, so it is not a
screen.

**Selection.** Rank by WyFormer-LeMat-Energy's predicted e_hull. Keep the top 60 genes per
system, at most 10 per formula and 3,000 atoms per system, and no gene over 200 atoms;
PES scans and MD cost grow with the cell. That is at most 30,000 genes and 1.5M atoms. The
families campaign relaxed 28,851 genes and 1.19M atoms in about 30 h on hera and iapetus.

**Pilot first.** Run 20 open-arm systems through relaxation. **Gate G4:** at least 15% of
the relaxed genes are within 0.1 eV/atom of the hull; the families campaign had 64%. Below
that, the open arm moves to more crowded systems by lowering `--system-temperature` and the
novel fraction, and the pilot is repeated. The conditioning study measured structures 0.24
eV/atom above the hull where the model was taught little, so this gate is a real risk.

## Phase 5 — Relaxation and hull

This is unchanged from the families campaign: the protocol's wide NEP89 → ORB arm, with
e_hull measured against the `orb_conserv_inf` split of LeMat-Bulk-MLIP-Hull. hera and
iapetus run in lanes as before (`relax --cpu-lanes 2 --cpu-cores 12 --remote-max-atoms
60000`).

## Phase 6 — Structure screens

These run in order, on the free-relaxation track:
1. e_hull ≤ 0.1 eV/atom. In the pilot, known phases relaxed from their own genes sat
   0.03–0.045 eV/atom above this hull.
2. Ionic under `BVAnalyzer`.
3. **Coordination check.** Every Li and framework-cation environment (element, CN,
   polyhedron) must occur in `coordination_motif_screen`'s frequency tables. The ICML
   candidates included square-planar LiO4 and trigonal-planar LiF3. The rarity cutoff is set
   from the Phase 2 known compounds so that it removes ≤ 1% of them.
4. Known-SE framework exclusion, with the tolerances calibrated at G1b; then clustering by
   framework.
5. **Join flag.** A composition on a pseudo-binary join is flagged if LeMat holds no
   intermediate compound on that join; it is not removed. The ICML paper's LiLaCl4 and
   Li3MgF5, and the families campaign's ternaries, lie on such joins.
6. The descriptors that passed G1, as an OR of the frozen-framework and empty-framework
   arms, as in ICML §3.4. Thresholds are re-fitted on ORB in Phase 1, because ICML §3.4
   notes the domain shift between generated structures and Ξ's tuning data.

## Phase 7 — Dynamics

**Protocol (ICML §3.4).**
- A supercell of ≥ 100 atoms that holds a 10 Å sphere.
- 10 ps of thermalisation, then 100 ps of NVT production at 500 K and 1000 K.
- Diffusion from 15 ps windows, as in Kahle 2020.

The ICML paper did this with TensorNet-MatPES and a Nosé–Hoover thermostat. Here ORB is used
if it passed G1c; Langevin BAOAB is already in `scripts/cu_ge_te_switching/s4_md.py`.

**Filters.**
- The slowest non-Li species at 1000 K must have MSD ≤ 2 Å² over 100 ps; otherwise the
  framework melts or decomposes.
- Li must reach MSD > 4 Å² at 500 K.
- σ_300K is extrapolated with a pessimistic E_a of 0.5 eV, and σ_300K − 1 s.e.m. is reported
  in two tiers: ≥ 1e-6 S/cm (coating) and ≥ 1e-4 S/cm (electrolyte).

**Scope.**
- Up to 3 structures per candidate family, plus the Phase 2 clusters.
- **7b, doping.** For the ten best new frameworks, add one Li-excess and one Li-deficient
  variant made by aliovalent substitution, as Jun 2022 did. This is the only place the plan
  reaches the class beyond one ordered structure.
- **MD is used for ranking only.** Universal MLIPs soften the PES and underestimate barriers
  (Deng 2025).

The MD cost per structure is not known for ORB at this length; it is measured on the G1c
references first.

**Optional.** Rodin 2025's time-local equation of motion has so far been validated only on a
model system. If the co-authors want to try it as a cheaper step between Ξ and MD, the G1c
references are the test set.

## Phase 8 — Report and hand-off

The report is organised by candidate family. Each family has:
- its framework and members;
- each member's free and fixed-symmetry e_hull;
- the descriptors and MD σ tiers;
- the window;
- the join flag and the coordination flags.

Logged to W&B as artifacts, as the families campaign was. Then, in order:
1. an ICSD and literature check of every family;
2. PBE/r2SCAN relaxation;
3. AIMD for the best ~10 members, by the co-authors, as for the papers' validation;
4. electrochemical windows and interface reactions against the intended electrodes, by
   Richards' method.

## Gates and order of work

| gate | after | passes if | otherwise |
|---|---|---|---|
| G1 | Phase 1 | a descriptor's pAUC interval excludes random on both label sets, and it rediscovers at least half of the held-out families | stop; write up the negative result |
| G1b | Phase 1 | ≥ 18/20 relaxed controls matched to their family | re-tune the matcher before any exclusion |
| G1c | Phase 1 | MD AUROC ≥ 0.8 on the references | switch the MD potential |
| G2 | Phase 3 | system-grouped AUROC ≥ 0.80 or Spearman ≥ 0.6, ≥ 80% recall | no gene-level transport screen |
| G4 | Phase 4 pilot | ≥ 15% of relaxed genes ≤ 0.1 eV/atom | move the open arm to crowded systems; repeat the pilot |

Phases 1–3 need no generation and can start now. The descriptor code is the long pole and
depends on the co-authors' code. Generation and relaxation take about two days at the
families campaign's rates. Phase 7's length is unknown until G1c measures it.

## References

Local copies are in [archive/](archive/).
- Dembitskiy, Yamazaki, Maevskiy, Kazeev et al., ICML 2026 AI4Science workshop:
  [archive/101_Generative_Pipeline_for_Di.pdf](archive/101_Generative_Pipeline_for_Di.pdf).
- Maevskiy et al., Phys. Rev. Research 7, 023167 (2025):
  [archive/PhysRevResearch.7.023167.pdf](archive/PhysRevResearch.7.023167.pdf).
- Rodin, Olsen, Ustyuzhanin, Maevskiy, Phys. Rev. Research 7, 033120 (2025),
  doi:10.1103/jnzr-q953: [archive/jnzr-q953.pdf](archive/jnzr-q953.pdf).
- Kraft et al., J. Am. Chem. Soc. 139, 10909 (2017), doi:10.1021/jacs.7b06327:
  [archive/ja7b06327.pdf](archive/ja7b06327.pdf).
- Richards et al., Chem. Mater. 28, 266 (2016), doi:10.1021/acs.chemmater.5b04082:
  [archive/acs.chemmater.5b04082.pdf](archive/acs.chemmater.5b04082.pdf).
- Sendek et al., Energy Environ. Sci. 10, 306 (2017), doi:10.1039/C6EE02697D.
- Wang et al., Nat. Mater. 14, 1026 (2015).
- Jun et al., Nat. Mater. 21, 924 (2022), doi:10.1038/s41563-022-01222-4.
- Jun, Lee, Kam, Ceder, PNAS (2024), "The nonexistence of a paddlewheel effect in
  superionic conductors": [PMC11067015](https://pmc.ncbi.nlm.nih.gov/articles/PMC11067015/).
- Muy et al., iScience 16, 270 (2019): [PMC6581664](https://pmc.ncbi.nlm.nih.gov/articles/PMC6581664).
- Zhang et al., Nat. Commun. 10, 5260 (2019), doi:10.1038/s41467-019-13214-1.
- Kahle, Marcolongo, Marzari, Energy Environ. Sci. 13, 928 (2020), doi:10.1039/C9EE02457C.
- He, Zhu, Mo, Nat. Commun. 8, 15893 (2017), doi:10.1038/ncomms15893.
- Di Stefano et al., Chem 5, 2450 (2019), doi:10.1016/j.chempr.2019.07.001.
- Kozinsky et al., Phys. Rev. Lett. 116, 055901 (2016), doi:10.1103/PhysRevLett.116.055901.
- Zeng et al., Science 378, 1320 (2022).
- Laskowski, McHaffie, See, Energy Environ. Sci. 16, 1264 (2023), doi:10.1039/D2EE03499A.
- Hargreaves et al., npj Comput. Mater. 9 (2023):
  [Liverpool repository](https://livrepository.liverpool.ac.uk/3169593/).
- Therrien et al., OBELiX, arXiv:2502.14234; Digital Discovery (2026).
- Deng et al., npj Comput. Mater. (2025), doi:10.1038/s41524-024-01500-6.
- Dembitskiy et al., LiTraj, npj Comput. Mater. (2025): [`litraj`](https://pypi.org/project/litraj/).
- Chen, Wong, Adams, softBV, Acta Cryst. B (2019):
  [author copy](https://www.dmse.nus.edu.sg/asn/download/Chen_wong_Adams_Acta_Cryst_B_2019.pdf).
