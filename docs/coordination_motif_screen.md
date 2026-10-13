# How rare are the "unprecedented coordination motifs"? A screen of `lemat_bulk_fmax1_stress_ehull01`

**Date:** 2026-09-27. **Branch:** `coordination-motifs`. **Code:**
`src/wyckoff_transformer/evaluation/coordination_motifs.py` (measurements),
`scripts/screen_coordination_motifs.py` (shard / run / report), tests in
`src/wyckoff_transformer/tests/test_coordination_motifs.py`.
**Outputs:** `$WYFORMER_RUNS/coordination_motifs/` =
`/scratch/users/nus/kna/WyFormer/runs/coordination_motifs/`: `report.md` (the full
generated report), `environment_frequency.csv` (species, CN, polyhedron),
`sharing_frequency.csv` (polyhedron pair, sharing mode), and per-shard
`results/*.{structures,sites,pairs}.parquet`.

The rules come from `docs/ideas/Theoretical & Crystallographic Rules Defining
Unprecedented Coordination Motifs.md` (untracked in the main checkout), as amended
there: rule 1 marked weak; rules 3 & 4, LFSE and SOJT restated.

## What was run

All 1,637,402 structures of `data/lemat_bulk_fmax1_stress/{train,val,test}.csv.gz`
with `energy_above_hull <= 0.1`. That is exactly the row count of the cached
`lemat_bulk_fmax1_stress_ehull01`. The run took about 25 minutes on the 16 cores of one
`gdev` job, so no PBS submission was needed.

| status | structures |
|---|---|
| no anion (intermetallics etc.) | 1,040,532 |
| **in scope**: ionic, one integer oxidation state per element | **381,369** |
| no single-valued oxidation-state assignment (mixed valence or unassignable) | 191,543 |
| assignment not ionic (an anion element positive, a zero state) | 23,955 |

Oxidation states are the top ICSD-prior guess of the vendored LeMat-GenBench routine.
Every rarity below is conditional on that guess being right. Where it is visibly
wrong (chalcogenides with anion–anion bonds, actinides), the hits are dominated by
misassignments. That is why most sections also report an **O/F-only** count.

**Method, briefly.** Bonds are anions whose distance divided by the pair's O'Keeffe–Brese
R0 is within 25% of the shortest. The first pass used raw distances. It truncated
mixed-anion polyhedra (O4Cl2 read as square-planar O4) and was discarded;
`results_v1_raw_distance/` keeps it. Shapes come from comparing the sorted angle list
with ideal polyhedra. A site is *clean* when the next anion is ≥10% farther in d/R0
and the shape misfit is ≤10° RMS. Site point groups come from spglib at 0.1 Å, the
tolerance the Wyckoff genes were built with. Sharing modes come from counting the
anions shared by each pair of cation polyhedra. The dimensionality of a face-sharing
network is computed from the periodic quotient graph.

**Validation.** Cubic SrTiO3, rutile, zinc blende, PtS and corundum give the textbook
answers (unit tests). 6H-BaTiO3 (mp-5933) is flagged as having finite face-sharing
Ti⁴⁺ dimers. The face-sharing d⁰ list is headed by the known M2O9-dimer structures
Mg4Nb2O9, Mn4Nb2O9, Ba4Nb2O9 and Mg4Ta2O9.

## Results

| motif (doc section) | in-scope structures with it | verdict |
|---|---|---|
| Radius-ratio CN wrong (1A) | 46% of cation sites off by ≥1 band, 6.4% by ≥2 | Rule 1 is weak, as amended |
| Some anion's Pauling sum off by >35% (1B) | 8.8% | Common. Mostly oxidation-state or cutoff issues, not novelty |
| Edge-sharing tetrahedra, z ≥ +4, O/F (1C) | **18** (24 of 60,976 tet–tet links, 0.04%) | Very rare |
| Face-sharing tetrahedra, z ≥ +4, O/F (1C) | **0** | Absent |
| Rule 4 proper: different z ≥ +4, CN ≤ 4 cations sharing edge/face (1C) | **0** | Absent |
| Face-sharing d⁰ z ≥ +4 octahedra (1C) | 109 (0.5% of d⁰ oct–oct links) | Known, all finite |
| … extended (1D/2D/3D) face-sharing d⁰ networks | **0** of 120 face-sharing networks | Absent; confirms the amended claim |
| Tetrahedral Cr³⁺, O/F, clean (2A) | 7 of 1,788 clean Cr³⁺ sites | Very rare |
| Tetrahedral Mn⁴⁺, O/F, clean (2A) | 18 of 1,218 | Rare but real (orthomanganates(IV): K4MnO4, Ba3MnO5) |
| Tetrahedral/prismatic Pd²⁺/Pt²⁺/Au³⁺, O/F (2A) | **0** tetrahedral, 1 prismatic | Absent; the 1,218 non-O/F "hits" are misassigned chalcogenides/halides |
| Square-planar d⁰, z ≥ +4, O/F (2A) | **1** (mp-759588 Cr(Bi7O12)2, borderline shape) | Essentially absent in oxides. In nitrides, forced by 4/m site symmetry (Hf4ZrN4 family) |
| SOJT cation on a non-polar site, O/F (2B) | Mo⁶⁺ 74%, V⁵⁺ 62%, Ti⁴⁺ 27%, Nb⁵⁺ 31% of octahedral sites; lone pairs 3–28% | **Not rare**; see below |
| HSAB inversion (3) | **2** (Cs2Pt(IF2)2, Rb2Pt(IF2)2, Alexandria) of 6,598 mixed hard/soft-ligand structures | Nearly absent |
| 3D corner-sharing perovskite with t < 0.70 or t > 1.15 (4) | **2** (TlCrO3, t = 1.16; Tl⁺/Cr⁵⁺ assignment doubtful) | Nearly absent. No ABX3 at all has t < 0.70 |
| ≥ 10 Wyckoff orbits of one element (rule 5, gene level) | 2.8% | Common enough to be a weak signal |

### Reading the SOJT row

Symmetry pins a strong-SOJT d⁰ octahedron (Mo⁶⁺, V⁵⁺) most of the time. That does not
mean dynamically stable centrosymmetric Mo⁶⁺ is common. The pinned fraction is 34% in
Alexandria and 37% in OQMD, against 13% in MP, and the examples (LiVF6, KVF6, TlVF6,
Ca2VSbO6) are Alexandria prototypes. The stored structures are exactly symmetric: no
pinned site is displaced by more than 0.05 Å. So these are symmetry-constrained
relaxations, which cannot break inversion. Without phonons they cannot be told apart from
symmetric saddles (see `docs/orb_hull_rattle_report.md`). The gene-level flag is still
the right *preselection* (1,671 pinned sites in MP alone, all cations of the table),
but its rarity is decided by phonons, not by this screen.

### What the numbers say about the doc

* The motifs the amendments pointed at are genuinely rare in the 381k in-scope
  near-hull structures: face-sharing high-valent tetrahedra (0), Rule-4 hetero-sharing
  (0), extended face-sharing d⁰ octahedra (0), tetrahedral low-spin d⁸ in oxides (0),
  square-planar d⁰ oxides (≈1), HSAB inversion (2), and out-of-range corner-sharing
  perovskites (2, both doubtful). A generated structure with any of these, confirmed
  on or near the hull with real phonons, would be a result.
* Tetrahedral Mn⁴⁺ is not a target (18 clean O/F sites), while tetrahedral Cr³⁺ is
  (7). The doc should keep Cr³⁺ and drop Mn⁴⁺ from the d³ list.
* Most of the few existing hits are Alexandria hypotheticals: 16 of 18 edge-sharing
  tetrahedra, both HSAB inversions, and 334 of 371 tetrahedral or prismatic d³. They are
  candidates to check, not precedent.
* Gene-level signals: site symmetry forces the non-classical geometry for only 10 d³
  sites (0.13%). It rules out the tetrahedron for 24% of d⁰ sites, but those are almost
  all octahedra. The gene is therefore a useful necessary-condition filter for SOJT
  (non-polar sites), cheap for Goldschmidt t and HSAB composition, and nearly useless
  for LFSE and sharing, which need the relaxed structure.

## Caveats

* The rarity is measured in DFT databases, not ICSD. Alexandria dominates the in-scope set (298k of 381k).
* Out of scope: mixed-valence compounds (192k structures), intermetallics, and any
  motif that depends on an oxidation state the guesser gets wrong. The O/F-only
  columns are the trustworthy ones.
* The shape assignment by sorted angles and the 25% d/R0 cutoff are simple. A hit
  should be confirmed with ChemEnv (continuous symmetry measures) before it is quoted.
* "Unprecedented" here means rare in this dataset. The frequency CSVs give the
  per-species baseline for judging a generated structure.

## Reproduce

```bash
cd /home/users/nus/kna/scratch/WyFormer/worktrees/coordination-motifs
OUT=$WYFORMER_RUNS/coordination_motifs
R=scripts/platforms/aspire2a/run_in_singularity.sh
bash $R python scripts/screen_coordination_motifs.py shard  --out $OUT   # ~5 min, one pass over the CSVs
bash $R python scripts/screen_coordination_motifs.py run    --out $OUT --workers 16   # ~25 min on 16 cores
bash $R python scripts/screen_coordination_motifs.py report --out $OUT
```

`run` is resumable per shard and splits across processes with `--shard-mod I N`.

## Strategy measurements (2026-09-28)

The goal is a handful of structures striking enough for a materials-science journal.
Genes are cheap and structures are not, so the plan is to target motifs a **gene
guarantees** by symmetry and to spend structures only on stability. Three
measurements decide whether that works: M1, M3 and M4. M2, the energy predictor on
rare motifs, was dropped because the predictor is at best a 2× speed-up.

### M1: motifs in generated pools that have already been relaxed

`scripts/screen_generated_pools.py`, run on 14 protocol pools
(`runs/chemsys_study/artifacts/protocol_*`, the torpedo joint and reference pools,
`generated/gene_min_formation_energy…`). It leaves out the contaminated
`protocol_ehull5x-20260904-213346`, the `generated/` copy of the ehull_5x pool and the
10-gene pilots. Output is in `runs/coordination_motifs/pools/`. Energies and novelty
come from each pool's own ORB hull and `novel_structure`.

13,969 relaxed structures; **1,825 (13%) are in scope** (the rest are 9,375
intermetallics, 2,394 without a single-valence assignment, and 375 non-ionic). 1,574
are novel, and 553 are novel with e_hull ≤ 0.1.

| motif | structures (all novel) | e_hull ≤ 0.1 |
|---|---|---|
| edge-sharing tetrahedra, z ≥ +4, O/F | 6 (0.33% of in-scope vs 0.005% in the database) | 2: Hf3K12Ge12O36 (0.018), Rb4P4O12 (0.071) |
| face-sharing tetrahedra; Rule 4 hetero-sharing | 0; 1 (K2U2C8H22O48Se12, 0.52) | 0 |
| face-sharing d⁰ octahedra (any dimensionality) | 0 | 0 |
| tetrahedral Cr³⁺; tetrahedral/prismatic ls-d⁸; square-planar d⁰ (O/F) | 0 | 0 |
| HSAB inversion | 1 (PtAsIO2, 0.77) | 0 |
| out-of-range corner-sharing perovskite | 0 | 0 |
| strong-SOJT octahedron on a non-polar site | 2 | 1: CeDy2MoZn2O10 (0.096) |

**Reading.** Unconstrained generation rarely reaches the motifs, as expected. The one it
over-produces, edge-sharing high-valent tetrahedra (70× the database rate), is the one
Pauling's rules predict to be costly. The two near-hull cases need DFT before they
mean anything: an ORB e_hull of 0.02–0.07 eV for edge-sharing PO4 would be remarkable
if it held. Only 13% of generated structures are even ionic enough to judge, so a
motif campaign has to steer the composition (chemical-system conditioning or an
element palette), not just filter.

### M3 and M4: 2 × 1M generated genes

`scripts/platforms/aspire2a/generate_gene_pool.pbs` (jobs 25584285 and 25584286,
2026-09-28) drew 1M genes per arm from `ehull_adamw_wsd_5x_cfg-20260924-223159` at
e_hull = 0.05, with guidance w = 1 and w = 3. `scripts/screen_gene_motifs.py` then
computed keys and flags for the pools and for all 5,327,342 LeMat-Bulk genes
(`lemat_bulk_fmax1_stress`, all e_hull). Outputs are in
`runs/coordination_motifs/{genes,gene_screen}/`, with the report in
`gene_screen/report.md`.

The gene keys are checked against the protocol's own `gene_novel` on the e9ywwsie
pool: 997 of 999 agree. The prototype key is the strict AFLOW fingerprint: space
group plus the Wyckoff orbits of each element, with elements anonymised and
equivalent settings merged.

**M4: new structure types.**

| arm | unique genes | novel genes | novel prototypes | distinct novel prototypes |
|---|---|---|---|---|
| w = 1 | 865,602 | 67.5% | 19.8% | 77,191 |
| w = 3 | 759,908 | 75.8% | 30.4% | 101,969 |

Guidance raises prototype novelty as well as gene novelty, but novelty is almost
entirely a matter of cell complexity. For w = 3, genes with ≤ 4 Wyckoff sites are
new structure types 5–8% of the time; 4–6 sites, 27%; 6–10 sites, 62%; above 10
sites, 82–98%. The archive holds 46,199 prototypes (25,764 near the hull). New
structure types are therefore cheap at gene level. They mostly come with large
cells, where reconstruction recovers only about 13% (`docs/de_novo_search_ideas.md`).

**M3: gene-guaranteed motifs, per unique gene.**

| flag | archive, all | archive, e_hull ≤ 0.1 | w = 1 | w = 3 |
|---|---|---|---|---|
| 2H face-sharing d⁰ chain | 0 | 0 | 0 | 0 |
| perovskite, t outside [0.70, 1.15] | 49 | 3 | 6 (cubic CsMO3; 4 already in the archive) | 0 |
| Cr³⁺ on a site with no octahedron | 17 | 0 | 2 (e.g. Rb6Cr2O6, Cr on −6) | 0 |
| d⁸ on a site with no square plane | 0.041% | 0.021% | 0.043% | 0.026% |
| strong SOJT (Mo⁶⁺, V⁵⁺) on a non-polar site | 0.046% | 0.029% | 0.030% | 0.011% |
| any SOJT d⁰ on a non-polar site | 0.31% | 0.40% | 0.22% | 0.11% |
| lone pair on a non-polar site | 0.34% | 0.34% | 0.35% | 0.22% |

The generator reproduces the archive's rates of gene-level motifs, and stability
guidance (w = 3) lowers them. The striking motifs (extended face-sharing d⁰, Cr³⁺
forced off the octahedron, perovskites with t out of range) are at or near zero in
2M genes, and the few hits are archive-style prototype decorations. **Caveat on the d⁸
flag:** "no square-planar site" still allows an octahedron, and octahedral Pd²⁺ is
known in fluorides (PdF2). The flag should require a site that also excludes the
octahedron before it counts as unprecedented. Many flagged d⁸ and SOJT genes are
actinide or mixed-lanthanide compositions whose oxidation states are doubtful.

**Consequence for the campaign.** Screening an unconstrained stream will not find the
headline motifs, however large it is. They have to be written as genes: 2H-type
ABX3 with d⁰ B across A/X palettes, Cr³⁺ on hexagonal-symmetry sites in oxides and
fluorides, and so on. The model then only scores how plausible each gene is, and
structures test stability. Unconstrained pools remain a good source of new structure
types of moderate size (4–10 sites) and of the weaker pinned-SOJT and lone-pair motifs.
