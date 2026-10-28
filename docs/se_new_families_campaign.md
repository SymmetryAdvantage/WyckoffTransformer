# Generative search for new Li SE families: the record

> **For a self-contained summary, read [se_new_families_results.md](se_new_families_results.md)
> first.** This page is the full record: methods, every table, commands and data paths.
>
> **Status, 2026-10-28:** Phases 1, 2 and 8 are complete.
> - G1b passes (18/20) and G1c passes (MD AUROC 0.957).
> - G1 passes formally for one descriptor, the empty-framework E_2D barrier, at the
>   margin. The pass rests on anion-only frameworks; on frameworks with cations no
>   descriptor passes ([G1](#gate-g1-the-transport-screen-bench-evaluate)).
> - Phase 2 (mining LeMat) screened and ranked 5,707 compounds
>   ([Phase 2](#phase-2-mining-lemat-mine-)). Of the 20 best clusters, 7 reach an MD tier
>   ([Results](#results)). The two in the electrolyte tier are Li4B4Sb2SO16, a computed MP
>   structure, and Li2HN, a known conductor that OBELiX lacks.
> - **A random control (2026-10-28) shows that E_2D's line enriches the MD tiers on
>   frameworks with cations.** 7 of 20 picks are tiered, against 2 of 40 random survivors
>   outside the line (one-sided Fisher p = 0.0044, fixed before the draw). The electrolyte
>   tier alone does not differ, 2/20 against 1/40
>   ([control](#random-control-for-the-md-tiers-mine-md-control)).
> - Phase 8's checks on those 7 clusters ran on hera without DFT
>   ([Phase 8](#phase-8-the-mined-candidates-phase8-)): literature and ICSD, structure
>   types, windows and interface reactions (Richards 2016, MP2020), joins, a structure
>   check and MD on further members.
>   - The one new-framework family in the electrolyte tier, the borate-sulfates
>     Li4B4Sb2SO16 and Li6Sn2B4SO16 (tychite-derived), is stable only from 3.4 and
>     2.1 V up. It is a high-voltage material, not an electrolyte against Li or
>     graphite. MD on the Sn member reached only the coating tier.
>   - MD on further members put a third structure in the electrolyte tier: Cmcm
>     Li4TiS4, in the Li4GeO4-type chalcogenide cluster (window 1.7–2.1 V). In its
>     lower Pnma polymorph Li does not move. Pnma Li4SnSe4, the measured polymorph,
>     lands near its measured 2e-5 S/cm.
>   - The other clusters are known compounds, known structure types or compositions in
>     unreported polymorphs, with narrow windows.
>   - No DFT or AIMD inputs are prepared (the user, 2026-10-28). The energies are
>     LeMat-Bulk-fixed's PBE ones, and the transport numbers come from ORB MD.
> - Phase 3 has no target, since no gene-level descriptor passed G1. Nothing has been
>   generated, and Phase 4 waits for the authors' decision on G1.
>
> This page records what was run and what it found. The plan, with its gates fixed
> before any result, is [se_new_families_campaign_plan.md](se_new_families_campaign_plan.md).
> Branch `se-new-families`. Campaign `se-new-families-hera-20261025`; config
> [`yamls/campaigns/se_new_families_hera_20261025.yaml`](../yamls/campaigns/se_new_families_hera_20261025.yaml),
> driver [`scripts/se_new_families_campaign.py`](../scripts/se_new_families_campaign.py),
> library `src/wyckoff_transformer/transport/`. Outputs are in the runs store under
> `se-new-families-hera-20261025/` and in W&B run
> [`symmetry-advantage/WyckoffTransformer/se-new-families-hera-20261025`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/se-new-families-hera-20261025)
> (uploaded 2026-10-27), as artifacts `se-new-families-hera-20261025-benchmark`, `-gates`
> and `-mining` (all v0). Everything up to that upload ran on uncommitted code on base `b31fb2d`;
> the run's config records `code_dirty: true`. Phase 8 is artifact `-phase8:v0`
> (uploaded 2026-10-27 13:19 UTC, uncommitted code on `8c0e0a3`). The random control
> and the Pnma polymorphs are `-mining:v1` and `-phase8:v1` (uploaded 2026-10-28
> 17:43 UTC, uncommitted code on `1fb5a24`); `-phase8:v1` no longer holds the withdrawn
> DFT and AIMD inputs.

```bash
PY=.venv/bin/python S=scripts/se_new_families_campaign.py
$PY $S data && $PY $S bench-set && $PY $S bench-relax
$PY $S bench-frameworks && $PY $S bench-describe
$PY $S bench-pes --pes-kinds frozen && $PY $S bench-pes --pes-kinds empty
$PY $S xi-check && $PY $S matcher-calibrate && $PY $S md-calibrate
$PY $S bench-evaluate
$PY $S mine-gap && $PY $S coordination-reference && $PY $S mine-screen
$PY $S mine-pes --devices cuda:0,cuda:1 --workers-per-device 1
$PY $S mine-rank && $PY $S mine-md && $PY $S mine-dimension && $PY $S upload
$PY $S mine-md-control    # 2026-10-28
```

## Phase 0: data

| input | where | pinned |
|---|---|---|
| OBELiX table and CIFs | `data/obelix/all.csv`, `all_cifs.tar.gz` (NRC-Mila/OBELiX @`4eaac88`, `data/downloads/`) | sha256 in the config |
| Kahle 2020 AIMD | `data/kahle2020/data.json`, the first member of the 21 GB AiiDA export (Materials Cloud doi:10.24435/materialscloud:vg-ya), fetched with an HTTP range request | sha256 in the config |
| LiTraj BVEL13k | `data/litraj/BVEL13k.zip` (litraj 0.1's URL) | sha256 in the config |
| LeMat-Bulk structures | `data/lemat-bulk/lemat_pbe.csv.gz`, joined on `immutable_id` | — |

- **OBELiX:** 599 entries. 321 have a CIF; only 72 of those are fully ordered.
  Conductivities below the measurable range are strings such as `<1E-10`; they are kept as
  upper bounds.
- **Kahle 2020:** `data.json` holds every node's attributes. Each node of the
  `diffusion_coefficients` group is traced to its trajectory (for the thermostat
  temperature) and to its supercell (for the Li density). The result is 200 records over
  121 structures: 122 at 1000 K, and 26 each at 500, 600 and 750 K. The diffusion
  coefficient is converted to σ by Nernst–Einstein. At 1000 K, 35 structures are fast
  (σ ≥ 0.1 S/cm) and 66 poor (≤ 0.01 S/cm), Maevskiy et al.'s thresholds. That matches
  their 121 materials.
- **Not obtained:** Maevskiy et al.'s 5,997 published Ξ values. The constructor.app
  project page is a JavaScript application with no data endpoint the driver can read. The
  ~39 values printed in the paper's tables (Tables I–II, Fig. 2) are the reproduction
  check instead (`xi-check`).
- **Co-authors' code** for Ξ, the empty-framework barrier and the MD protocol was not
  available to this run. All three are reimplemented from the papers, which is the plan's
  fallback (see [Transport descriptors](#transport-descriptors)).
- **Packages:** `bvlain` 0.25.1, `litraj` 0.1 and `ions` 0.5.1 were installed into the
  venv with `uv pip`; they are not yet in `pyproject.toml`.

## Phase 1: the benchmark set

`bench-set` reproduces the plan's counts exactly from `lemat_bulk_fixed_fmax1_stress`:
- 49,148 ionic Li compounds within 0.1 eV/atom;
- 12,383 with every element in the palette, 387,558 atoms in their conventional cells;
- 10,725 of those outside the families campaign's 126 systems, in 2,464 systems.

Every LeMat CIF agreed with the cache's composition, so none of LeMat-Bulk-fixed's 74
replaced structures is in the set. The benchmark adds Kahle's 121 cells and OBELiX's 72
ordered CIFs: 12,576 structures in all. Each is reduced to its primitive cell, 264,809
atoms, because the PES scans cost grows with the square of the cell.

`bench-relax` puts every structure through the protocol's own relaxation task
(`cli.protocol._relax_one`): orb_conserv_inf, BFGS, fmax 0.05 eV/Å, 500 steps per stage,
fix-cell warm-up, symmetric stage, free stage and rattle, each trial seeded by its key.
That is exactly how a generated candidate is relaxed. A 40-structure pilot relaxed 40/40
at ~0.15 worker-s per atom. The full run, with 4 workers on each GPU, relaxed all
12,576 with no failure in 65 min (2026-10-25, 11:14–12:19).

**OBELiX labels.** The 321 OBELiX CIFs give 95 framework groups: 20 fast, 53 poor and 22
unlabelled. Matching every relaxed structure against them labels 932 palette compounds
fast and 522 poor. The fast labels are lopsided:

| OBELiX family (fast) | palette compounds | framework |
|---|---|---|
| layered | 550 | XeRn2: CdI2-like MX2 layers |
| antiperovskite | 96 | anion-only (Rn) |
| NASICON | 78 | XeRn12/5 |
| nitrides | 70 | anion-only (Rn) |
| argyrodites | 50 | XeRn6 |
| halides | 39 | XeRn6 |
| LGPS | 31 | XeRn4 |
| garnet | 16 | Xe5Rn12 |

- The 550 layered compounds all inherit their label from one OBELiX entry,
  Li0.6[Li0.2Sn0.8S2] at 1.2×10⁻⁴ S/cm. The matches are structurally real: removing Li
  from an ordered layered rock salt leaves CdI2-like layers. What is fragile is the rule
  "fast if any member ≥ 10⁻⁴", which lets one marginal entry speak for 550 compounds.
- The anion-only frameworks of antiperovskites and of Li3N/Li3P are a single species
  matched with loose tolerances, so they carry little framework information.

The labels are framework labels, so the experimental pAUC weighs every OBELiX framework
group equally: each compound gets 1/(compounds its group labels). The unweighted,
per-compound pAUC is reported beside it. The PES scans take at most 40 labelled
compounds per group, a seeded sample.

## Transport descriptors

All are in `src/wyckoff_transformer/transport/`, with unit tests in
`tests/test_transport.py`.

**Frameworks** (`frameworks.py`). A structure's framework is the structure with Li
removed and every other species relabelled as cation or anion.
- Roles come from `BVAnalyzer` oxidation states when the structure is ordered. Otherwise
  they come from a composition rule: O and the halogens are anions; N, S, Se and Te are,
  unless O or F is present; without any of those, the most electronegative of P, As, Sb,
  Bi, C, Si, Ge, Sn, B and H is.
- On disordered sites, a site whose non-Li occupancy is at least ½ is a framework site;
  anything else is dropped with the Li.
- Frameworks are compared with `StructureMatcher` (`scale=True`, pymatgen's default
  tolerances) within the same framework formula. The tolerances are G1b's to set.

On OBELiX's 321 CIFs this gives 95 framework groups. Garnets (53 entries), argyrodites
(41), NASICON (33), LGPS (16) and LLTO perovskites (15) each come out as one group.
Li3YCl6 (P-3m1) and Li3InCl6 (C2/m) stay apart, as they should. One group mixes LiAlCl4
with LISICON/thio-LISICON tetrahedral frameworks.

**Frozen-framework PES** (`pes.py`, Maevskiy et al. 2025).
- One Li is removed and put back on a grid whose step along each lattice vector is at
  most 0.25 Å, skipping points within 1.2 Å of any atom. The rest of the structure is
  frozen.
- The PES, measured from its minimum, is tiled into the 2×2×2 supercell centred on that
  minimum, faces included.
- FV_d(0.5 eV) is the fraction of that supercell below 0.5 eV. FV_c(0.5 eV) is the
  fraction in the face-connected component of `E < 0.5 eV` that holds the minimum. MPE is
  the lowest threshold at which that component reaches the faces.
- Ξ = s(2.00 + log10 FV_c)·s(1.15 + log10 FV_d), with s(x) = 1/(1+e^(−10x)).
- Every Li contributes. The masks of the different Li, each centred on its own minimum,
  are OR-ed before the volume is taken, as the paper describes. To save cost, only one Li
  per spglib orbit (symprec 0.1 Å) is scanned. The other members' masks are mapped from
  it by the symmetry operation, resampled to the nearest grid offset.
- Energies come from orb_conserv_inf, batched on the GPU. The graphs are built with
  `from_ase_atoms_list`, which is 15× faster than one call per configuration.

  The energy-only forward pass skips the force backward pass. It agrees with
  `predict()` within the run-to-run scatter of `predict()` itself. On the GPU that
  scatter is up to 0.01–0.03 eV at points 15 eV above the minimum, from nondeterministic
  float32 reductions.

**Empty-framework PES** (ICML 2026 §3.4). Every Li is removed and one Li is scanned on
the same grid. E_1D/E_2D/E_3D are found by the same bisection on connected components of
a 2×2×2 tiling as `bvlain`'s BVSE barriers: the lowest threshold at which one component
spans 2, 4 or 8 replicas of the cell.

**Cheap descriptors** (`descriptors.py`).
- BVSE barriers from `bvlain`, at its default 0.2 Å resolution and 10 Å cutoff.
- Bonds use `coordination_motifs`' rule (d/R0 within 25% of the shortest). From them:
  - corner-sharing-or-isolated framework (Jun et al. 2022);
  - framework network dimensionality and the share of anions in finite groups;
  - Li CN, bond-length distortion and off-centring;
  - shortest Li–Li and Li–anion distances.
- The anion sublattice is matched to ideal bcc/fcc/hcp (Wang et al. 2015).
- Volume per anion.
- Gene-level baselines, from the pyxtal gene at 0.1 Å: Li fraction, mean anion
  electronegativity, number of Li orbits and the share of Li on sites with free
  coordinates.

**Scores** (`scores.py`). The pAUC is the area under the ROC curve for FPR ≤ 0.1, divided
by 0.1, so random scores 0.05. Every descriptor's orientation is fixed from the
literature in `DESCRIPTORS` before any score is seen.
- *Experimental set:* palette compounds whose framework matches a labelled OBELiX group.
  The bootstrap resamples framework groups, not compounds.
- *Kahle set:* the relaxed Kahle cells, with Maevskiy's Gaussian-smoothed labels.
- *Rediscovery:* the percentile ranks among the 12,383 palette compounds of each OBELiX
  family's fast members. For descriptors computed only on the PES selection, a
  percentile is read off a uniform random control of 500 palette compounds.

**MD** (`md.py`, ICML §3.4).
- A supercell of ≥ 100 atoms whose every width is ≥ 10 Å; positions relaxed to
  0.005 eV/Å.
- 10 ps of thermalisation and 100 ps of NVT production, both with Langevin BAOAB
  (friction time 100 fs). The time step is 2 fs, or 0.5 fs with H; the paper gives none.
- D comes from consecutive 15 ps windows, the MSD slope fitted over 3–15 ps; σ from
  Nernst–Einstein. The cell's centre-of-mass drift is subtracted first: Langevin does not
  conserve momentum, and the drift (D ≈ kT·τ/M, 0.014 Å²/ps for a 104-atom cell at
  500 K) otherwise makes every framework look molten and LiF look like a 0.6 S/cm
  conductor.
- Filters: framework 6·D·100 ps ≤ 2 Å² at 1000 K; Li max MSD > 4 Å² at 500 K; σ_300K
  from 500 K with Ea = 0.5 eV.

## Cost and scope of the PES scans

ORB conservative-inf manages 35–40k atom-evaluations/s on a free RTX 6000 Ada with the
energy-only forward pass, independent of batch size (8k or 24k atoms). The direct-20
model is 2× faster but is not the potential the relaxation used, so it was not adopted.

The estimated cost of the plan's selection on two GPUs, from grid size, Li orbits and
cell size:

| set | structures | frozen scan | empty scan |
|---|---|---|---|
| labelled (≤ 40 per OBELiX group) + Kahle + OBELiX | 880 | 14 h | 1.9 h |
| best 3,000 BVSE (E_1D) | 2,874 | 19 h | 4.4 h |
| random control from the palette | 500 | 8.8 h | 0.9 h |
| selection | 4,226 | ~42 h | ~7 h |
| all 12,576 | 12,576 | ~104 h | ~23 h |

A few low-symmetry cells would take up to 11 GPU-hours each. The frozen scan therefore
scans at most 12 Li orbits per structure, a seeded subset when there are more, and
records `n_orbits` and `n_scanned`; such a structure's free volumes are lower bounds.

## Reproducing Maevskiy et al.'s Ξ (`xi-check`)

The MP structures behind the paper's printed values were scanned unrelaxed, as the paper
did, with ORB instead of M3GNet. M3GNet-MP-2021.2.8-PES itself is no longer in matgl
(4.1.1 lists only MatPES-trained M3GNets), and the DGL-based matgl that had it does not
run on this Torch/Python, so the original potential could not be rerun.

| MP id | formula | published Ξ | ORB Ξ | ORB FV_c | ORB FV_d |
|---|---|---|---|---|---|
| mp-570935 | LiI | 0.981 | 0.905 | 0.056 | 0.119 |
| mp-1211296 | Li(BH)6 | 0.990 | 0.984 | 0.179 | 0.184 |
| mp-1185263 | LiPaO3 | 0.942 | 0.987 | 0.193 | 0.193 |
| mp-1222392 | LiUI6 | 0.82 | 0.911 | 0.030 | 0.124 |
| mp-644223 | LiBH4 | 0.965 | 0.779 | 0.046 | 0.095 |
| mp-2530 | Li2Te | 0.971 | 0.041 | 0.006 | 0.070 |
| mp-34038 | Li6NCl3 | 0.814 | 0.070 | 0.016 | 0.040 |
| mp-1222398 | LiGa(GeSe3)2 | 0.896 | 0.078 | 0.018 | 0.041 |
| mp-1211100 | LiB3H8 | 0.992 | 0.000 | 0.002 | 0.037 |
| mp-985582 | Li6PS5I | 0.703 | 0.002 | 0.005 | 0.039 |
| mp-755463 | Li3SbS3 | 0.887 | 0.000 | 0.002 | 0.022 |
| mp-768738 | Li3Bi(BO3)2 | 0.00 | 0.000 | 0.001 | 0.007 |

Ξ needs FV_c ≥ 0.01 and FV_d ≥ 0.071 to exceed ¼.
- **Where the free volumes are large,** in open halides, hydroborates and iodides, ORB
  agrees with the published values.
- **Elsewhere ORB's free volumes are smaller,** most clearly in the sulfides and
  argyrodites, which then fall below Maevskiy's thresholds.
- **It is not a defect of the scan.** The PES minimum lies at the Li's own site, 0.1–0.5 Å
  from it and within 0.02 eV of the energy there; there is no spurious low-energy hole
  near the exclusion spheres.
- **It is not ORB alone.** MACE-MPA-0 (one ASE call per point) gives Li2Te Ξ = 0.029,
  FV_c 0.0060, FV_d 0.058, essentially ORB's values. The likeliest explanation is
  M3GNet-MP-2021's softer PES (Deng et al. 2025), not the implementation.

Maevskiy's thresholds were tuned on M3GNet, so they do not transfer. As the plan expects,
Ξ is refitted on ORB: `xi_lofo` is a leave-one-family-out logistic fit on log10 FV_c,
log10 FV_d and MPE. The raw free volumes, which need no threshold, are scored too.

## BVSE against LiTraj's BVEL13k (`bvse-check`)

1,638 of BVEL13k's MP ids are in the benchmark set. On a random 500 of them, `bvlain` on
MP's unrelaxed geometry reproduces the published barriers:

| barrier | n | Spearman | median abs. difference |
|---|---|---|---|
| E_1D | 500 | 0.987 | 0.000 eV |
| E_2D | 500 | 0.989 | 0.010 eV |
| E_3D | 496 | 0.995 | 0.010 eV |

The benchmark's BVSE barriers are computed the same way, on the ORB-relaxed structures.

## Gate G1b: the framework matcher (`matcher-calibrate`)

20 OBELiX-labelled palette compounds, one per framework group with fast groups first,
were relaxed from their own Wyckoff genes by the protocol arm the candidates go through:
NEP89 prescreen, then ORB. Their frameworks were then matched against OBELiX's.

- **First attempt** (`g1b/`, 2026-10-25 14:09–14:20): controls picked by e_hull alone.
  It gave **10/20, FAIL** at pymatgen's default tolerances. Three of the misses had P1 or
  near-P1 genes: Li6AsS5I, Li10Si(PSe6)2 and the R3 NASICON LiSn3Sb(PO4)6. The protocol
  relaxed them into other structures, which no framework tolerance can rescue.
- **Second attempt** (`g1b_v2/`, 14:43–16:22): controls whose genes have a space group of
  at least 3 and at most 12 free coordinates. The protocol rebuilt 17 of the 20
  compounds' own structures.

| tolerance (ltol, stol, angle) | free track matched | also match another group | fixed track matched |
|---|---|---|---|
| 0.2, 0.3, 5° (pymatgen default) | **18/20, PASS** | 0 | 17/20 |
| 0.3, 0.5, 10° | 19/20 | 11 | 19/20 |
| 0.4, 0.6, 15° | 19/20 | 14 | 19/20 |

The structure screens run on the free-relaxation track, and on it the default tolerances
pass with no false match. Looser ones buy one control at the price of 11–14 matches to a
second group, so the defaults stay. The two misses:
- LiBO2, which the looser tolerances recover;
- Li3ScF6, whose protocol structure is another polymorph.

## Gate G1c: ORB MD on the references (`md-calibrate`)

The plan's 19 references were taken from the relaxed benchmark set, in the phase the plan
names where LeMat has it and otherwise the lowest-e_hull polymorph. Two are not among the
relaxed palette compounds and were dropped: LiTaOCl4 and LiBH4 (H is in the palette, but
LeMat's LiBH4 entries are not within 0.1 eV/atom). Three stand in with another phase:
- Li10GeP2S12: a P1 ordering (agm003282505), not the P4₂/nmc cell;
- Li3YCl6: P2₁/c, not P-3m1;
- Li3InCl6: R-3, not C2/m.

The 34 runs (17 references × 500 and 1000 K) used 100–252-atom cells and took
0.5–3.0 h each, 64 worker-hours with three workers sharing a GPU (2026-10-25 13:36 to
2026-10-26 11:40).

| reference | label | σ(500 K), S/cm | Li max MSD, 500 K (Å²) | σ(1000 K) | slowest framework D, 1000 K (Å²/ps) |
|---|---|---|---|---|---|
| Li6PS5Cl | fast | 0.40 ± 0.11 | 12.4 | 0.88 | < 0.001 |
| Li7P3S11 | fast | 0.34 ± 0.08 | 36.7 | 1.62 | < 0.001 |
| Li10GeP2S12 | fast | 0.27 ± 0.07 | 31.2 | 1.27 | < 0.001 |
| Li3YCl6 | fast | 0.12 ± 0.05 | 14.9 | 0.63 | 0.0047 (melts) |
| LiAlCl4 | poor | 0.094 ± 0.067 | 39.2 | 1.02 | 0.062 (melts) |
| Li3InCl6 | fast | 0.089 ± 0.023 | 9.4 | 0.79 | < 0.001 |
| Li3PS4 | fast | 0.021 ± 0.021 | 5.9 | 1.83 | 0.0014 |
| Li3N | fast | 0.006 ± 0.008 | 0.4 | 2.13 | 0.0018 |
| LiTi2(PO4)3 | poor | 0.0014 ± 0.0008 | 1.7 | 0.023 | < 0.001 |
| eight other poor references | poor | ≤ 0.0006 | ≤ 0.35 | ≤ 1.09 | ≤ 0.0024 |

**AUROC of σ(500 K), fast against poor: 0.957. G1c passes**, so ORB stays the MD
potential. The two misorderings are physical, not failures of the potential:
- LiAlCl4 melts at about 420 K. At 500 K it is a liquid in reality too, and its framework
  melts in the 1000 K run.
- Stoichiometric Li3N has no vacancies, and its conduction is vacancy-mediated. A perfect
  112-atom cell barely diffuses at 500 K (MSD 0.4 Å²) but conducts at 1000 K.

A first table, printed by a process started before the centre-of-mass correction, gave
0.700 and a FAIL. Every number above is re-analysed from the saved trajectories
(`g1c/trajectories/*.npz`, `g1c/md_analysis.csv`, `g1c/references.csv`).

## Gate G1: the transport screen (`bench-evaluate`)

Run on 2026-10-26 at 15:41, on the uncommitted branch `se-new-families` (base
`b31fb2d`). Outputs are `bench/evaluation/descriptors.csv` and `table.parquet`.

**Coverage.**
- *Empty-framework scan:* all 4,225 structures of the selection. The 4,226th, OBELiX
  entry 12511 (S8Si2 once its partial Li is ordered away), has no Li to scan.
- *Frozen-framework scan:* stopped at 3,749 of 4,226. The 477 left are the largest
  cells, about a day of both GPUs, and they include every LGPS and garnet member. It was
  stopped because it cannot change the verdict. Every frozen-framework descriptor
  rediscovers at most 1 of the 6 families it already covers. With LGPS and garnet both
  rediscovered it would reach at most 3 of 8, short of the 4 G1 needs.

Selected rows of the 24 descriptors (pAUC at FPR ≤ 0.1, random = 0.05; 95% intervals
from 1,000 bootstraps; the experimental set weighs OBELiX framework groups equally):

| descriptor | class | experimental pAUC | Kahle pAUC | families rediscovered |
|---|---|---|---|---|
| Li fraction | gene | 0.247 [0.069, 0.515] | 0.012 [0.000, 0.077] | 3/8 |
| composition, LOFO fit | gene | 0.205 [0.050, 0.500] | 0.172 [0.054, 0.318] | 0/8 |
| isolated-anion fraction | coarse | 0.202 [0.099, 0.529] | 0.117 [0.074, 0.207] | 0/8 |
| shortest Li–anion distance | coarse | 0.165 [0.039, 0.405] | 0.341 [0.173, 0.521] | 0/8 |
| corner-sharing (Jun 2022) | coarse | 0.062 [0.043, 0.085] | 0.067 [0.051, 0.088] | 0/8 |
| bcc anion lattice (Wang 2015) | coarse | 0.038 [0.015, 0.072] | 0.042 [0.018, 0.079] | 0/8 |
| BVSE E_3D | energy | 0.109 [0.014, 0.345] | 0.305 [0.153, 0.501] | 1/8 |
| Ξ (ORB, Maevskiy's thresholds) | energy | 0.249 [0.122, 0.415] | 0.333 [0.159, 0.574] | 0/6 |
| FV_c | energy | 0.275 [0.160, 0.431] | 0.329 [0.154, 0.560] | 0/6 |
| Ξ refitted on ORB, LOFO | combination | 0.218 [0.106, 0.375] | 0.328 [0.139, 0.610] | 0/6 |
| empty-framework E_1D | energy | 0.204 [0.007, 0.464] | 0.111 [0.014, 0.303] | 2/8 |
| **empty-framework E_2D** | energy | **0.354 [0.161, 0.568]** | **0.170 [0.058, 0.374]** | **4/8** |
| empty-framework E_3D | energy | 0.301 [0.135, 0.522] | 0.139 [0.041, 0.338] | 3/8 |
| all structure descriptors, LOFO | combination | 0.135 [0.004, 0.449] | 0.360 [0.186, 0.541] | 1/8 |

**G1 passes for one descriptor: the empty-framework E_2D barrier.** By the plan's rule it
alone enters the post-relaxation screen. Every gene-level and coarse descriptor fails, so
by Phase 3's own wording no gene-level surrogate qualifies.

The pass is marginal on both criteria:
- The Kahle interval clears random by 0.008.
- Rediscovery is exactly 4 of 8. The family medians are argyrodites 0.032, nitrides 0.073,
  antiperovskites 0.085 and LGPS 0.097, against a line at 0.10. The misses are halides
  0.182, layered 0.370, garnet 0.537 and NASICON 0.552.
- Resampling the 500-compound control and each family's members 2,000 times gives ≥ 4/8
  in only 36% of draws (E_3D 38%, E_1D 9%).
- It is the best of 24 descriptors scored against one fixed line, with no correction for
  that choice.
- Two of the four rediscovered families, antiperovskites and nitrides, have anion-only
  frameworks matched with loose tolerances (see the OBELiX label caveats above). They say
  less than the argyrodites and LGPS do.

**The pass rests on anion-only frameworks.** In an antifluorite or antiperovskite, the
framework left after Li is removed is a bare anion lattice (framework formula `Rn`). Its
holes are large, so its empty-framework barrier is low by construction. In the control,
E_2D's median is 0.47 eV on anion-only frameworks and 0.99 eV on the rest. Anion-only
frameworks are 7% of the control but 28% of the compounds within the line. All 101
anion-only compounds in the experimental set are labelled fast (the antiperovskite and
nitride groups); none is poor, though the G1c MD shows that Li2O, LiF and LiCl, which
share such frameworks, do not conduct. A post-hoc check on frameworks with at least
one cation:

| E_2D, frameworks with cations only | value | G1 criterion |
|---|---|---|
| experimental pAUC (612 compounds) | 0.286 [0.051, 0.553] | met by 0.001 |
| Kahle pAUC (118 of 121 cells) | 0.154 [0.045, 0.354] | **not met** |
| rediscovery (antiperovskites and nitrides removed) | 2/6, argyrodites and LGPS | **not met** |

For new framework families, which by definition have cations, **no descriptor passes
G1**. The formal pass stands under the plan's rule. Whether it justifies generation
(Phase 4) is a decision for the authors, not something the gate settles. A random
control on Phase 2's MD, run on 2026-10-28, bears on that decision
([below](#random-control-for-the-md-tiers-mine-md-control)): on frameworks with cations,
the line's picks reach an MD tier at 7/20 against 2/40.

**What E_2D sees.** The oxide frameworks are missed: LLZO's own E_2D is 0.95 eV, the
control's median, though BVSE puts it at 0.52 eV. The line at the control's top 10%
(E_2D ≤ 0.37 eV) is reached by:

| framework anions (control) | n | median E_2D (eV) | within the line | AUROC within the class, experimental set |
|---|---|---|---|---|
| O | 156 | 1.22 | 1.3% | 0.60 (279 compounds, 74 fast) |
| F | 57 | 1.13 | 8.8% | 0.68 (30, 18) |
| S, Se, Te | 133 | 0.84 | 12% | 0.71 (179, 96) |
| Cl, Br, I | 88 | 0.68 | 18% | 0.84 (63, 40) |
| mixed | 49 | 0.79 | 22% | 0.72 (134, 95) |
| N | 17 | 2.66 | 0% | 0.95 (28, 3) |

E_2D ranks within the halides and chalcogenides, but within the oxides it is close to
random. Its line therefore selects halides, chalcogenides and mixed-anion frameworks, and
the Phase 2 hits will be mostly those. Oxide conductors are out of its reach.

## Phase 2: mining LeMat (`mine-*`)

**Screens** (`mine-screen`, 2026-10-26), applied to the palette compounds outside the
families campaign's systems:

| screen | compounds left |
|---|---|
| outside the families campaign's 126 systems | 10,725 |
| gap ≥ 0.5 eV (MP GGA, else `3t16iys0`) | 7,415 |
| generic anion roles, formula | 6,255 |
| generic anion roles, bond valence of the ORB structure | 6,195 |
| coordination check | 6,137 |
| not a known-SE framework (no match to a fast OBELiX group) | **5,707** |

- **Anion roles** (`transport/screens.py`). One oxidation state per element, from
  pymatgen's common and ICSD-frequent states. Halogens and O must be negative. N, S, Se
  and Te may be positive only next to O or F. B, C, Si, Ge, Sn, P, As, Sb, Bi and H take
  either sign, metals are positive, and every anion is more electronegative than every
  cation. The formula stage removes polyhalides and Zintl-like formulas (LiI3, LiIBr3,
  LiTe2, LiBrF4, LiSeF6) and mixed-valence intercalates (Li(ZrS2)2, Li3Er3ScS8). Its
  known casualties are the mixed-H borohydride amides and closo-borates, as in the
  families campaign. BVAnalyzer then removes 53 compounds it cannot assign and 7
  tellurides with positive Te and no O or F.
- **Coordination check.** The reference is LeMat's clean cation environments (species,
  CN, polyhedron), from `screen_coordination_motifs.py` rerun on hera over the 622,766
  e_hull ≤ 0.1 structures with an anion (`mine/coordination/`). 400,789 of them are in
  scope, against 381,369 on aspire2a's unfixed dataset; there are 2,442 environments.
  Following the plan, the cutoff removes at most 1% of these compounds: an environment
  must occur in ≥ 44 LeMat structures, which removes 58 of 6,195 (0.94%). Phase 6 uses
  the same cutoff on the generated structures.

**Empty-framework scans** (`mine-pes`, finished 2026-10-27 03:04). The benchmark had
already scanned part of the survivors; the other 3,457 went to `mine/pes_empty.parquet`,
a separate file so that poor-labelled compounds beyond the benchmark's 40-per-group cap
stay out of G1's numbers. 431 of them first ran out of GPU memory while the frozen scan
shared the cards; the rerun scanned them all. Every one of the 5,707 survivors has an
E_2D.

**Ranking** (`mine-rank`, 2026-10-27). A survivor is a hit if its E_2D is within the
line G1 used for rediscovery, the random control's top 10% (`mining.rank.top_fraction`).
Hits are clustered by framework with G1b's matcher, and clusters are ranked by their best
member (`mine/hits.parquet`, `mine/clusters.csv`).
- 518 of the 5,707 survivors (9.1%) are hits, in 153 framework clusters.
- 31 clusters, with 142 hits, have anion-only frameworks. They include rank 1, Li2O
  with 63 members, and LiI, LiBr, LiCl, Li8SeN2 and Li7HN2. They are listed but not
  taken further, for the reason given under [G1](#gate-g1-the-transport-screen-bench-evaluate).
- 122 clusters, with 376 hits, have frameworks with cations.

| framework anions | survivors | hits | hits, frameworks with cations |
|---|---|---|---|
| O | 1,981 | 33 (1.7%) | 30 |
| S, Se, Te | 1,385 | 178 (13%) | 147 |
| Cl, Br, I | 1,075 | 211 (20%) | 133 |
| mixed | 554 | 57 (10%) | 37 |
| F | 499 | 19 (3.8%) | 12 |
| N | 213 | 20 (9.4%) | 17 |

As the G1 section predicted, the hits are halides and chalcogenides; oxides are a third of
the survivors but 6% of the hits.

Three properties of the hits limit what they can show:
- **Most are hypothetical.** 431 hits come from Alexandria, 58 from MP and 29 from OQMD.
  Only 14 are ICSD-backed, 7 of them in frameworks with cations.
- **Many have small gaps.** The tellurides and selenides at the top have GGA gaps of
  1.0–1.5 eV, above the 0.5 eV screen but narrow for an electrolyte.
- **Known-poor frameworks are not kept out.** 331 survivors match an OBELiX framework
  group labelled poor. 16 of them (4.8%) are hits, against 9.1% of all survivors:
  Li3CuS2, Li3BiS3, LiBSe2, Li2BeSiTe4 and Li2CuIBr2-type compounds. E_2D thus halves the
  rate of known-poor frameworks but does not exclude them.

"New" here means only that the framework matches no fast OBELiX group. The clusters have
not been checked against the literature.

The 20 best clusters with cations go to MD (`mine-md`, Phase 7's protocol and filters,
on the best member of each). The plan says only "the top clusters". The number, 20 (about
12 GPU-hours), and the exclusion of anion-only clusters were chosen after G1, from a
ranking run on part of the scans.

| rank | best member | LeMat id | e_hull (eV/atom) | GGA gap (eV) | E_2D (eV) | framework, dimension | other members |
|---|---|---|---|---|---|---|---|
| 2 | LiYTe2 | agm005107396 | 0.053 | 1.18 | 0.039 | Rn2Xe, 2D | LiLaSe2, LiScSe2, LiScTe2, LiSbTe2, LiAsSe2, LiSnTe2 |
| 3 | Li4GeSe4 | agm006046064 | 0.020 | 1.83 | 0.107 | Rn4Xe, 0D | Li4SnSe4, Li4TiS4 |
| 4 | LiErSe2 | agm005672218 | 0.047 | 1.47 | 0.127 | Rn2Xe, 2D | |
| 5 | Li4CuN2 | agm006036579 | 0.064 | 0.79 | 0.127 | Rn2Xe, 0D | Li4ZnN2 |
| 9 | Li2HN | mp-35695 | 0.087 | 2.48 | 0.137 | RnXe, 0D | |
| 10 | LiYTe2 | agm003097245 | 0.020 | 1.01 | 0.156 | Rn2Xe, 2D | 17 members in 11 systems, e.g. LiLaSe2, Li2ZnO2, Li2ZrN2, LiErS2 |
| 11 | LiInS | agm002096009 | 0.087 | 0.67 | 0.146 | RnXe, 2D | |
| 12 | Li4B4Sb2SO16 | mp-768968 | 0.100 | 2.56 | 0.146 | Rn16Xe7, 3D | Li6Sn2B4SO16 |
| 13 | LiErTeS | agm005191437 | 0.025 | 1.38 | 0.166 | Rn2Xe, 2D | LiYTeS |
| 16 | LiLaSbO | agm005202614 | 0.062 | 0.66 | 0.166 | Rn2Xe, 2D | LiErTeO, LiLuTeO |
| 19 | SrLiTeBr | agm005216705 | 0.033 | 1.35 | 0.176 | Rn2Xe, 2D | 12 more in 11 systems, e.g. SrLiSeCl, BaLiTeI, LiBiSe2 |
| 21 | Li3BiS3 | agm003214583 | 0.049 | 2.43 | 0.176 | Rn3Xe, 0D | |
| 22 | Li3CuS2 | mp-753714 | 0.077 | 2.46 | 0.176 | Rn2Xe, 0D | |
| 23 | Li2CaSO | agm001609015 | 0.078 | 2.38 | 0.176 | Rn2Xe, 3D | Li2CaNCl, Li2CaSeO |
| 26 | Li2BeCl4 | agm003448550 | 0.000 | 5.40 | 0.195 | Rn4Xe, 0D | 26 more in 20 systems, isolated tetrahedra: Li2ZnCl4, Li4SiSe4, Li3AsSe4, Li2SO4, Li2BeF4, … |
| 27 | Li4AgI5 | agm006102054 | 0.002 | 2.57 | 0.195 | Rn5Xe, 2D | Li4AgBr5, Li4AgCl5 |
| 28 | Li4TiS4 | agm003244046 | 0.025 | 1.49 | 0.186 | Rn4Xe, 0D | |
| 29 | Li3LaS3 | agm005957577 | 0.030 | 1.68 | 0.186 | Rn3Xe, 2D | 13 more in 13 systems, e.g. Li3YSe3, Li3ErS3, Li3ScS3, Li3InO3 |
| 30 | Li3Ca2Cl7 | agm005636540 | 0.044 | 5.34 | 0.186 | Rn7Xe2, 2D | 17 more in 17 systems, e.g. Sr2Li3I7, Li3Cd2Br7, Li3Pb2Cl7, Li3Zn2F7 |
| 31 | LiInS | agm002096021 | 0.091 | 0.67 | 0.186 | RnXe, 2D | |

The dimension is that of the best member's framework with Li removed (CrystalNN bonds,
Larsen's algorithm). 11 of the 20 frameworks are layers, 7 are isolated units and 2 are
three-dimensional. Over the control and all the survivors, E_2D selects by connectivity,
though not in favour of layers:

| framework, Li removed | control | median E_2D (eV) | within the line | survivors | hits |
|---|---|---|---|---|---|
| anion only | 34 | 0.47 | 41% | 270 | 142 (53%) |
| isolated units (0D) | 97 | 0.62 | 22% | 890 | 150 (17%) |
| chains (1D) | 45 | 0.80 | 8.9% | 514 | 47 (9.1%) |
| layers (2D) | 122 | 0.82 | 6.6% | 1,277 | 137 (11%) |
| three-dimensional | 202 | 2.04 | 1.5% | 2,756 | 42 (1.5%) |

Half the survivors have three-dimensional frameworks, but they give 8% of the hits. The
argyrodite and LGPS frameworks that E_2D rediscovered are also isolated units (PS4
tetrahedra). This survey (`mine-dimension`, `mine/framework_dimension.parquet`) was run
after the ranking, as a diagnostic.

The clusters do not map one-to-one onto compounds. Li4GeSe4 belongs to two Rn4Xe
clusters (ranks 3 and 26) and Li4TiS4 to three, and LiInS appears at ranks 11 and 31.
The matcher splits polymorphs of one compound, and it groups different compounds whose
anonymised frameworks agree.

## Results

**Phase 2 MD** (`mine-md`, 2026-10-27 03:05–08:49, both GPUs, 3 runs per card; G1c's
protocol: ORB `orb_conserv_inf`, Langevin BAOAB at 500 K and 1000 K, 10 ps thermalised,
100 ps sampled; `mine/md/candidates.csv`). G1c's references, run the same way, are the
scale: Li6PS5Cl reaches σ_300K − 1 s.e.m. = 2.2e-4 S/cm, Li7P3S11 2.0e-4, LGPS 1.4e-4 and
Li3InCl6 4.8e-5, and no poor reference reaches a tier.

| rank | compound | LeMat id | σ_500K (S/cm) | σ_300K − 1 s.e.m. (S/cm) | filters | tier |
|---|---|---|---|---|---|---|
| 12 | Li4B4Sb2SO16 | mp-768968 | 0.42 ± 0.06 | 2.6e-4 | pass | **electrolyte** |
| 9 | Li2HN | mp-35695 | 0.40 ± 0.08 | 2.3e-4 | pass | **electrolyte** |
| 3 | Li4GeSe4 | agm006046064 | 0.16 ± 0.03 | 9.6e-5 | pass | coating |
| 16 | LiLaSbO | agm005202614 | 0.079 ± 0.033 | 3.4e-5 | pass | coating |
| 10 | LiYTe2 | agm003097245 | 0.11 ± 0.07 | 3.0e-5 | pass | coating |
| 28 | Li4TiS4 | agm003244046 | 0.045 ± 0.028 | 1.2e-5 | pass | coating |
| 21 | Li3BiS3 | agm003214583 | 0.031 ± 0.029 | 1.6e-6 | pass | coating |
| 22 | Li3CuS2 | mp-753714 | 0.12 ± 0.04 | 6.4e-5 | framework moves (Cu) | — |
| 23 | Li2CaSO | agm001609015 | 0.038 ± 0.015 | 1.6e-5 | Li MSD 1.0 Å² at 500 K | — |
| 2 | LiYTe2 | agm005107396 | ≈ 0 | < 0 | pass | — |

The other ten fail a filter, with σ_500K ≤ 0.016 S/cm. Five of them have Li that does not
move at 500 K: LiErSe2,
Li4CuN2, LiErTeS, SrLiTeBr and Li3LaS3. In the other five (both LiInS clusters, Li2BeCl4,
Li4AgI5 and Li3Ca2Cl7) the framework moves at 1000 K. Overall, 6 of the 20 frameworks
move at 1000 K, 11 of the 20 have Li that does not move at 500 K, and 7 reach a tier.

What the two electrolyte-tier compounds are:
- **Li2HN, lithium imide**, is a known Li⁺ conductor: about 2.5–3e-4 S/cm at room
  temperature ([PCCP 2012](https://pubs.rsc.org/en/content/articlelanding/2012/cp/c2cp23636b)),
  and above 1e-3 S/cm after a low-temperature synthesis
  ([2025](https://pubs.rsc.org/en/content/articlehtml/2025/eb/d5eb00058k)). It is not in
  OBELiX's fast groups, so the screen did not know it. This is a blind rediscovery of a
  conductor, not a new family. In the 500 K run H also moves (D_H = 0.006 Å²/ps against
  D_N = 0.0003). The framework filter tests only the slowest non-Li species.
- **Li4B4Sb2SO16** is a computed MP structure (Fddd; trigonal BO3, octahedral SbO6 and a
  sulfate), not ICSD-backed. A web search found no experimental report. Its e_hull,
  0.0996 eV/atom, is at the screen's 0.1 limit. Its cluster also holds Li6Sn2B4SO16.

Among the coating tier, Li4GeSe4 sits in the selenido-LISICON family. Members of that
family (Li4SnSe4 among them) have been synthesised and measured, the best at about 2e-5
S/cm at 20 °C (Kaib et al., Chem. Mater.). LiLaSbO's GGA gap is 0.66 eV, LiYTe2's 1.0 eV.

The novelty checks were single web searches, not the Phase 8 literature check. As the
plan says, MD ranks; it does not predict σ. Each run is one ordered cell, ORB softens the
PES (Deng 2025), and σ_300K is extrapolated from 500 K with E_a = 0.5 eV.

### Random control for the MD tiers (`mine-md-control`)

**Question.** Does E_2D's line enrich the MD tiers among frameworks with cations? Without
a control, 7 of 20 says nothing. The user asked for the control after Phase 8. It was
fixed in the config (`mining.md_control`) on 2026-10-28, before the draw.

**Draw.** From the 4,648 eligible survivors, 40 were taken in a seeded random order (seed
20261028), one per framework (G1b's matcher). Eligible survivors:
- lie outside the line (best percentile above 0.1) and have a descriptor value;
- have a framework with cations;
- have at most 54 atoms, the largest `mine-md` structure.

They ran the same MD (2026-10-28 03:32–17:35, both GPUs, 3 runs per card;
`mine/md_control/`; W&B `-mining:v1` and tables `phase2_md_control` and
`phase2_md_control_comparison`).

| arm | structures | tiered | electrolyte tier | Li moves at 500 K | framework moves at 1000 K |
|---|---|---|---|---|---|
| `mine-md`: best member of the line's 20 best clusters | 20 | 7 (35%) | 2 | 9 | 6 |
| control: random survivors outside the line | 40 | 2 (5%) | 1 | 6 | 11 |

**The pre-registered test passes:** one-sided Fisher exact test on the tier rate,
p = 0.0044 (α = 0.05; `comparison.csv`). The electrolyte tier alone does not differ
(2/20 against 1/40, p = 0.26).

The control's two tiered structures:
- **Li2HN** `mp-699284`, another ordering of lithium imide (percentile 0.26), at
  1.9e-4 S/cm, in the electrolyte tier;
- **Li2Ti(BO3)2** `mp-770749` (0.080 eV/atom), at 1.9e-5, in the coating tier.

Three more have mobile Li but fail the framework filter: Li2BF5, LiHBrCl and Li2MgSnCl6.

**Not pre-registered.** The line favours soft anions: 15 of the 20 `mine-md` structures
have only S, Se, Te, Cl, Br or I as anions, against 15 of the 40 controls. The
enrichment holds on both sides of that split, though the strata are small:

| stratum | `mine-md` tiered | control tiered | one-sided p |
|---|---|---|---|
| only S, Se, Te, Cl, Br or I anions | 4 / 15 | 0 / 15 | 0.050 |
| the rest | 3 / 5 | 2 / 25 | 0.022 |

The arms also differ in cell size (median 8 against 18 atoms); this was not tested.

**What it says.**
- Among real compounds with cation frameworks, the structures Phase 2 picked by E_2D's
  line reach an ORB-MD tier seven times as often as random survivors outside the line.
  This is the first evidence that the screen ranks frameworks with cations. G1's
  labelled sets did not show it.
- It does not say that the hits are new: Phase 8 found mostly known compounds and
  types.
- It does not show enrichment in the electrolyte tier.
- The label is ORB MD, which G1c validated on 20 references (AUROC 0.957). ORB MD also
  lands near the measurement for Pnma Li4SnSe4
  ([Phase 8](#md-on-the-pnma-polymorphs-phase8-md-polymorphs)).

Nothing has been generated. Phase 3 has no target (no gene-level descriptor passed G1),
and Phase 4 waits for the authors' decision on G1.

## Phase 8: the mined candidates (`phase8-*`)

Run on hera on 2026-10-27, on the uncommitted tree on `8c0e0a3` (branch `se-new-families`).
Outputs are in `phase8/` of the run directory and in the W&B artifact
`se-new-families-hera-20261025-phase8:v0` (run `se-new-families-hera-20261025`, tables
`phase8_report` and `phase8_interfaces`). LeMat-Bulk-fixed's rows (`entries.parquet`)
are not uploaded; `phase8-entries` rebuilds them from the pinned file.

```bash
$PY $S mine-md-members && $PY $S phase8-entries && $PY $S phase8-window
$PY $S phase8-check && $PY $S phase8-prototypes && $PY $S phase8-report
$PY $S phase8-md-polymorphs    # 2026-10-28
```

The plan's Phase 8 was written for generated structures. Here its candidates are the 34
hits in the 7 clusters whose MD reached a tier. Each is a LeMat entry already relaxed with
PBE, so:
- its e_hull is LeMat-Bulk-fixed's PBE one; the plan's "free and fixed-symmetry e_hull"
  has no counterpart;
- what needs checking is that the ORB structure MD ran on is still the DFT one;
- DFT would add only r2SCAN energies. On 2026-10-28 the user decided that no DFT or AIMD
  inputs are prepared ([below](#dft-and-aimd)).

### Data and method

**Reference.** hera had only upstream LeMat-Bulk, whose MP energies are stale
([lemat_stale_mp_energies.md](lemat_stale_mp_energies.md)). LeMat-Bulk-fixed's
`lemat_pbe_ehull.csv.gz`, the reference of every e_hull in this campaign, was copied from
zeus (sha256 pinned in the config). `phase8-entries` keeps its 72,778 rows whose chemical
system lies inside one of 147 systems: each member's system, alone and joined with each
electrode's elements (up to 7 elements). From these rows every member's raw e_hull is
recomputed equal to LeMat-Bulk-fixed's (max |Δ| 1e-16 eV/atom).

**MP2020.** pymatgen's `MaterialsProject2020Compatibility` is applied to each row's
structure; the structure is what tells a sulfate from a sulfide and a peroxide from an
oxide. U follows MPRelaxSet's rule: the U elements get U when the most electronegative
element is O or F. That LeMat stores them this way on every source was checked from the
data, not assumed (`u_scheme_check.csv`):
- At compositions where LeMat's MP row and an Alexandria or OQMD row both lie within
  0.1 eV/atom of the hull, the lowest energies agree. Alexandria − MP has a median of
  +0.002 eV/atom over 521 compositions (range −0.08 to +0.08); OQMD − MP +0.031 over 29.
- A plain-GGA row would sit 0.3–1 eV/atom lower.
- The larger differences (up to ±1.3 eV/atom) all involve a structure far above the
  hull, where the two lowest rows are different polymorphs.

MP2020 refused no row. It does put the polysulfides LiS4, LiS5 and Li2S13 and the
superoxide-classified LiO8 on the Li–S and Li–O hulls: it gives polysulfide S the sulfide
correction and every O of a superoxide the superoxide one. They appear among some
oxidation products below. The raw windows, reported beside the MP2020 ones, are the check.

**Windows and interfaces** (`phase8-window`, `transport/electrochem.py`), following
Richards et al. 2016 ([archive](archive/acs.chemmater.5b04082.pdf)):
- *Window:* the range of μ_Li over which the composition is on the hull of the
  Li-open grand potential. As in the paper, a member above the hull is first placed on
  it. Its window is then the range in which its decomposition assemblage coexists; when
  the assemblage spans a facet, the window shrinks to a single potential (LiLaSbO,
  Li2TiN2). Windows therefore depend only on the composition.
- *Interfaces,* against each electrode of the panel decided on 2026-10-27, at its
  nominal potential (Li 0 V, LiC6 0.1, Li4Ti5O12 1.55, LiFePO4 3.45, LiCoO2 3.9,
  LiMn2O4 4.0, LiNi0.5Mn1.5O4 4.7 V; the paper used computed voltages):
  - eq. 2, the closed-system reaction, per atom;
  - eq. 4, open to Li at the electrode's potential, per atom other than Li. The
    electrode is held at its grand-potential equilibrium, the member on the hull;
  - eq. 5, the x = 0 term of eq. 4: decomposition by the Li reservoir alone.
  All are in `interfaces.csv`, with MP2020 and raw energies and their products.
- *Join flag* (plan, Phase 6): the join runs from the Li binary to the rest of the
  formula, one per anion that can take all of Li's charge (bond-valence states). It is
  flagged when LeMat holds no other composition strictly between the two. A five-element
  "rest" such as B4Sb2SO14 is not a compound, so for those members the flag says little.

**Structure types** (`phase8-prototypes`). Each member is matched against LeMat's 43,407
ICSD-backed structures with every species relabelled (`fit_anonymous`, pymatgen's
default tolerances). The screens' framework match anonymised only the cation and anion
roles, and only against known SEs.

### The families

| rank | family (structure type) | members (distinct) | MD (σ_300K − 1 s.e.m., S/cm) | window, MP2020 (V) | electrodes inside | known? |
|---|---|---|---|---|---|---|
| 12 | borate-sulfates (tychite-derived) | 3 (2) | Li4B4Sb2SO16 **2.6e-4**; Li6Sn2B4SO16 5.5e-6 | 3.44–4.33; 2.10–3.66 | LiFePO4, LiCoO2, LiMn2O4; LiFePO4 | theoretical |
| 9 | Li2HN (imide) | 3 (1) | **2.3e-4** | 0.24–0.63 | — | known conductor |
| 3 | Cmcm Li4MX4 (Li4GeO4 type) | 4 | Li4TiS4 **1.7e-4**; Li4GeSe4 9.6e-5; Li4SnSe4 6.4e-5 | 1.67–2.07 | — | known compositions, unreported polymorph |
| 10 | CdI2-type layers (CuScS2, anti-La2O3, TlCrTe2) | 18 | LiYTe2 3.0e-5; Li2ZrN2, Li2HfN2 Li immobile | 0.58–1.62; Li2MN2 0–1.1 | Li4Ti5O12; Li2MN2 Li, LiC6 | Li2ZrN2, Li2HfN2, LiErS2 known |
| 16 | ZrCuSiAs type, Li on the Cu site | 3 | LiLaSbO 3.4e-5; LiErTeO, LiLuTeO Li immobile | 0.80–0.83; 0.19–1.54 | — | not found |
| 21 | Li3BiS3 (proustite type) | 2 (1) | 1.6e-6 | 1.68–2.10 | — | not found |
| 28 | a second Cmcm Li4TiS4 | 1 | 1.2e-5 | 1.71–2.07 | — | not found |

Bold marks the electrolyte tier. "Distinct" counts ORB structures that pymatgen's default
StructureMatcher tells apart. Per-member numbers are in `phase8/report.csv`.

**Cluster 12, the borate-sulfates: Li4B4Sb2SO16 and Li6Sn2B4SO16.** This is the only
new-framework family that reached the electrolyte tier.
- *Structure.* Li6Sn2B4SO16 (Fd-3) is a borate analogue of tychite: it matches
  Na6Mg2(CO3)4(SO4) (mp-540866, mp-6892) and ferrotychite Na6Fe2(CO3)4(SO4)
  (mp-21590), with BO3 for CO3. Li4B4Sb2SO16 (Fddd) matches no ICSD-backed type in
  LeMat. Its Li-free framework clusters with Li6Sn2B4SO16's: a tychite-derived framework
  with Sb⁵⁺ for Sn⁴⁺ and a third fewer Li.
- *Literature.* Both are theoretical MP entries (`mp-768968`, `mp-769001`; MP provenance
  "theoretical", no ICSD id). The Alexandria entries `agm003282527` and `agm003282630`
  are the same structures. No report of either compound was found. Synthetic tychite is
  a measured Na⁺ conductor, 6.0e-7 S/cm at 100 °C
  ([Dalton Trans. 2024](https://pubs.rsc.org/en/content/articlelanding/2024/dt/d4dt00396a)).
- *e_hull.* 0.100 and 0.065 eV/atom, identical with MP2020 (MP's own: 0.094 and 0.068).
  They decompose to Li3B7O12 + LiSb3O8 + Li2SO4 + B2O3, and to LiBO2 + Li2SO4 + SnO2.
- *Window.* 3.44–4.33 V and 2.10–3.66 V (MP2020; raw 3.44–3.99 and 2.04–3.32 V). Below
  them Sb⁵⁺ and Sn⁴⁺ are reduced. Li4B4Sb2SO16's window is the only one that contains
  LiFePO4, LiCoO2 and LiMn2O4.
- *Interfaces.* Against every cathode, |ΔΦ| ≤ 0.04 eV per non-Li atom (LiNi0.5Mn1.5O4
  0.13 for the Sn member). Li metal and graphite reduce both hard (−1.5 to −1.9 eV) to
  Li2O, LiB, Li3Sb or Li17Sn4 and Li2S. Those products conduct electrons, so they will
  not passivate.
- *Joins:* none (flagged), but the "rest" is not a compound. *Coordination:* tested and
  passed.
- *Structure check.* Li4B4Sb2SO16's ORB structure is its DFT one (rms 0.007).
  Li6Sn2B4SO16 relaxed under ORB to P1 (rms 0.10, +1.7% volume); its MD ran on that cell.
- *MD.* Li4B4Sb2SO16 is in the electrolyte tier (2.6e-4 S/cm). Li6Sn2B4SO16 is in the
  coating tier only (5.5e-6; σ_500K 0.032 against 0.42).

On thermodynamics alone this is a high-voltage material: a catholyte or a cathode coating,
not an electrolyte against Li, graphite or Li4Ti5O12. Its transport rests on the Sb
member. Its PBE e_hull has no r2SCAN check: MP's r2SCAN data reaches only ternaries in
these systems.

**Cluster 9, lithium imide Li2HN.** It is a known conductor (2.5e-4 S/cm, [PCCP 2012](https://pubs.rsc.org/en/content/articlelanding/2012/cp/c2cp23636b)),
so this is a blind rediscovery.
- *Structure.* The three members are MP's orderings of the disordered ICSD imide (MP
  remark "Ordering of Disordered Crystal") and match one another. LeMat holds 14 Li2HN
  orderings. The lowest, `mp-1189725` (Pnma), is MP's "Dilithium imide – orthorhombic"
  (ICSD 168791).
- *e_hull.* The MD'd R3m ordering is 0.087 eV/atom above it.
- *Structure check.* ORB relaxed that ordering to I4mm with 8% less volume (rms 0.20);
  MD ran on this cell.
- *Window.* 0.24–0.63 V: reduced to Li4NH, oxidised to LiNH2 + Li2N2. It is an
  anode-side window, and every cathode oxidises it (eq. 4 −2.6 to −3.8 eV).
- *Join.* Li3N–NH3 holds LiNH2 (ICSD) and three more.

**Cluster 3, Cmcm Li4GeSe4, Li4SnSe4, Li4TiS4: the Li4GeO4 type in chalcogenides.**
- *Framework.* The Li-free frameworks match OBELiX's LISICON group. The group is
  unlabelled: no member reaches 1e-4 S/cm and not all are below 1e-7. "New" here only
  meant no fast group.
- *Structure type.* Li4GeO4 and Li4TiO4 (Cmcm), and Li4PbO4 for the Sn member.
- *Literature.* Li4SnSe4 is known (Kaib et al., Chem. Mater. 2013, about 2e-5 S/cm at
  20 °C), in Pnma: ICSD 193768, `mp-1194700`. That polymorph survived the Phase 2
  screen; its E_2D, 0.205 eV, lies just outside the line. The Cmcm Li4SnSe4 is
  0.017 eV/atom above it. Cmcm Li4GeSe4 is LeMat's lowest Li4GeSe4 (0.020 eV/atom, no
  ICSD); no report of it was found (Li4GeS4 is Pnma). For Li4TiS4 no report was found;
  LeMat's lowest is Pnma `mp-766540`, 0.013 eV/atom.
- *Window.* 1.67–1.92, 1.71–1.92 and 1.71–2.07 V, the narrow thio- and selenide window.
  Li metal and graphite reduce them, and every cathode oxidises them (−1.0 to
  −2.1 eV). Only Li4Ti5O12 is close (−0.03 to −0.08 eV).
- *Joins.* Populated (Li2GeSe3, Li8GeSe6, …; two ICSD-backed on Li2Se–SnSe2).
- *MD.* All three members reach a tier. Li4TiS4 (`mp-756811`) is in the electrolyte tier
  at 1.7e-4 S/cm, Li4GeSe4 at 9.6e-5 and Li4SnSe4 at 6.4e-5.
- *r2SCAN (MP).* MP's r2SCAN data has Pnma Li4TiS4 (`mp-766540`) 0.026 eV/atom above
  its hull (PBE 0.013) and Pnma Li4GeSe4 and Li4SnSe4 on it (MP API, 2026-10-27). It
  has none of the Cmcm members.

It is a known composition family in an unreported polymorph. Its fastest member is
Li4TiS4. None of LeMat's ten Li4TiS4 entries is ICSD-backed, no report of it was found,
and it has the narrow sulfide window. In the Pnma polymorph, 0.025 eV/atom lower, Li does
not move ([below](#md-on-the-pnma-polymorphs-phase8-md-polymorphs)).

**Cluster 10, the CdI2-type layers: LiYTe2 and 17 members.** Several known structure
types share one Li-free framework:
- CuScS2 or LiMnTe2 type for LiYTe2 and LiLaTe2;
- anti-La2O3 (Ce2SO2, CaAl2Si2) type for Li2MN2 (M = Zr, Hf, Ti, Sn) and for
  Li2CdO2 and Li2ZnO2;
- TlCrTe2 or LiCrS2 type for the P-3m1 LiErS2 and LiLaSe2.

Li2ZrN2 is ICSD-backed (`mp-3216`, three ICSD ids). Li2HfN2 has been made from Li3N and
HfN (Pauling File `pf-36313`; [Dalton Trans. 1974](https://pubs.rsc.org/en/content/articlelanding/1974/dt/dt9740002166)).
Li2SnN2 comes from MP's computed nitride map (CNGMD). LiErS2 is known in R-3m (ICSD,
`mp-15791`); the P-3m1 member is 0.049 eV/atom above it. No Li-conductivity report was
found for Li2ZrN2 or Li2HfN2.
- *Windows.* Li2ZrN2 0–1.07 V and Li2HfN2 0–1.12 V: the only members stable against Li
  metal and graphite. LiYTe2 0.58–1.62 V, LiLaTe2 0.71–1.37 V.
- *MD.* In Li2ZrN2 and Li2HfN2, Li does not move at 500 K (max MSD 0.10 Å²). The
  cluster's coating tier rests on LiYTe2 (3.0e-5 S/cm).

**Cluster 16, P4/nmm LiLaSbO, LiErTeO, LiLuTeO: the ZrCuSiAs (LaCuOS) type with Li on
the Cu site.** Each matches 82–93 ICSD-backed structures, among them CeCuTeO, PrCuSO and LaFeAsO.
No report of the Li compounds was found.
- *LiLaSbO is not ionic.* Its bond-valence states give Sb(−2), and its GGA gap is
  0.66 eV. The motif analysis refused it, so its coordination was never tested.
- *Windows.* LiLaSbO 0.80–0.83 V: its assemblage La2SbO2 + Li2Sb spans a facet.
  LiErTeO and LiLuTeO 0.19 and 0.29–1.54 V.
- *Joins:* none (flagged).
- *MD.* In LiErTeO and LiLuTeO, Li does not move at 500 K (max MSD 0.24–0.25 Å²). The
  cluster's coating tier rests on LiLaSbO, which is not ionic.

**Cluster 21, R3c Li3BiS3: the proustite and pyrargyrite type (Ag3AsS3, Ag3SbS3).** No
report of Li3BiS3 was found. LeMat's lowest is R-3 (`mp-753720`, 0.025 eV/atom); the R3c
members are at 0.049.
- *Coordination.* No clean cation environment, so the check was not tested.
- *Window.* 1.68–2.10 V.

**Cluster 28, a second Cmcm Li4TiS4 (`agm003244046`).** It matches no ICSD-backed type,
at 0.025 eV/atom (Pnma 0.013). Its window and interfaces are cluster 3's Li4TiS4.

### Windows and interfaces

| compound | window, MP2020 (V) | raw (V) | reduced to (MP2020) | oxidised to (MP2020) |
|---|---|---|---|---|
| Li4B4Sb2SO16 | 3.44–4.33 | 3.44–3.99 | 0.47 Li3B7O12 + 0.27 LiSb3O8 + 0.22 Li2SO4 + 0.04 SbO2 | 0.43 B2O3 + 0.32 LiSb3O8 + 0.22 Li2SO4 + 0.03 O2 |
| Li6Sn2B4SO16 | 2.10–3.66 | 2.04–3.32 | 0.52 LiBO2 + 0.32 Li2SnO3 + 0.14 Li2SO4 + 0.02 SnS2 | 0.47 Li3B7O12 + 0.26 SnO2 + 0.22 Li2SO4 + 0.05 LiO8 |
| Li2HN | 0.24–0.63 | 0.15–0.63 | 1.00 Li4HN | 0.75 LiH2N + 0.25 LiN |
| Li4GeSe4 | 1.67–1.92 | 1.67–1.89 | 0.80 Li2Se + 0.20 Ge | 0.80 Li2GeSe3 + 0.20 LiSe3 |
| Li4SnSe4 | 1.71–1.92 | 1.71–1.89 | 0.60 Li2Se + 0.40 SnSe | 0.80 Li2SnSe3 + 0.20 LiSe3 |
| Li4TiS4 | 1.71–2.07 | 1.71–2.00 | 0.60 LiTiS2 + 0.40 Li2S | 0.80 TiS3 + 0.20 LiS4 |
| LiYTe2 | 0.58–1.62 | 0.58–1.62 | 0.73 Y5Te6 + 0.27 Li2Te | 0.56 Y2Te3 + 0.44 YTe3 |
| LiLaTe2 | 0.71–1.37 | 0.71–1.37 | 0.67 LaTe + 0.33 Li2Te | 1.00 LaTe2 |
| LiLaSe2 | 0.79–1.69 | 0.79–1.69 | 0.67 LaSe + 0.33 Li2Se | 0.97 La10Se19 + 0.03 Li2Se |
| LiErS2 | 0.64–2.28 | 0.64–2.23 | 0.78 LiEr3S4 + 0.22 Li2S | 1.00 ErS2 |
| LiSnSeI | 1.46–2.02 | 1.46–2.02 | 0.33 Li2Se + 0.33 Sn + 0.33 LiI | 0.50 SnSe2 + 0.50 SnI2 |
| Li2ZrN2 | 0.00–1.07 | 0.00–0.93 | stable against Li | 0.78 LiZr3N4 + 0.22 N2 |
| Li2HfN2 | 0.00–1.12 | 0.00–0.97 | stable against Li | 0.78 LiHf3N4 + 0.22 N2 |
| Li2TiN2 | 0.87–0.88 | 0.79–0.79 | 0.53 Li5TiN3 + 0.47 LiTi3N4 | 0.78 LiTi3N4 + 0.22 LiN |
| Li2SnN2 | 0.67–0.90 | 0.61–0.67 | 0.67 LiN + 0.33 LiSn | 0.67 SnN + 0.33 LiN |
| Li2CdO2 | 1.72–2.89 | 1.72–2.66 | 0.67 Li2O + 0.33 Cd | 0.67 CdO + 0.33 Li2O2 |
| Li2ZnO2 | 1.35–2.98 | 1.35–2.76 | 0.83 Li6ZnO4 + 0.17 Zn | 0.67 ZnO + 0.33 Li2O2 |
| LiLaSbO | 0.80–0.83 | 0.80–0.83 | 0.83 La2SbO2 + 0.17 Li3Sb | 1.00 LaSbO |
| LiErTeO | 0.19–1.54 | 0.19–1.54 | 0.56 Er2O3 + 0.33 Li2Te + 0.11 Er | 0.83 Er2TeO2 + 0.17 LiTe3 |
| LiLuTeO | 0.29–1.54 | 0.29–1.54 | 0.56 Lu2O3 + 0.33 Li2Te + 0.11 Lu | 0.83 Lu2TeO2 + 0.17 LiTe3 |
| Li3BiS3 | 1.68–2.10 | 1.68–2.00 | 0.75 Li2S + 0.25 Bi | 0.75 LiBiS2 + 0.25 LiS4 |

Interface reaction energies, eq. 4 with MP2020 energies (eV per non-Li atom; 0 is stable).
Eqs. 2 and 5 and the raw energies are in `interfaces.csv`.

| compound | Li | LiC6 | Li4Ti5O12 | LiFePO4 | LiCoO2 | LiMn2O4 | LiNi0.5Mn1.5O4 |
|---|---|---|---|---|---|---|---|
| Li4B4Sb2SO16 | -1.93 | -1.76 | -0.35 | -0.01 | -0.01 | 0.00 | -0.04 |
| Li6Sn2B4SO16 | -1.55 | -1.38 | -0.20 | -0.04 | -0.02 | -0.03 | -0.13 |
| Li2HN | -0.24 | -0.29 | -0.67 | -2.57 | -3.02 | -3.12 | -3.82 |
| Li4GeSe4 | -1.63 | -1.48 | -0.08 | -1.05 | -1.41 | -1.49 | -2.05 |
| Li4SnSe4 | -1.63 | -1.47 | -0.07 | -1.05 | -1.41 | -1.49 | -2.05 |
| Li4TiS4 | -0.80 | -0.87 | -0.03 | -1.08 | -1.44 | -1.52 | -2.08 |
| LiYTe2 | -0.32 | -0.32 | -0.02 | -0.65 | -1.24 | -0.94 | -1.17 |
| LiLaTe2 | -0.24 | -0.21 | -0.06 | -0.69 | -1.14 | -0.94 | -1.18 |
| LiLaSe2 | -0.26 | -0.23 | 0.00 | -0.58 | -0.96 | -0.79 | -1.00 |
| LiErS2 | -0.19 | -0.29 | -0.03 | -0.52 | -0.95 | -0.74 | -0.94 |
| LiSnSeI | -1.58 | -1.37 | 0.00 | -0.48 | -0.74 | -0.66 | -0.89 |
| Li2ZrN2 | 0.00 | 0.00 | -0.27 | -1.53 | -1.83 | -1.89 | -2.36 |
| Li2HfN2 | 0.00 | 0.00 | -0.28 | -1.53 | -1.83 | -1.90 | -2.36 |
| Li2TiN2 | -0.09 | -0.09 | -0.36 | -1.57 | -1.87 | -1.94 | -2.41 |
| Li2SnN2 | -1.14 | -0.88 | -0.40 | -1.67 | -1.97 | -2.03 | -2.50 |
| Li2CdO2 | -1.39 | -1.22 | -0.21 | -0.29 | -0.54 | -0.61 | -1.08 |
| Li2ZnO2 | -0.98 | -0.88 | -0.12 | -0.28 | -0.51 | -0.58 | -1.04 |
| LiLaSbO | -0.24 | -0.20 | -0.24 | -0.87 | -1.21 | -1.06 | -1.29 |
| LiErTeO | -0.06 | -0.11 | -0.02 | -0.59 | -0.96 | -0.82 | -1.05 |
| LiLuTeO | -0.10 | -0.14 | -0.04 | -0.61 | -0.99 | -0.86 | -1.09 |
| Li3BiS3 | -1.79 | -1.64 | -0.10 | -0.96 | -1.29 | -1.37 | -1.89 |

What the windows show:
- Only Li2ZrN2 and Li2HfN2 are stable against Li metal.
- Only the borate-sulfates have small reaction energies against the cathodes.
- No candidate is stable across a full cell.
- The chalcogenides have the narrow 1.6–2.1 V windows Richards et al. found for the
  thiophosphates.
- Li4Ti5O12 is the electrode most of the sulfides, selenides and tellurides touch
  least (|ΔΦ| ≤ 0.1 eV).

### Structure check (`phase8-check`)

All 34 ORB structures match their LeMat-Bulk-fixed DFT structures with pymatgen's default
StructureMatcher. Volumes agree within 1% for 23 of them. The exceptions:
- **Li2HN `mp-35695`,** the MD'd member: R3m → I4mm, rms 0.20, 8% denser.
- **Li6Sn2B4SO16 `mp-769001`:** Fd-3 → P1, rms 0.10, +1.7%.
- **Smaller changes:** the five OQMD Li2CdO2 entries +3.9–4.3%; the OQMD LiErTeO and
  LiLuTeO +2.4%; Li2ZnO2 +1.2% with C2/m → P-3m1; Li2HN `mp-34465` −1.1%.

For Li2HN the MD therefore probed an ORB-modified cell; the others are the DFT structures.

### MD on further members (`mine-md-members`)

The plan's Phase 7 runs MD on "up to 3 structures per candidate family". `mine-md`
ran only each cluster's best member. `mine-md-members` (2026-10-27 11:11–13:13, both
GPUs, `phase8/md/`) takes each tiered cluster's members by e_hull, up to 3 distinct
structures. It skips a member whose ORB structure matches one already chosen:
- Li2HN `agm003173795` and `mp-34465`, which match `mp-35695`;
- Li4B4Sb2SO16 `agm003282527`, which matches `mp-768968`;
- Li3BiS3 `mp-753677`, which matches `agm003214583`.

Of the 14 distinct members chosen, 7 had already run, so 7 were new. The protocol is
Phase 2's.

| rank | compound | LeMat id | e_hull (eV/atom) | σ_500K (S/cm) | Li max MSD, 500 K (Å²) | σ_300K − 1 s.e.m. (S/cm) | tier |
|---|---|---|---|---|---|---|---|
| 3 | Li4TiS4 (Cmcm) | mp-756811 | 0.038 | 0.29 ± 0.06 | 18.7 | 1.7e-4 | **electrolyte** |
| 3 | Li4SnSe4 (Cmcm) | agm006046394 | 0.017 | 0.13 ± 0.04 | 14.2 | 6.4e-5 | coating |
| 12 | Li6Sn2B4SO16 | mp-769001 | 0.065 | 0.032 ± 0.024 | 6.5 | 5.5e-6 | coating |
| 10 | Li2HfN2 | agm002156433 | 0 | ≈ 0 | 0.10 | < 0 | Li immobile |
| 10 | Li2ZrN2 | mp-3216 | 0 | ≈ 0 | 0.10 | < 0 | Li immobile |
| 16 | LiErTeO | oqmd-3271894 | 0.054 | ≈ 0 | 0.24 | < 0 | Li immobile |
| 16 | LiLuTeO | oqmd-4527247 | 0.067 | ≈ 0 | 0.25 | < 0 | Li immobile |

Every framework passes the 1000 K filter. What the members change:
- **Cluster 3 gains an electrolyte-tier member.** Cmcm Li4TiS4 `mp-756811` scores
  1.7e-4, between G1c's LGPS (1.4e-4) and Li7P3S11 (2.0e-4) references. Cluster 28's
  Cmcm Li4TiS4 (`agm003244046`, 0.025 eV/atom), a different arrangement of the same
  composition, reaches only 1.2e-5.
- **Cluster 12's electrolyte tier rests on Li4B4Sb2SO16 alone.** Li6Sn2B4SO16 is 13 times
  slower at 500 K. Its MD ran on the ORB P1 cell.
- **Cluster 10:** the two members stable against Li metal, Li2ZrN2 and Li2HfN2, have Li
  that does not move at 500 K. The cluster's tier rests on LiYTe2.
- **Cluster 16:** in LiErTeO and LiLuTeO Li does not move. The cluster's tier rests on
  LiLaSbO, which is not ionic.

### MD on the Pnma polymorphs (`phase8-md-polymorphs`)

Cluster 3's members are Cmcm. `phase8-md-polymorphs` (2026-10-28 03:00–03:32, both
GPUs, `phase8/md_polymorphs/`, keys in `phase8.md_polymorphs`) ran the same MD on two
Pnma polymorphs:
- Li4SnSe4 `mp-1194700`, the measured one: ICSD 193768, about 2e-5 S/cm at 20 °C
  (Kaib et al. 2013);
- Li4TiS4 `mp-766540`, LeMat's lowest Li4TiS4.

Alexandria's `agm003244253` and `agm003244254` are the same structures.

| compound | LeMat id | e_hull (eV/atom) | σ_500K (S/cm) | Li max MSD, 500 K (Å²) | σ_300K (S/cm) | σ_300K − 1 s.e.m. (S/cm) | tier |
|---|---|---|---|---|---|---|---|
| Li4SnSe4 (Pnma) | mp-1194700 | 0.003 | 0.065 ± 0.042 | 6.5 | 4.7e-5 | 1.7e-5 | coating |
| Li4TiS4 (Pnma) | mp-766540 | 0.013 | ≈ 0 | 0.9 | — | < 0 | Li immobile; framework moves |

- **The measured polymorph lands near its measurement.** Kaib et al.'s 2e-5 S/cm lies
  between σ_300K − 1 s.e.m. and σ_300K. It is one compound, and the extrapolation's
  E_a = 0.5 eV is the plan's pessimistic choice, not a fit. The Cmcm Li4SnSe4 of cluster 3
  scores about 4 times higher (6.4e-5).
- **Li4TiS4's fast transport is a property of the Cmcm cell only.** In Pnma Li4TiS4, Li
  does not move at 500 K. At 1000 K its slowest non-Li species moves 3.1 Å² over the
  100 ps (filter ≤ 2 Å²). The electrolyte-tier Cmcm member (`mp-756811`) is 0.025
  eV/atom above Pnma.

### DFT and AIMD

On 2026-10-28 the user decided that the DFT and AIMD inputs are not prepared. So:
- e_hull is LeMat-Bulk-fixed's PBE one;
- the borate-sulfates have no r2SCAN check, since MP's r2SCAN thermo data reaches only
  ternaries in B–Li–O–S–Sb and B–Li–O–S–Sn;
- σ is ORB MD's ranking number.

On 2026-10-27 a stage that has since been removed, `phase8-bundle`, wrote an r2SCAN
bundle (`dft_bundle/`) and an AIMD package (`aimd_package/`). Both were withdrawn and
deleted from the run directory; W&B's `-phase8:v0` still holds them. The config keys
`phase8.r2scan_clusters` and `phase8.aimd_members` are no longer read.
