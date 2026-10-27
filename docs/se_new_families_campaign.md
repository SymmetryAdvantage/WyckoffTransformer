# Generative search for new Li SE families: Phases 0–2

> **Status, 2026-10-27:** Phases 1 and 2 are complete.
> - G1b passes (18/20) and G1c passes (MD AUROC 0.957).
> - G1 passes formally for one descriptor, the empty-framework E_2D barrier, at the
>   margin. The pass rests on anion-only frameworks; on frameworks with cations no
>   descriptor passes ([G1](#gate-g1-the-transport-screen-bench-evaluate)).
> - Phase 2 (mining LeMat) screened and ranked 5,707 compounds
>   ([Phase 2](#phase-2-mining-lemat-mine-)). Of the 20 best clusters, 7 reach an MD tier
>   ([Results](#results)). The two in the electrolyte tier are Li4B4Sb2SO16, a computed MP
>   structure, and Li2HN, a known conductor that OBELiX lacks.
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
> and `-mining`. Everything up to that upload ran on uncommitted code on base `b31fb2d`;
> the run's config records `code_dirty: true`.

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
(Phase 4) is a decision for the authors, not something the gate settles.

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

Nothing has been generated. Phase 3 has no target (no gene-level descriptor passed G1),
and Phase 4 waits for the authors' decision on G1.
