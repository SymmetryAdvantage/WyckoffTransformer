# New Li solid-electrolyte families: results so far

> **Status, 2026-10-28.** Phases 0–2 and 8 of the
> [plan](se_new_families_campaign_plan.md) have run, and so has a random control for
> Phase 2's MD. Nothing has been generated: Phase 4 (generation) waits for the authors'
> decision, which [the last section](#the-decision-for-the-authors) sets out.
>
> This page is the summary for co-authors. It assumes no knowledge of the campaign. Every
> number comes from the full record, [se_new_families_campaign.md](se_new_families_campaign.md),
> which has the methods, commands and data paths. Branch `se-new-families`; W&B run
> [`se-new-families-hera-20261025`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/se-new-families-hera-20261025).

## In short

- **A transport screen that knows no families works only partly.** Of 24 descriptors,
  one passed the plan's benchmark gate: E_2D, the barrier for one Li to cross the
  framework with every other Li removed. It passed at the margin, and the pass rests on
  frameworks made of anions only. On frameworks that contain cations, which every new
  family would have, no descriptor passed.
- **E_2D's picks from the database reach an MD conductivity tier seven times as often
  as random compounds.** A tier means at least 10⁻⁶ S/cm extrapolated to room
  temperature from molecular dynamics (MD); see [Terms](#terms).
  - 7 of the 20 picks did, against 2 of 40 random compounds from outside E_2D's top 10%
    (one-sided Fisher p = 0.0044, test fixed before the draw).
  - The electrolyte tier alone (≥ 10⁻⁴ S/cm) shows no difference: 2 of 20 against 1
    of 40.
- **The picks are mostly known chemistry.** Of the 7 families that reached a tier:
  - one is a known conductor (lithium imide, Li2HN);
  - four are known structure types; one of them holds known compositions in an
    unreported polymorph;
  - two match no ICSD-backed structure type in full: the borate-sulfates and a second
    Li4TiS4 polymorph.

  The one genuinely new framework in the electrolyte tier, a borate-sulfate
  (Li4B4Sb2SO16), sits at the stability limit used (0.1 eV/atom above the convex hull).
  It is stable only above 3.4 V against Li, so at best it is a cathode coating.
- **MD agrees with the two measured compounds it was checked on:**
  - Li4SnSe4: 1.7–4.7 × 10⁻⁵ S/cm in MD, about 2 × 10⁻⁵ measured;
  - Li2HN: 2.3–2.9 × 10⁻⁴ S/cm in MD, 2.5–3 × 10⁻⁴ measured.
- **Nothing here used DFT beyond the database's PBE energies, or AIMD.** It was decided
  on 2026-10-28 not to prepare DFT or AIMD inputs for these candidates.

## The question

An earlier campaign ([solid_electrolyte_campaign.md](solid_electrolyte_campaign.md))
generated crystals with WyFormer only in the 126 chemical systems where Li solid
electrolytes are already known. Inside those systems it could assume that a stable new
crystal might conduct Li. To find a new *family*, the search has to leave those systems,
and then that assumption is gone. Something else has to say which candidates might
conduct: a transport screen that knows no families.

The plan ([se_new_families_campaign_plan.md](se_new_families_campaign_plan.md), written
2026-10-25 with its gates fixed before any result) therefore:
1. tests whether a family-blind screen can find known families it was not tuned on;
2. applies it to a database of existing compounds before generating anything;
3. checks the screen's picks with molecular dynamics (MD).

## Terms

- **Framework:** a crystal with every Li removed and every other atom relabelled only
  as cation or anion (Jun et al. 2022). Two compounds with matching frameworks are one
  structural family. A **new framework** matches no fast-conducting framework group in
  OBELiX. "New" means only that until the literature check.
- **Anion-only framework:** a framework with no cations left, such as the bare anion
  lattice of Li2O or of an antiperovskite. Its holes are large, so any barrier computed
  on it is low by construction.
- **LeMat-Bulk:** a database of DFT (PBE) structures pooled from the Materials Project
  (MP), Alexandria and OQMD. The campaign uses its corrected release, LeMat-Bulk-fixed.
  **e_hull** is the energy above the convex hull in eV/atom; 0 is stable, and the
  campaign keeps compounds within 0.1.
- **Palette:** the 44 elements WyFormer generates with.
- **ORB** (`orb_conserv_inf`): the machine-learned interatomic potential used for every
  relaxation, energy scan and MD run here.
- **E_2D:** the empty-framework barrier (ICML 2026 §3.4). Every Li is removed, one Li is
  moved over a fine grid in the cell with ORB, and E_2D is the lowest energy at which
  the low-energy region connects across the cell in two directions. Lower is better.
- **The line:** E_2D's top 10% within a random control of palette compounds, E_2D ≤
  0.37 eV. A compound within the line is a **hit**.
- **pAUC:** the area under the ROC curve at false-positive rates up to 10%, rescaled so
  that random scores 0.05 and perfect scores 1.
- **Label sets:** OBELiX (experimental room-temperature conductivities, which label
  frameworks fast or poor) and Kahle et al. 2020 (121 AIMD simulations at 1000 K).
- **MD tiers:** ORB MD at 500 and 1000 K (10 ps equilibration, 100 ps sampled).
  - Conductivity is extrapolated to 300 K from 500 K with a deliberately pessimistic
    activation energy of 0.5 eV. The value used is that estimate minus one standard
    error: σ_300K − 1 s.e.m.
  - **Electrolyte tier:** ≥ 10⁻⁴ S/cm. **Coating tier:** ≥ 10⁻⁶ S/cm.
  - Filters: Li must move (mean square displacement > 4 Å² at 500 K), and the rest of
    the structure must not (≤ 2 Å² over 100 ps at 1000 K).
  - For scale, the same MD gives Li6PS5Cl 2.2 × 10⁻⁴, Li7P3S11 2.0 × 10⁻⁴,
    Li10GeP2S12 1.4 × 10⁻⁴ and Li3InCl6 4.8 × 10⁻⁵ S/cm; no known poor conductor
    reaches a tier.
- **Gates:** pass/fail criteria fixed in the plan before any result:
  - **G1:** does the transport screen work?
  - **G1b:** does the framework matcher recognise families?
  - **G1c:** does ORB MD separate fast from poor conductors?

## What was run

| step | what | result |
|---|---|---|
| Phase 1 benchmark | 12,576 structures relaxed with ORB: the palette's ionic Li compounds in LeMat-Bulk-fixed, Kahle's 121 cells and OBELiX's 72 ordered CIFs; 24 descriptors scored against both label sets | one descriptor, E_2D, passes G1 at the margin |
| G1b | 20 known compounds rebuilt from their WyFormer genes and matched to their framework groups | 18/20, passes |
| G1c | ORB MD on 17 known fast and poor conductors | AUROC 0.957, passes |
| Phase 2 | mining: the 10,725 palette compounds outside the earlier campaign's systems, screened and ranked by E_2D | 518 hits in 153 framework clusters; MD on the best of the 20 top clusters with cations: 7 reach a tier |
| Phase 8 | checks on those 7 families: literature and ICSD, structure types, electrochemical windows and interface reactions, MD on further members | below |
| control | MD on 40 random compounds from outside the line, the same way | 2 reach a tier |

Everything ran on hera between 2026-10-25 and 2026-10-28.

## Finding 1: the benchmark (gate G1)

G1 asked two things of a descriptor:
- its pAUC interval excludes random on both label sets;
- it rediscovers at least half of the known families: each family's median fast member
  lands within the line.

| descriptor | experimental pAUC | Kahle pAUC | families rediscovered |
|---|---|---|---|
| **E_2D** | **0.354 [0.161, 0.568]** | **0.170 [0.058, 0.374]** | **4/8** |
| E_3D (same scan, 3D) | 0.301 [0.135, 0.522] | 0.139 [0.041, 0.338] | 3/8 |
| Ξ (Maevskiy et al. 2025), on ORB | 0.249 [0.122, 0.415] | 0.333 [0.159, 0.574] | 0/6 |
| BVSE E_3D | 0.109 [0.014, 0.345] | 0.305 [0.153, 0.501] | 1/8 |
| corner-sharing framework (Jun 2022) | 0.062 [0.043, 0.085] | 0.067 [0.051, 0.088] | 0/8 |
| Li fraction (from the composition) | 0.247 [0.069, 0.515] | 0.012 [0.000, 0.077] | 3/8 |

Random pAUC is 0.05; intervals are 95% bootstrap. The other 18 descriptors are in the
[record](se_new_families_campaign.md#gate-g1-the-transport-screen-bench-evaluate).

**E_2D passes, but at the margin.**
- Its Kahle interval clears random by 0.008, and it rediscovers exactly 4 of 8 families
  (argyrodites, nitrides, antiperovskites, LGPS).
- It is the best of 24 descriptors, with no correction for that choice.
- Resampling the control and the families gives ≥ 4/8 in only 36% of draws.

**The pass rests on anion-only frameworks.** Two of the four rediscovered families,
nitrides and antiperovskites, have them. Restricted to frameworks with at least one
cation, E_2D meets the experimental criterion by 0.001, fails the Kahle criterion, and
rediscovers 2 of 6 families. So for new frameworks, which have cations by definition,
**no descriptor passes G1.** The plan's rule passes E_2D formally; whether that justifies
generation is the authors' call.

**What E_2D sees:** halides and chalcogenides. It ranks well within those classes
(AUROC 0.71–0.84) but close to random within oxides (0.60). It misses the oxide
conductors: LLZO's E_2D is 0.95 eV, the control's median.

**No gene-level screen exists.** Every descriptor computable from a WyFormer gene (the
composition and Wyckoff positions, before relaxation) failed G1. So Phase 3, which would
have trained gene-level surrogates, has no target. Generation cannot be steered towards
conductors; any screen would act only after relaxation.

## Finding 2: mining the database (Phase 2)

| screen | compounds left |
|---|---|
| palette compounds outside the earlier campaign's 126 systems | 10,725 |
| band gap ≥ 0.5 eV | 7,415 |
| sensible oxidation states (formula, then bond valence) | 6,195 |
| no rare cation environment | 6,137 |
| framework not a known fast conductor's | **5,707** |

- **Hits:** 518 of the 5,707 (9.1%), in 153 framework clusters. 31 clusters are
  anion-only (Li2O, LiI, LiBr, …) and were set aside. The best member of each of the
  20 top clusters with cations went to MD.
- **What the hits are:** halides and chalcogenides, as Finding 1 predicts. Oxides are a
  third of the survivors but 6% of the hits.
- **Most hits are hypothetical:** 431 come from Alexandria, 58 from MP and 29 from OQMD.
  Only 14 are ICSD-backed.
- **MD result:** 7 of the 20 reach a tier, 2 of them the electrolyte tier:
  - Li4B4Sb2SO16, a computed MP structure;
  - Li2HN, lithium imide.

  Of the other 13:
  - 6 have a framework that moves at 1000 K;
  - 6 more have Li that does not move at 500 K;
  - 1 passes both filters but is below the coating tier.

## Finding 3: the random control

**Why.** 7 of 20 means nothing without a baseline. The control asks whether E_2D's
ranking, rather than the earlier screens, picks the compounds that reach a tier. It was
fixed in the config on 2026-10-28, before the draw.

**How.** 40 compounds were drawn at random (seed 20261028) from the 4,648 survivors
that:
- lie outside the line;
- have a framework with cations;
- have no more atoms than the largest Phase 2 MD cell (54).

One compound was taken per framework, as the clusters hold one each. They ran the same
MD.

| | Phase 2 picks (within the line) | random control (outside the line) |
|---|---|---|
| structures | 20 | 40 |
| **reach a tier** | **7 (35%)** | **2 (5%)** |
| electrolyte tier | 2 | 1 |
| Li moves at 500 K | 9 | 6 |
| framework moves at 1000 K | 6 | 11 |

- **The pre-registered test passes:** one-sided Fisher exact test on the tier rate,
  p = 0.0044 (α = 0.05).
- **The electrolyte tier alone does not differ:** 2/20 against 1/40, p = 0.26. The
  control's electrolyte-tier compound is another arrangement (ordering) of lithium imide.
- **Not explained by anion chemistry** (checked after the fact, small numbers). The line
  favours soft anions, so the arms were split by that:

  | compounds | picks tiered | control tiered | one-sided p |
  |---|---|---|---|
  | anions only S, Se, Te, Cl, Br or I | 4 of 15 | 0 of 15 | 0.050 |
  | the rest | 3 of 5 | 2 of 25 | 0.022 |

- **Not checked:** the picks have smaller cells (median 8 atoms against 18).

**What it shows.** On real compounds with cation frameworks, E_2D ranks ORB-MD
transport. This is the first evidence of that; the benchmark's label sets did not show
it. It does not show that E_2D finds electrolyte-grade conductors, or new ones.

## Finding 4: what the 7 families are (Phase 8)

Phase 8 checked all 34 members of the 7 tiered clusters:
- **Literature and ICSD:** LeMat's ICSD provenance, MP's and OQMD's pages, papers.
- **Structure types:** a match to any ICSD-backed LeMat structure with all species
  relabelled.
- **Windows and interfaces:**
  - the window is the voltage range against Li/Li⁺ in which the compound is
    thermodynamically stable;
  - interfaces are its reaction energies with seven electrodes, from Li metal (0 V) to
    LiNi0.5Mn1.5O4 (4.7 V);
  - both follow Richards et al. 2016, on LeMat's PBE energies with MP's 2020
    corrections.
- **Structure check:** that ORB kept the DFT structure.
- **MD on further members:** up to three structures per family.

| family (structure type) | best MD result, S/cm | window (V) | known? |
|---|---|---|---|
| borate-sulfates Li4B4Sb2SO16, Li6Sn2B4SO16 (tychite-derived) | **2.6 × 10⁻⁴** (Sb); 5.5 × 10⁻⁶ (Sn) | 3.4–4.3; 2.1–3.7 | theoretical MP entries |
| Li2HN, lithium imide | **2.3 × 10⁻⁴** | 0.24–0.63 | known conductor, ~2.5 × 10⁻⁴ S/cm |
| Li4GeSe4, Li4SnSe4, Li4TiS4 (Cmcm, Li4GeO4 type) | **1.7 × 10⁻⁴** (Li4TiS4) | 1.7–2.1 | known compositions; this polymorph unreported |
| LiYTe2 and 17 others in CdI2-like layers | 3.0 × 10⁻⁵ | 0.6–1.6 | known types; Li2ZrN2, Li2HfN2, LiErS2 known |
| LiLaSbO, LiErTeO, LiLuTeO (ZrCuSiAs type) | 3.4 × 10⁻⁵ | 0.2–1.5 | not reported |
| Li3BiS3 (proustite type) | 1.6 × 10⁻⁶ | 1.7–2.1 | not reported |
| a second Cmcm Li4TiS4 | 1.2 × 10⁻⁵ | 1.7–2.1 | not reported |

Bold marks the electrolyte tier. For comparison, a bulk electrolyte facing Li metal and
a 4 V cathode would need a window of about 0–4 V. The sulfide electrolytes in use have
windows of about 1.7–2.1 V (Richards et al. 2016) and rely on interphases.

What stands out:
- **The borate-sulfates are the one new framework in the electrolyte tier.** They are
  computed MP structures with no experimental report. Li6Sn2B4SO16 is a borate analogue
  of the mineral tychite, which in synthetic form conducts Na⁺ weakly (6 × 10⁻⁷ S/cm at
  100 °C).
  - Li4B4Sb2SO16 is 0.100 eV/atom above the hull, at the campaign's limit.
  - Their windows start at 3.4 and 2.1 V. Against the cathodes they hardly react
    (within 0.04 eV per non-Li atom; 0.13 for the Sn member against LiNi0.5Mn1.5O4).
    Li metal and graphite reduce them strongly (1.4–1.9 eV per non-Li atom).
  - They are therefore high-voltage materials, at best a cathode coating or
    catholyte. Only the Sb member is fast; the Sn member reached only the coating
    tier.
- **Li4TiS4's fast transport belongs to one metastable polymorph.**
  - The Cmcm structure reaches the electrolyte tier. In Pnma Li4TiS4, LeMat's lowest
    polymorph and 0.025 eV/atom more stable, Li does not move.
  - No Li4TiS4 is in ICSD.
- **MD meets experiment where it can be checked.**
  - Pnma Li4SnSe4 is the polymorph Kaib et al. (2013) synthesised and measured at about
    2 × 10⁻⁵ S/cm. MD puts it at 1.7 × 10⁻⁵ (minus one standard error) to
    4.7 × 10⁻⁵ S/cm.
  - Li2HN gives 2.3–2.9 × 10⁻⁴ S/cm against a measured 2.5–3 × 10⁻⁴ (PCCP 2012).
    Its MD ran on a cell that ORB had densified by 8%.
- **The only members stable against Li metal**, Li2ZrN2 and Li2HfN2 (both known
  compounds), have Li that does not move in MD.

## Limitations

- **MD ranks; it does not predict conductivity.**
  - Each run is one ordered cell of at least 100 atoms.
  - Universal potentials such as ORB soften the energy landscape and underestimate
    barriers (Deng et al. 2025).
  - The 300 K value is an extrapolation with an assumed activation energy.
- **Stability is PBE only.** No r2SCAN, no further DFT relaxation and no AIMD were run.
- **Some choices were made after G1:**
  - 20 clusters for MD, and setting the anion-only clusters aside;
  - the random control itself, though its test was fixed before its draw.
- **The anion-chemistry split of the control was chosen after seeing the result,** and
  each half holds 15–30 compounds.

## The decision for the authors

The plan's rule passes E_2D, so generation (Phase 4) may go ahead. But the plan also
says: if no descriptor passes, stop and write up the negative result, because generation
would then only produce more stable Li insulators. For frameworks with cations, the case
that matters for new families, E_2D did not pass the benchmark. Before generation can
start, the authors need to decide which reading applies.

**For generating:**
- The random control shows that E_2D ranks MD transport on cation frameworks of real
  compounds (7/20 against 2/40).
- ORB MD passed its own gate and agrees with the two measured compounds checked.

**Against:**
- There is no gene-level screen, so generation cannot be aimed at conductors. E_2D and
  MD would filter only after relaxation.
- E_2D reaches only halides and chalcogenides, and the electrolyte tier was not
  enriched.
- Mining with the same screen turned up mostly known chemistry. Its one new framework
  in the electrolyte tier is a high-voltage material at the stability limit.

**The two paths:**
1. **Write up now.** The paper would cover the family-blind benchmark, the mining, the
   random control and the Phase 8 checks. Its message: a transport screen validated on
   known electrolytes ranks MD transport, but among existing compounds with cations it
   found little that is new.
2. **Generate** (plan, Phase 4: about 500 systems, 2.5M draws), screened after
   relaxation by E_2D and MD.
   - Since E_2D reaches only halides and chalcogenides, the systems could be limited to
     those. That would be a change to the plan.
   - This tests whether generation finds what the database lacks. The results above
     give little reason to expect it will.

## Where things are

- **Record:** [se_new_families_campaign.md](se_new_families_campaign.md). It has the
  methods, every table, the commands and the gates' details.
- **Plan:** [se_new_families_campaign_plan.md](se_new_families_campaign_plan.md).
- **Code:** `scripts/se_new_families_campaign.py` (one stage per step) and
  `src/wyckoff_transformer/transport/` (descriptors, MD, windows).
- **Config:** `yamls/campaigns/se_new_families_hera_20261025.yaml`. Each key added after
  the start carries a dated comment.
- **Outputs:**
  - hera: `~/.local/share/wyformer/runs/se-new-families-hera-20261025/`;
  - W&B artifacts `-benchmark`, `-gates`, `-mining` and `-phase8`, with tables for the
    descriptors, the MD candidates, the control and the Phase 8 report.
