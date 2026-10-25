# Generative search for new Li solid electrolytes

> **Scope:** generate crystals in the chemical families known to contain Li solid
> electrolytes (SEs), keep only those that are new and not known SEs, and find which
> of them are near the convex hull. Campaign `se-families-hera-20261022`, run on
> hera (and iapetus for part of the ORB relaxation) from 2026-10-22, branch
> `se-families-campaign`.
>
> **Status:** run through to the hull on 2026-10-24. After the report's checks,
> 2,852 compositions have a free-relaxed ORB structure within 0.1 eV/atom of the hull,
> and 2,556 of them are new to LeMat-Bulk. See [Relaxation and hull](#relaxation-and-hull).
> Nothing here has been checked against ICSD or by DFT. The two band-gap regressors are
> on Hugging Face (2026-10-25); see [Screens](#screens-and-why-each-is-there).

The pipeline is the one of the ICML 2026 AI4Science paper
([archive/101_Generative_Pipeline_for_Di.pdf](archive/101_Generative_Pipeline_for_Di.pdf)),
up to and including the hull. Electrochemical windows, PES descriptors of Li transport
and MD are not part of it.

| stage | here | the paper |
|---|---|---|
| generation | [WyFormer-LeMat-Chemsys](https://huggingface.co/SymmetryAdvantage/WyFormer-LeMat-Chemsys), one exact chemical system per request, 20,000 draws per system | Alex-MP-20 WyFormer, CSX mode, Li required, 42 allowed elements |
| gene novelty | augmented-fingerprint key against every split of `lemat_bulk_fmax1_stress` (4.82M keys) | SWORD vs LeMat-Bulk and ICSD |
| known SEs | reduced formula not in [OBELiX](https://github.com/NRC-Mila/OBELiX) (599 measured Li SEs) | — |
| validity | single-valence charge balance from pymatgen's common oxidation states | d_min > 0.7 Å after reconstruction |
| band gap | gene-level regressor on MP 2026.09.28's GGA/GGA+U gaps (`mp_20260928_gga_gap`), ≥ 0.5 eV | gene-level regressor on MP, ≥ 0.5 eV |
| selection | top 400 per system by WyFormer-LeMat-Energy predicted hull distance, ≤ 40 per formula | random 50,000 of 101,789 |
| Wyckoff → structure | the protocol's [wide NEP89 → ORB arm](de_novo_ranking_protocol_nep89_variants.md#variant-2--wide-then-narrow) | PyXtal + MACE-MPA-0 |
| stability | `e_above_hull` against the `orb_conserv_inf` split of LeMat-Bulk-MLIP-Hull | MACE-MPA-0 vs re-relaxed MP |

Everything the campaign does is set in
[`yamls/campaigns/solid_electrolyte_families_hera_20261022.yaml`](../yamls/campaigns/solid_electrolyte_families_hera_20261022.yaml)
and run by [`scripts/solid_electrolyte_campaign.py`](../scripts/solid_electrolyte_campaign.py):

```bash
.venv/bin/python scripts/solid_electrolyte_campaign.py systems
.venv/bin/python scripts/solid_electrolyte_campaign.py generate --device cuda:0
.venv/bin/python scripts/solid_electrolyte_campaign.py bandgap --device cuda:0
.venv/bin/python scripts/solid_electrolyte_campaign.py select
.venv/bin/python scripts/solid_electrolyte_campaign.py relax
.venv/bin/python scripts/solid_electrolyte_campaign.py report --wandb
```

Outputs live in the runs store under `se-families-hera-20261022/` and are logged to the
W&B run of the same name.

## The families

Enumerated from the "Family" column of OBELiX, Bachman et al.
([Chem. Rev. 116, 140, 2016](https://doi.org/10.1021/acs.chemrev.5b00563)), recent
halide and oxyhalide reviews (LiMOCl4, M = Nb, Ta), and the newer phosphide
(Li9AlP4, Li14SiP6), thioborate (Li9B19S33) and sulfide-iodide (Li7Si2S7I) conductors.
Each family is a set of chemical systems; a system is requested exactly, so Li-Y-Cl
means a compound of all three.

| family | known SE prototypes | systems | n |
|---|---|---|---|
| halide | Li3YCl6, Li3InCl6, Li2ZrCl6, LiAlCl4 | Li-M-X, M ∈ {Y, Sc, La, Er, Lu, In, Ga, Al, Zr, Hf, Ta, Nb, Mg, Bi}, X ∈ {Cl, Br, I} | 42 |
| oxyhalide | LiNbOCl4, LiTaOCl4, Li2ZrOCl4 | Li-M-O-X, M ∈ {Nb, Ta, Zr, Hf, Al}, X ∈ {Cl, Br} | 10 |
| argyrodite | Li6PS5Cl, Li7Si2S7I | Li-M-S-X, M ∈ {P, Si, Ge}, X ∈ {Cl, Br, I} | 9 |
| thio | β-Li3PS4, Li4GeS4, Li3SbS4, Li5B7S13 | Li-M-S, M ∈ {P, Ge, Si, Sn, Sb, As, Al, Ga, B} | 9 |
| lgps | Li10GeP2S12 | Li-M-P-S, M ∈ {Ge, Si, Sn} | 3 |
| antiperovskite | Li3OCl, Li2OHCl | Li-O-X, Li-S-X, X ∈ {Cl, Br, I} | 6 |
| fluoride | Li3AlF6, Li2ZrF6 | Li-M-F, M ∈ {Al, Ga, Sc, Y, Zr, Hf, Nb, Ta, Mg, In} | 10 |
| phosphide | Li9AlP4, Li8SiP4, Li14SiP6 | Li-M-P, M ∈ {Al, Ga, Si, Ge, Sn, In} | 6 |
| nitride | Li3N, Li7PN4, Li5NCl2 | Li-M-N, M ∈ {P, Si, B, Al, Ga, Ge}; Li-N-X | 9 |
| garnet_perovskite | Li7La3Zr2O12, Li5La3Ta2O12, LLTO | Li-La-M-O, M ∈ {Zr, Hf, Ta, Nb, Ti, Sn} | 6 |
| nasicon | LiTi2(PO4)3, LiGe2(PO4)3 | Li-M-P-O, M ∈ {Ti, Ge, Zr, Hf, Sn} | 5 |
| lisicon_oxide | Li4SiO4–Li3PO4, Li3BO3, Li7TaO6 | Li-M-O, M ∈ {Si, Ge, P, B, Ta, Nb, Zr, Al} | 8 |
| hydride | LiBH4, Li2B12H12, Li4BN3H10 | Li-B-H, Li-Al-H, Li-B-N-H | 3 |

126 systems in all. Na conductors were left out by choice.

## Screens, and why each is there

**Gene novelty** drops a gene whose augmentation-invariant key occurs anywhere in
LeMat-Bulk (`lemat_bulk_fmax1_stress`, all splits). The protocol's `score` stage judges
*structure* novelty against the same reference, so the two never disagree about which
database counts.

**Known SEs.** A gene whose reduced formula equals an OBELiX "Reduced Composition"
(e.g. Li3YCl6, Li6PS5Cl, Li7La3Zr2O12) is dropped even when its structure is new: the
question is new electrolytes, not new polymorphs of known ones. OBELiX's doped,
fractional compositions only ever match at exactly that stoichiometry. The data are in
the store at `data/obelix/` (commit `4eaac88`, sha256 pinned in the campaign yaml).

**Charge balance** was added after a first generation pilot on the same three systems.
Ranking by predicted hull distance put impossible stoichiometries at the top
(Li46P4S7Cl4, LiY2Cl8, LiLa10Zr4O19): the energy regressor is optimistic exactly where
it has no data. The check asks for one common oxidation state per element (pymatgen
`common_oxidation_states`) summing to zero, with every anion more electronegative than
every cation. Every prototype in the table passes except the mixed-valence-hydrogen
closo-borates and amide-borohydrides (Li2B12H12, Li4BN3H10). The ICSD-prior guess the
evaluation code uses (`coordination_motifs.guess_oxidation_states`) was tried first and
dropped: it took 27 s on Li13P6S39Cl8 and admits Zintl-type states such as Y⁺ and S⁻.

**Band gap.** No gene-level band-gap checkpoint existed: the paper's MP-20 regressors
(W&B `i705onzv`, `l2tpl829`, MAE 0.23 eV) logged no weights, and the three
`band_gap_adamw_wsd` runs on `mp_2026_gga_gap` (2026-09-11) failed with
`KeyError: 'band_gap'` because their tensors predated the tokeniser carrying that field.
[`yamls/models/mp_2026_gga_gap/der/band_gap_adamw_wsd.yaml`](../yamls/models/mp_2026_gga_gap/der/band_gap_adamw_wsd.yaml)
was trained first on `mp_2026_gga_gap` (W&B
[`ksvnn3oj`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ksvnn3oj)). By
then MP release 2026.09.28 was live, so the same config was retrained on its GGA/GGA+U
gaps ([`mp_20260928_gga_gap`](../data/mp_20260928_gga_gap/README.md), W&B
[`3t16iys0`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/3t16iys0)) and
on its r2SCAN gaps ([`mp_20260928_r2scan_gap`](../data/mp_20260928_r2scan_gap/README.md),
W&B [`xy2dr9us`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/xy2dr9us)),
both 2026-10-23 from commit `110ba09`. The screen uses `3t16iys0`: the paper's 0.5 eV
threshold is a PBE one, and the r2SCAN set is 95% GNoME intermetallics (19.8% with a gap)
under CC BY-NC. `xy2dr9us` is predicted for every gene and reported, not screened on.

Test-split scores through the screening path (`predict_gene_scalars`, 8 equivalent
descriptions per gene), from
[`scripts/evaluate_band_gap.py`](../scripts/evaluate_band_gap.py); the JSONs are in the
campaign's `band_gap_eval/`:

| model | test split | n | MAE (eV) | MAE, gap > 0 | P / R @ 0.5 eV |
|---|---|---|---|---|---|
| `ksvnn3oj` | `mp_2026_gga_gap` | 14,180 | 0.345 | 0.526 | 0.838 / 0.866 |
| `3t16iys0` | `mp_20260928_gga_gap` | 14,206 | 0.356 | 0.539 | 0.837 / 0.862 |
| `3t16iys0` | `mp_20260928_r2scan_gap` | 11,680 | 0.183 | 0.782 | 0.853 / 0.682 |
| `xy2dr9us` | `mp_20260928_r2scan_gap` | 12,276 | 0.069 | 0.295 | 0.917 / 0.909 |
| `xy2dr9us` | `mp_20260928_gga_gap` | 14,167 | 0.789 | 1.051 | 0.657 / 0.838 |

The two 2026.09.28 sets share no material (each material has one blessed gap), so the
cross rows are transfer checks across both chemistry and functional. `ksvnn3oj` must not be
scored on `mp_20260928_gga_gap`: the labels are the April ones under MP's new ids, so most
of that test split is in its training set (it scores a meaningless 0.153 eV there). The
paper's 0.23 eV was measured on MP-20 and is not comparable. A gene carries no bond
lengths, so single-element genes are weak: `3t16iys0` puts diamond Si, a training example
with a 0.61 eV PBE gap, at 0.01 eV. It gives LiCl 6.24 eV (MP 6.25) and Li2O 5.09 eV
(MP 4.90). The r2SCAN model puts Li2O, which its data lack, at 1.4 eV.

Both models are on the Hub with model cards, uploaded 2026-10-25 from the campaign's
`hf_stage/` with `scripts/push_to_hub.py`:
[`SymmetryAdvantage/WyFormer-MP-BandGap-PBE`](https://huggingface.co/SymmetryAdvantage/WyFormer-MP-BandGap-PBE)
(`3t16iys0`, CC BY 4.0, Hub revision `4f087059a03d`) and
[`SymmetryAdvantage/WyFormer-MP-BandGap-r2SCAN`](https://huggingface.co/SymmetryAdvantage/WyFormer-MP-BandGap-r2SCAN)
(`xy2dr9us`, CC BY-NC 4.0, revision `ae44293e3a69`). Each Hub checkpoint is
byte-identical to the local one, and `WyckoffTrainer.from_huggingface` reproduces
the LiCl and Li2O gaps above. The cards say `predict_gene_scalars` is not yet in the
public repository, which stays true until this branch is pushed.

**Selection.** The genes passing all screens are ranked by WyFormer-LeMat-Energy's
predicted distance to the LeMat-Bulk-fixed PBE hull. Each system keeps its best genes, at
most 40 per reduced formula, up to 400 genes or 12,000 conventional-cell atoms, whichever
comes first; genes over 300 atoms are dropped. The [pilot](#the-pilot) priced the wide arm
at ~1.7 CPU-s and ~1 ORB-s on hera per atom, so the budget is in atoms rather than genes:
one 168-atom garnet gene costs as much as five 32-atom halides.

## Wyckoff → structure → hull

Each family is one run of `wyformer-protocol`; halides are split by anion. The run
uses `generate --trial-multiplier 10 --pyxtal-timeout 60`, then `prescreen` (NEP89,
fixed symmetry, dedup), then `relax --relax-from prescreen` (`orb_conserv_inf`, four
stages), then `score`. The CPU stages of later runs overlap the relaxations of earlier
ones. iapetus relaxes runs of up to 2,500 genes and 60,000 atoms through its own
launcher, and the outputs sync back to hera for scoring. Scoring reads only the
aggregated `structures*.csv` and `cifs*/`, so the round trip does not affect it.

The run was launched as

```bash
SSH_AUTH_SOCK= .venv/bin/python scripts/solid_electrolyte_campaign.py relax \
    --cpu-lanes 2 --cpu-cores 12 --remote-max-atoms 60000
```

These flags change where and how fast the stages run, not what they compute: each
trial's seed is fixed by the trial itself. They were added an hour into the run. A
single CPU lane took the runs in family order, and the first five are all too big for
iapetus, so iapetus would have sat idle for about 11 h. At its pilot rate of ~4.6k
atoms/h, thio's 105k atoms (2,158 genes, within the gene cap) would then have kept it
busy for ~23 h, past the end of hera's share. Now one lane takes the runs in campaign
order and the other takes the smallest run iapetus may relax first (phosphide, hydride,
lgps, nasicon), each with 12 of hera's 24 cores. hera's relax lane takes the runs
iapetus may not have first. The 60,000-atom cap is about 13 h of iapetus time.

## The pilot

2026-10-22/23, runs store `se-families-hera-20261022-pilot/`. It ran the 20
best-ranked genes from each of three runs (Li-Y-Cl, Li-P-S-Cl, Li-La-Zr-O), plus two
known LeMat genes as controls in each of the first two. All 1,910 PyXtal trials were
answered (3 timed out) and no relaxation failed. The Li-P-S-Cl relaxation went through
the iapetus lane and its outputs were scored on hera.

| system | genes | e_hull ≤ 0 | ≤ 0.1 | ≤ 0.2 | median e_hull | CPU-s / atom | ORB-s / atom |
|---|---|---|---|---|---|---|---|
| Li-Y-Cl | 20 | 0 | 19 | 20 | 0.055 | 1.61 | 0.93 |
| Li-P-S-Cl | 20 | 0 | 6 | 19 | 0.136 | 1.69 | 3.91 (partly iapetus) |
| Li-La-Zr-O | 20 | 0 | 1 | 19 | 0.155 | 1.74 | 1.20 |

e_hull is in eV/atom, the lower of the free and fixed-symmetry relaxations per gene. All
60 came out as novel structures.

The controls do not return to the hull. LiYCl4 lands at 0.040, Li3YCl6 at 0.045,
Li5P(S2Cl)2 at 0.028 and Li6PS5Cl at 0.033 eV/atom, so a known phase relaxed from its own
gene sits a few hundredths of an eV/atom above the ORB hull in these chemistries. That
makes the 0.1 eV/atom tier the meaningful one. Only Li5P(S2Cl)2 was matched to its LeMat
entry; the other three were scored as novel structures. Structure-level novelty therefore
misses known compounds after this relaxation, and the gene-level novelty and OBELiX screens
are what exclude known materials. The report applies both, and re-checks OBELiX on the
relaxed formula.

## Results

### Selection funnel

Generated 2026-10-22, screened and selected 2026-10-23 (commit `ffcee91`); per-system
counts are in the campaign's `selected/funnel_by_system.csv`. Each column counts unique
genes passing every screen to its left. "unique, exact system" drops draws whose gene
lacks one of the requested elements.

| family | systems | draws | unique, exact system | gene novel | not in OBELiX | charge balanced | gap ≥ 0.5 eV | selected | atoms |
|---|---|---|---|---|---|---|---|---|---|
| halide | 42 | 840,000 | 145,961 | 145,083 | 143,878 | 18,236 | 15,337 | 8,985 | 274,172 |
| oxyhalide | 10 | 200,000 | 112,552 | 112,549 | 112,549 | 18,686 | 13,455 | 2,610 | 119,616 |
| argyrodite | 9 | 180,000 | 44,447 | 44,439 | 44,397 | 6,768 | 6,488 | 3,050 | 93,035 |
| thio | 9 | 180,000 | 59,815 | 59,507 | 58,060 | 6,688 | 5,686 | 2,158 | 104,537 |
| lgps | 3 | 60,000 | 29,949 | 29,948 | 29,890 | 4,826 | 4,722 | 878 | 35,870 |
| antiperovskite | 6 | 120,000 | 60,115 | 59,938 | 59,619 | 8,832 | 7,595 | 2,241 | 70,671 |
| fluoride | 10 | 200,000 | 54,678 | 54,433 | 54,283 | 7,803 | 7,422 | 2,429 | 111,813 |
| phosphide | 6 | 120,000 | 37,198 | 36,965 | 36,914 | 2,555 | 620 | 570 | 20,803 |
| nitride | 9 | 180,000 | 65,757 | 65,590 | 65,255 | 6,000 | 5,274 | 2,350 | 105,769 |
| garnet_perovskite | 6 | 120,000 | 50,038 | 50,016 | 49,647 | 6,231 | 4,621 | 807 | 71,770 |
| nasicon | 5 | 100,000 | 45,080 | 44,979 | 44,844 | 8,388 | 5,625 | 754 | 59,663 |
| lisicon_oxide | 8 | 160,000 | 78,753 | 78,489 | 77,292 | 9,419 | 7,946 | 1,446 | 95,361 |
| hydride | 3 | 60,000 | 34,305 | 34,227 | 33,897 | 1,601 | 1,530 | 573 | 31,694 |
| **total** | 126 | 2,520,000 | 818,648 | 816,163 | 810,525 | 106,033 | 86,321 | 28,851 | 1,194,774 |

- Novelty removes few unique genes but many draws: the known genes are the ones the model
  draws often, so 14% of all draws (339,149 of 2,456,396) are of genes already in LeMat-Bulk.
- Charge balance is the main cut: 13% of the unique novel genes pass it.
- The band-gap screen passes 81% of the charge-balanced genes overall, but only 24% of
  the phosphides.
- Of the 126 systems, 67 stop at the 12,000-atom budget and 6 at 400 genes. The other 53
  run out of eligible genes first, mostly through the 40-per-formula cap: 40 of the 42 halide
  systems, the phosphides, and a few sulfides, fluorides and hydrides. In all, 28,851
  genes and 1.19M atoms go to 15 protocol runs.

### Relaxation and hull

The run started on 2026-10-23 at 06:57. It was restarted at 08:12 with two CPU lanes, and
halide_Cl's prescreen resumed from its 40,484 finished trials. The last score finished on
2026-10-24 at 14:03, about 30 h after the restart.

All 15 runs completed every stage:
- No PyXtal, prescreen or relax trial went unanswered. 16 of the 86,022 ORB relaxations
  failed: 11 in lgps, 4 in phosphide and 1 in hydride.
- iapetus relaxed phosphide (3.8 h) and lgps. hera took nasicon while iapetus was busy.
- Measured in relaxed atoms, ORB ran at ~15k atoms/h on iapetus. On hera it ran at 72k
  (hydride) to ~157k atoms/h (halide_Cl), so the cost per atom differs ~2× by chemistry.
- Phosphide's `score` failed once, at 12:48 on 2026-10-23. A host cuDSS upgrade had removed
  the library Torch links against (see the
  [hera notes](platforms/hera/environment.md#cudss-10-upgrade-2026-10-23)). It was re-run
  by hand at 13:03, so the driver's closing `Protocol runs with a failed stage:
  ['phosphide']` is stale.

The report ran on 2026-10-24 (`report --wandb`, commit `e69b91d`). Everything it read
and wrote is on the W&B run
[se-families-hera-20261022](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/se-families-hera-20261022):
- `se-families-hera-20261022-screens:v0`: the per-system screens and the selection;
- `-protocol:v0`: each run's protocol outputs and CIFs;
- `-candidates:v0`: `candidates.csv`, `excluded.csv`, `shortlist.csv`,
  `tiers_by_family.csv`, `scored.parquet` and the free-track candidate CIFs, filed by
  family and system.

It adds three checks to the novelty and OBELiX screens. Each one turned out to be needed once the relaxed structures were in hand.

- **Family anion roles.** The selection's charge balance only asks that every anion be
  more electronegative than every cation. That admits N+3 and S+2 next to a halide, and
  halogen oxoanions in Li-O-X. LiN2Cl11, LiNCl4 and LiSI3 were the three lowest e_hull
  structures, followed by iodates and periodates (LiIO3, Li5IO6). These are molecular
  N-Cl and S-I solids and oxidisers, not the family's chemistry. The report now requires
  the family's anions to be negative and every other element positive
  (`FAMILY_ANIONS`). Amide-borohydrides are the exception: H next to N takes either sign.
- **Prototypes by name.** OBELiX lacks several of the families' own prototypes, including
  Li9AlP4, Li2ZrCl6, Li3GaF6 and Li2ZrF6. Their new polymorphs were reaching the
  candidate list.
- **Composition novelty.** Gene novelty and structure novelty say nothing about the
  formula. Each candidate is flagged as either a new polymorph of a composition the
  novelty reference has, or a new composition. Such a composition is new to LeMat-Bulk,
  not necessarily to the literature: Li9AlP4 is a published SE that is absent from LeMat.

On the free track, these checks removed 1,011 structures at e_hull ≤ 0.1: 752 not ionic
in their family and 259 prototype polymorphs. Structures that pass are valid, unique,
novel, not in OBELiX, not a prototype and ionic. Here is the free-relaxation track (the
fixed-symmetry track is in `report/tiers_by_family.csv`):

| family | relaxed | valid, unique, novel | e_hull ≤ 0 | ≤ 0.1 | ≤ 0.2 | compositions ≤ 0.1 | new compositions ≤ 0.1 |
|---|---|---|---|---|---|---|---|
| halide | 8,984 | 8,966 | 106 | 8,368 | 8,821 | 780 | 685 |
| oxyhalide | 2,609 | 2,608 | 5 | 1,250 | 2,336 | 346 | 344 |
| argyrodite | 3,050 | 3,050 | 1 | 986 | 2,429 | 405 | 397 |
| thio | 2,158 | 2,120 | 8 | 1,525 | 2,039 | 244 | 214 |
| lgps | 877 | 877 | 0 | 187 | 731 | 101 | 100 |
| antiperovskite | 2,240 | 2,223 | 0 | 1,030 | 1,273 | 128 | 108 |
| fluoride | 2,429 | 2,392 | 16 | 2,209 | 2,313 | 270 | 234 |
| phosphide | 569 | 569 | 49 | 291 | 503 | 43 | 28 |
| nitride | 2,349 | 2,311 | 35 | 697 | 1,126 | 95 | 71 |
| garnet_perovskite | 804 | 802 | 0 | 266 | 772 | 88 | 80 |
| nasicon | 754 | 754 | 4 | 391 | 697 | 151 | 135 |
| lisicon_oxide | 1,446 | 1,420 | 23 | 896 | 1,348 | 148 | 110 |
| hydride | 573 | 573 | 0 | 257 | 419 | 53 | 50 |
| **total** | 28,842 | 28,665 | 247 | 18,353 | 24,807 | 2,852 | 2,556 |

e_hull is in eV/atom against the LeMat-Bulk ORB hull. "Relaxed" counts genes with a
structure, so 9 of the 28,851 selected genes gave none.

**Reading the tiers.**
- **≤ 0.1 is 64% of the relaxed genes.** The selection had already ranked them by
  predicted hull distance. In a ternary family, single-valence charge balance puts every
  composition on the join of two binaries (LiCl–YCl3, Li2S–GeS2), where ordered
  arrangements of the two end members are often only slightly above the hull.
- **≤ 0 is mostly polymorphs of known compositions.** Of the 247 such structures, 54 (23
  compositions) are new compositions. The other 193 claim a polymorph below LeMat's own
  phase at the same composition. In the pilot, known phases relaxed from their own genes
  sat 0.03–0.045 eV/atom *above* this hull. If that holds beyond the pilot's chemistries,
  a known composition below the hull is at least that far under the ORB-relaxed known
  phase, which is more likely an ORB or reference artefact than a new ground state. The
  phosphides point the same way. LeMat's own DFT hull puts its entries for these
  compositions 0.1–0.3 eV/atom above it (Li3AlP2, Li3GaP2, Li3Al2P3), although Li3AlP2
  and Li3GaP2 are known compounds, so the Li–M–P reference itself looks poorly
  converged.

**Li sites.** `report/shortlist.csv` keeps one structure per composition: the lowest free
relaxation at ≤ 0.1, 2,852 in all. Each has the CrystalNN coordination of every
symmetry-distinct Li site, under its ionic oxidation states with `cation_anion=True`.
- 891 of the 2,852 (31%) have a Li site with CN ≤ 3. That is normal for nitrides (Li3N
  has 2- and 3-coordinated Li), and 80 of the 95 nitride compositions are among them.
  Halides, oxides and sulfides rarely have Li this low, yet 75 of 128 antiperovskite and
  80 of 151 NASICON compositions do.
- 1,714 compositions are new and have every Li site at CN ≥ 4. 872 of these are within
  0.05 eV/atom and 19 are at or below the hull.
- 1,247 of the 2,852 free relaxations come out P1 at `symprec=0.1`. The free relaxation
  broke the gene's symmetry, so the gene-level band gap describes the structure it
  started from.

**Best new compositions per family.** Each family's three lowest free-relaxation
structures that are new compositions with every Li site at CN ≥ 4:

| family | formula | e_hull free | e_hull fixed-sym | SG gene → relaxed | atoms | gap GGA / r2SCAN (eV) | Li CN |
|---|---|---|---|---|---|---|---|
| halide | LiIn2I7 | −0.038 | −0.018 | 12 → P1 | 20 | 1.9 / 2.4 | 4 |
| halide | Li2TaI7 | −0.024 | −0.024 | 5 → C2 | 20 | 0.6 / 1.8 | 6 |
| halide | LiGa2Br7 | −0.008 | +0.166 | 12 → P1 | 20 | 2.9 / 3.8 | 4,5 |
| oxyhalide | LiZrBr3O | −0.012 | −0.012 | 62 → Imma | 24 | 2.8 / 3.4 | 6 |
| oxyhalide | LiZrCl3O | −0.009 | −0.005 | 15 → P2₁/c | 48 | 4.2 / 4.3 | 6 |
| oxyhalide | Li8TaClO6 | +0.001 | +0.001 | 148 → R-3 | 48 | 4.5 / 4.6 | 4,6 |
| argyrodite | Li6Si(S2Cl)2 | −0.004 | −0.004 | 38 → Amm2 | 26 | 3.0 / 4.3 | 4,5 |
| argyrodite | Li5Ge2S6Cl | +0.001 | +0.001 | 1 → P1 | 14 | 2.2 / 3.2 | 4 |
| argyrodite | Li5SiS4I | +0.004 | +0.004 | 36 → Cmc2₁ | 44 | 2.7 / 4.1 | 4,5 |
| thio | Li4Sb2S5 | −0.009 | −0.003 | 33 → Pbam | 44 | 1.9 / 2.5 | 4 |
| thio | Li6SiS5 | +0.002 | +0.002 | 2 → P-1 | 24 | 2.8 / 3.4 | 4,5 |
| thio | Li5GaS4 | +0.004 | +0.004 | 60 → Cmcm | 40 | 2.7 / 3.6 | 4 |
| lgps | Li2GeP2S7 | +0.024 | +0.080 | 5 → P1 | 24 | 2.3 / 2.8 | 4,5 |
| lgps | Li3GePS6 | +0.029 | +0.062 | 40 → Cc | 44 | 2.4 / 2.9 | 4 |
| lgps | Li7Si2PS10 | +0.037 | +0.042 | 105 → P1 | 40 | 2.4 / — | 4,6 |
| antiperovskite | Li8S3Cl2 | +0.016 | +0.016 | 12 → P-3m1 | 26 | 2.4 / 4.5 | 4 |
| antiperovskite | Li7S3Cl | +0.018 | +0.026 | 63 → Pnma | 44 | 2.2 / 5.2 | 4 |
| antiperovskite | Li8S3Br2 | +0.019 | +0.019 | 166 → R-3m | 39 | 2.0 / 3.9 | 4 |
| fluoride | Li4InF7 | −0.020 | −0.020 | 146 → F-43m | 36 | 4.7 / 5.8 | 4 |
| fluoride | LiHf2F9 | −0.016 | +0.145 | 165 → P-1 | 24 | 2.7 / 8.0 | 6 |
| fluoride | Li5InF8 | −0.005 | −0.005 | 164 → P-3m1 | 14 | 5.4 / 7.2 | 4,6 |
| phosphide | Li6GaP3 | −0.010 | +0.045 | 9 → Cm | 40 | 1.3 / 1.7 | 4,6 |
| phosphide | Li3Al4P5 | +0.045 | +0.045 | 12 → R-3m | 24 | 0.6 / 1.4 | 4,6 |
| phosphide | Li6InP3 | +0.079 | +0.079 | 166 → R-3m | 30 | 0.6 / 0.5 | 4,6 |
| nitride | Li8Br5N | +0.013 | +0.013 | 15 → C2/c | 56 | 2.3 / 0.5 | 4 |
| nitride | Li3B2N3 | +0.027 | +0.128 | 62 → Cmcm | 32 | 3.8 / 2.8 | 4 |
| nitride | Li3Al3N4 | +0.031 | +0.031 | 205 → Pa-3 | 80 | 3.3 / 3.0 | 4 |
| garnet_perovskite | LiLaTi3O8 | +0.017 | +0.019 | 15 → P2₁/c | 52 | 2.7 / 2.7 | 4 |
| garnet_perovskite | LiLaTi4O10 | +0.018 | +0.017 | 15 → C2/c | 64 | 3.0 / 3.5 | 4 |
| garnet_perovskite | LiLaTa4O12 | +0.027 | +0.027 | 5 → C2 | 36 | 1.7 / 3.8 | 4 |
| nasicon | Li2Hf(PO4)2 | −0.004 | +0.129 | 15 → C2/c | 52 | 5.0 / 5.9 | 4 |
| nasicon | Li2Ge(PO4)2 | −0.001 | −0.001 | 2 → C2/m | 13 | 3.4 / 5.0 | 4 |
| nasicon | Li2Ge(PO3)4 | −0.000 | +0.014 | 41 → Cc | 76 | 5.0 / 4.2 | 4 |
| lisicon_oxide | Li2P4O11 | −0.022 | −0.022 | 15 → C2/c | 68 | 5.6 / 6.3 | 4 |
| lisicon_oxide | LiP3O8 | −0.019 | −0.019 | 4 → P2₁ | 24 | 5.6 / 6.3 | 4 |
| lisicon_oxide | Li4Ta2O7 | −0.005 | −0.006 | 7 → C2/m | 26 | 3.9 / 4.2 | 5,6 |
| hydride | Li5AlH8 | +0.010 | +0.088 | 12 → P1 | 28 | 3.5 / 5.6 | 4,6 |
| hydride | LiB(H2N)4 | +0.010 | +0.011 | 88 → P2 | 56 | 5.1 / 5.1 | 4 |
| hydride | Li4AlH7 | +0.018 | +0.062 | 8 → P1 | 24 | 3.6 / 5.3 | 5,6 |

"Gap" is the gene-level prediction of the GGA (`3t16iys0`) and r2SCAN (`xy2dr9us`)
regressors; "—" means the gene is outside the r2SCAN regressor's vocabulary.

When the free and fixed-symmetry e_hull agree, the gene's own symmetry is the
low-energy structure. When the free one is much lower (LiGa2Br7, LiHf2F9,
Li2Hf(PO4)2), the gene was only a starting point.

None of this has been checked against ICSD or DFT. The table is a shortlist for that
work, not a set of discoveries. In order, the next steps would be:
1. a literature/ICSD check of the formulas;
2. r2SCAN or PBE relaxations of the shortlist;
3. electrochemical windows and Li-migration barriers for the survivors.
