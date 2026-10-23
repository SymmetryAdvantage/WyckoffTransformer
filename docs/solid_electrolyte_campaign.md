# Generative search for new Li solid electrolytes

> **Scope:** generate crystals in the chemical families known to contain Li solid
> electrolytes (SEs), keep only those that are new and not known SEs, and find which
> of them are near the convex hull. Campaign `se-families-hera-20261022`, run on
> hera (and iapetus for part of the ORB relaxation) from 2026-10-22, branch
> `se-families-campaign`.
>
> **Status:** in progress; results are added below as stages finish.

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

Both models are staged for the Hub with model cards
(`SymmetryAdvantage/WyFormer-MP-BandGap-PBE`, CC BY 4.0, and
`SymmetryAdvantage/WyFormer-MP-BandGap-r2SCAN`, CC BY-NC 4.0) in the campaign's
`hf_stage/`. They are not uploaded yet: hera's Hugging Face token is read-only.

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
stages), then `score`. CPU stages of the next run overlap the relaxation of the current
one. iapetus relaxes runs of up to 2,500 genes through its own launcher, and the outputs
sync back to hera for scoring. Scoring reads only the aggregated `structures*.csv` and
`cifs*/`, so the round trip does not affect it.

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

Running since 2026-10-23 06:57 on hera, with iapetus taking the runs of up to 2,500 genes.
At the pilot's rates the whole run takes about two days.
