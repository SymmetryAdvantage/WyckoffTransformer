# Which energy corrections MP-20 carries

> **STATUS: MEASUREMENT, 2026-09-26.** Repo at `c92fa82` (main), script
> `scripts/check_mp20_energy_corrections.py`, run with pymatgen 2026.9.24 and mp-api
> 0.46.5 in a throwaway venv. Materials Project database `2026.04.13` (new API); the legacy
> REST API reported database `2020_09_08`. No W&B runs involved.

## Answer

MP-20's `formation_energy_per_atom` and `e_above_hull` are Materials Project GGA/GGA+U
values **with the `MaterialsProject2020Compatibility` (MP2020) anion and +U corrections
applied**. They are neither raw energies nor the legacy `MaterialsProjectCompatibility`
scheme. MP-20 ships no total or uncorrected energy, so the corrections cannot be undone
from the dataset itself.

So MP-20 energies are on the same scale as the MP2020-corrected DFT energies of the
WyFormer DFT samples (`dft_e_corrected`), and **not** on the scale of LeMat-Bulk's raw
PBE(+U) energies; see [`e_hull_definitions.md`](../e_hull_definitions.md).

## What the original sources say

Nothing about corrections. The [CDVAE `data/mp_20` README](https://github.com/txie-93/cdvae/tree/main/data/mp_20)
(copied here as `data/mp_20/README.md`) and the [CDVAE paper](https://arxiv.org/abs/2110.06197)
(ICLR 2022, App. C.3) give only the filters: ICSD-sourced, at most 20 atoms,
`e_above_hull` < 0.08 eV/atom, formation energy < 2 eV/atom, "following Ren et al. (2020)".

That reference gives away the provenance. MP-20's CSV columns — `material_id,
formation_energy_per_atom, band_gap, pretty_formula, e_above_hull, elements, cif,
spacegroup.number` — are exactly the `query_properties` of
[FTCP's `data.py`](https://github.com/PV-Lab/FTCP/blob/master/data.py), which queries MP
through matminer's `MPDataRetrieval`, i.e. the legacy REST API. The values are therefore
whatever that API served on the day of the query. The CDVAE repository's only commit to
`data/mp_20` is `73874c4`, 2021-12-09; the paper's arXiv v1 is from October 2021.

## Method

Three independent comparisons; all three stages are in the script.

1. **Today's MP GGA_GGA+U thermo docs** (new API; MP2020 corrections). Fetched for all
   45,229 MP-20 ids: 42,773 have a GGA_GGA+U doc, 2,456 do not (deprecated or merged). The new
   API returns AlphaIDs, which are base-26 renderings of the classic integer
   (`mp-aaaaabwr` = `mp-1265`).
2. **Recomputation under both schemes from the same raw energies.** Each thermo doc
   carries its `ComputedStructureEntry`. The script strips its `energy_adjustments`, reapplies
   either MP2020 or the legacy scheme, and takes the formation energy against that scheme's
   own elemental references: the lowest corrected energy per atom among the 791 GGA_GGA+U
   elemental docs. Recomputing both from one energy means DFT recalculations since 2021
   affect the two schemes equally.
3. **The legacy REST API**, queried with a legacy (17-character) key for all MP-20 ids;
   44,780 are served. New-style 32-character keys are rejected with a 403.

## Results

MP-20's `formation_energy_per_atom` against each reference, eV/atom:

| reference | n | median \|Δ\| | \|Δ\| < 1 meV | \|Δ\| < 10 meV |
|---|---|---|---|---|
| today's MP GGA_GGA+U doc, as served | 42,773 | 0.0000 | 83.4% | 92.1% |
| recomputed, MP2020 | 42,773 | 0.0000 | **89.1%** | **98.1%** |
| recomputed, legacy scheme | 42,773 | 0.0087 | 41.7% | 56.8% |
| legacy REST API, as served today | 44,780 | 0.0099 | 21.1% | 47.1% |

**Where the two schemes disagree, MP2020 wins.** The recomputed schemes differ by more
than 5 meV/atom on 23,025 rows, and MP2020 is the closer one on 98.9% of those. By chemistry:

| subset (of the 23,025) | n | MP2020 closer | median err, MP2020 | median err, legacy |
|---|---|---|---|---|
| contains O | 10,109 | 98.3% | 0.0000 | 0.0102 |
| contains Fe | 1,451 | 99.6% | 0.0000 | 0.0802 |
| contains Br, I, Se, Si, Sb or Te | 8,368 | 99.4% | 0.0000 | 0.1180 |

The last row is the sharpest test. MP2020 added anion corrections for those species and the
legacy scheme has none, so a legacy-corrected dataset would sit about 0.1 eV/atom higher
on them. Examples (MP-20 / recomputed MP2020 / recomputed legacy):

| material | MP-20 | MP2020 | legacy |
|---|---|---|---|
| MgO `mp-1265` | −3.0534 | −3.0534 | −3.0610 |
| Fe₂O₃ `mp-19770` | −1.7071 | −1.7071 | −1.9071 |
| NiO `mp-19009` | −1.2181 | −1.2181 | −1.0372 |
| ZnSe `mp-1190` | −0.9515 | −0.9516 | −0.7156 |
| CsI `mp-614603` | −1.7354 | −1.7354 | −1.5459 |
| Sb₂Te₃ `mp-1201` | −0.3832 | −0.3832 | −0.1300 |

`e_above_hull` matches today's GGA_GGA+U doc within 1 meV/atom for 92.0% of rows (95.2%
within 10 meV).

**The legacy API now serves legacy-scheme values, and they do not match MP-20.** Against
the two recomputations, the legacy API's formation energies sit with the legacy scheme
(median |Δ| 0.0006, 55.1% within 1 meV) rather than MP2020 (median 0.0099, 20.3%). Its Fe₂O₃
is −1.9074 and its MgO −3.0624. Its `band_gap` matches MP-20 exactly for 94.1% of rows, so
it is serving the same materials. What differs is the energy correction.

## Dating the snapshot

MP's [release notes](https://docs.materialsproject.org/changes/database-versions)
introduce MP2020 in database v2021.05.13, and state that the legacy website and API were
frozen at that release when v2021.11.10 moved to the new API. MP-20 must therefore
have been pulled between 2021-05-13 and the CDVAE commit of 2021-12-09.

As of 2026-09-26, though, the legacy API reports database `2020_09_08` and serves pre-MP2020
values, contrary to those release notes. It can no longer reproduce MP-20. The legacy
website's material pages disagreed in the same way during this check: MgO showed
−3.062 eV/atom, the legacy-scheme value, although the page listed the MP2020 −0.687 eV/atom
oxide adjustment. **Do not use either legacy source as ground truth for MP-20
energies.** The GGA_GGA+U thermo docs of the current API are the closest surviving
reference.

## What was not checked

- **The residual ~2% of rows that miss MP2020 by more than 10 meV/atom.** Presumably these
  are materials recalculated or re-blessed since 2021; not investigated.
- **The 2,456 ids without a current GGA_GGA+U doc.** They enter only the legacy-API
  comparison; 19 ids are in neither API.
- **MP-20's row count.** It is 45,229 rows, not the README's 45,231; the difference was not
  investigated.
