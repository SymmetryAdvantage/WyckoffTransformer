# mp_20260928_r2scan_gap

122,756 Materials Project entries whose blessed band gap was computed with r2SCAN,
with the structure of that same task, from snapshot `2026.09.28`. Built by
[`scripts/build_mp_gga_band_gap.py`](../../scripts/build_mp_gga_band_gap.py) with
`--functional r2scan`.

> **Licence: mostly non-commercial.** 116,835 rows (95%) are GNoME materials
> (batch `gnome_r2scan_statics`) that MP serves under **CC BY-NC 4.0**; the other
> 5,921 are CC BY 4.0. `license` records which. A model trained on this dataset
> inherits the non-commercial term.

## What it is

MP release 2026.09.28 has 271,325 non-deprecated materials. For 148,443 the blessed
band gap is GGA/GGA+U ([mp_20260928_gga_gap](../mp_20260928_gga_gap/README.md)), for
122,756 r2SCAN, for 122 SCAN. The r2SCAN gaps come from r2SCAN static calculations
(279 of 300 sampled tasks; the rest r2SCAN structure optimisations), and 278 of 300
sampled materials are GNoME's. So this is not "MP at a better functional": it is
mostly GNoME's predicted crystals, which are dominated by intermetallics -- only
24,362 rows (19.8%) have a gap above zero, against 53.5% in the GGA set.

Energies come from MP's `R2SCAN` thermo documents, which carry no correction
(`mp_dft_energy_per_atom` equals `mp_dft_uncorrected_energy_per_atom` on every row);
their DFT setting is `R2SCAN_MP` in `wyckoff_transformer.energy_fields`. Never compare
them with the GGA set's MP2020-corrected energies.

## Selection

| step | rows |
| --- | --- |
| non-deprecated materials | 271,325 |
| with a band gap | 271,321 |
| gap computed with r2SCAN | **122,756** |
| the gap task also carries a structure | 122,756 |
| present in MP's r2SCAN phase diagram | 122,756 |

Splits are random at seed 20260909: 98,204 train, 12,276 val, 12,276 test.

## Usage

```bash
.venv/bin/python scripts/build_mp_gga_band_gap.py --name mp_20260928_r2scan_gap \
    --functional r2scan --workers 8 --allow-other-snapshot
wyformer-cache-dataset mp_20260928_r2scan_gap --n-jobs 11 --max-sites 61 --no-sort-by-letter
python scripts/tokenise_a_dataset.py mp_20260928_r2scan_gap \
    yamls/tokenisers/der_tokenizer_v1.yaml --new-tokenizer
```

## Citation

Materials Project: Jain et al., APL Materials 1, 011002 (2013). GNoME: Merchant et al.,
Nature 624, 80 (2023). r2SCAN in MP: Kingsbury et al., Phys. Rev. Materials 6, 013801
(2022).
