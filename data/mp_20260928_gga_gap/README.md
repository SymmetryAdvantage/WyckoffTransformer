# mp_20260928_gga_gap

147,417 Materials Project entries whose band gap **and** structure were both computed
with GGA or GGA+U, from snapshot `2026.09.28`, CC BY materials only. Built by
[`scripts/build_mp_gga_band_gap.py`](../../scripts/build_mp_gga_band_gap.py) with
`--functional gga --exclude-license BY-NC`; the reasoning behind each choice is in
[mp_2026_gga_gap's README](../mp_2026_gga_gap/README.md) and the script's docstring.

## The same labels as `mp_2026_gga_gap`, under new ids

MP release 2026.09.28 grew from 154,377 to 271,325 non-deprecated materials, but the
117k new ones all have r2SCAN band gaps (mostly GNoME; see
[mp_20260928_r2scan_gap](../mp_20260928_r2scan_gap/README.md)). The GGA/GGA+U side is
unchanged: after dropping the 113 GNoME materials MP serves under CC BY-NC, every
selection count below equals the 2026.04.13 build's, and every
(band gap, run type, uncorrected energy, hull distance) tuple of `mp_2026_gga_gap`'s
141,696 cached rows recurs here (checked 2026-10-23; the other 5,721 rows are the
structures over 61 sites that cache cut). MP has meanwhile moved to new material ids
(`mp-aaab...`), so the two cannot be joined by id, and the random split differs.

## Selection

| step | rows |
| --- | --- |
| non-deprecated materials | 271,325 |
| with a band gap | 271,321 |
| gap computed with GGA or GGA+U | 148,443 |
| licence not BY-NC | 148,330 |
| the gap task also carries a structure | 148,330 |
| present in MP's GGA/GGA+U phase diagram | **147,417** |

105,803 GGA and 41,614 GGA+U; 78,901 rows (53.5%) have a gap above zero. Splits are
random at seed 20260909: 117,933 train, 14,742 val, 14,742 test.

## Columns

As `mp_2026_gga_gap`, plus `license` (`BY-C` on every row).

## Usage

```bash
.venv/bin/python scripts/build_mp_gga_band_gap.py --name mp_20260928_gga_gap \
    --functional gga --exclude-license BY-NC --workers 8 --allow-other-snapshot
wyformer-cache-dataset mp_20260928_gga_gap --n-jobs 11 --max-sites 61 --no-sort-by-letter
python scripts/tokenise_a_dataset.py mp_20260928_gga_gap \
    yamls/tokenisers/der_tokenizer_v1.yaml --new-tokenizer
```

The build needs mp-api >= 0.47.0 for this release (the lock pins 0.46.5 as of
2026-10-23).

## Licence

Materials Project data, CC BY 4.0. Cite Jain et al., APL Materials 1, 011002 (2013).
