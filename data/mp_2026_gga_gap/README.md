# mp_2026_gga_gap

147,417 Materials Project entries whose band gap **and** structure were both
computed with GGA or GGA+U, from snapshot `2026.04.13`. Built by
[`scripts/build_mp_gga_band_gap.py`](../../scripts/build_mp_gga_band_gap.py),
whose module docstring carries the reasoning behind each choice below.

## Why the pairing needs care

MP mixes functionals inside a single `summary` document. Of the 154,377
non-deprecated materials in this snapshot the blessed structure comes from
r2SCAN for 37,129 and from SCAN for 939, while the blessed band gap comes from
GGA/GGA+U for 148,330. Taking `summary.structure` next to `summary.band_gap`
therefore hands ~31k rows a meta-GGA geometry with a semilocal gap.

This dataset takes the structure from *the task that produced the gap*, so the
two are the same calculation by construction.

## Selection

| step | rows |
| --- | --- |
| non-deprecated materials | 154,377 |
| with a band gap | 154,373 |
| gap computed with GGA or GGA+U | 148,330 |
| the gap task also carries a structure | 148,330 |
| present in MP's GGA/GGA+U phase diagram | **147,417** |

`energy_above_hull` is an AdaLN conditioning feature in the tokeniser configs,
so the 913 materials outside MP's GGA/GGA+U hull are dropped rather than carried
with a NaN label.

Composition: 105,803 GGA and 41,614 GGA+U; 78,901 rows (53.5%) have a gap above
zero. Splits are random at seed 20260909: 117,933 train, 14,742 val, 14,742 test.

### The cache is cut at 61 Wyckoff sites, the CSVs are not

Every sequence tensor is padded to the longest structure in the split, and the tail
here is long: the median structure has 7 Wyckoff sites, p90 is 28, p99 is 126, and
the maximum is 360. Padding all 147,417 rows to 360 costs 758 MB of tensor for a
handful of structures. Cutting at 61 sites, the same figure `lemat_bulk_fmax1` uses,
drops 5,721 rows (3.88%) and takes the padded width from 361 to 62 and the tensor
cache from 758 MB to 155 MB.

The cut **drops** over-long structures, it does not truncate them: truncating to the
first N Wyckoff positions changes the composition while leaving the band gap attached,
so the label would describe a compound that is not in the row. That is
`cache_a_dataset_reusing.py --max-sites`, not `cache_a_dataset.py --max-wp`.

The CSVs keep all 147,417 rows; the cut lives in the cache, so a different cut can be
made from the same source. Cached splits: 113,339 train, 14,177 val, 14,180 test
(141,696 total).

## Columns

| column | meaning |
| --- | --- |
| `material_id` | index |
| `cif` | the geometry the gap was computed on, P1 |
| `band_gap` | eV, MP's blessed value |
| `dft_run_type` | `GGA` or `GGA+U`, read off the band-gap task |
| `mp_dft_uncorrected_energy_per_atom` | eV/atom, raw VASP energy |
| `mp_dft_energy_per_atom` | eV/atom, after MP's anion and +U corrections |
| `mp_dft_formation_energy_per_atom` | eV/atom |
| `mp_dft_energy_above_hull` | eV/atom |
| `max_force`, `stress` | **all NaN — see below** |
| `max_force_missing`, `stress_missing` | 1.0 everywhere |
| `formation_energy_per_atom`, `energy_above_hull` | duplicates of the `mp_dft_` columns under the names `LEGACY_SCALAR_COLUMNS` and the tokeniser configs expect |

The `mp_dft_` prefix marks provenance: these are MP's own DFT numbers, taken
verbatim, not recomputed here and not from an MLIP.

`dft_run_type` follows the band gap because the gap is the label. MP blesses the
thermo entry independently, so for 596 rows the energies come from the other run
type; the column is not a claim about them.

### Forces and stress are not available from this snapshot

MP's parsed task collection stores forces as a nullable per-site property of
`output.structure` and has no stress tensor anywhere in its 29-field schema. A
count over the whole collection returns **0 of 2,015,403 tasks** with forces
populated, and every one of the 148,330 band-gap tasks pulled here came back
without them. Both columns are written as NaN with the `_missing` indicator set,
following the convention `build_lemat_bulk_fmax.py` uses for LeMat-Bulk's own
force-less MP rows.

LeMat-Bulk *does* carry forces for its MP subset, parsed from raw VASP output at
an older snapshot. Joining those in would mix snapshots silently and is
deliberately not done here.

## Usage

Build the full cache, then cut it to 61 Wyckoff sites by reusing those symmetry
records rather than recomputing them:

```bash
python scripts/cache_a_dataset.py mp_2026_gga_gap --n-jobs 16 \
    --scalar-columns dft_run_type \
        mp_dft_uncorrected_energy_per_atom mp_dft_energy_per_atom \
        mp_dft_formation_energy_per_atom mp_dft_energy_above_hull \
        max_force max_force_missing stress stress_missing
cp cache/mp_2026_gga_gap/data.pkl.gz /tmp/mp_gga_gap_full.pkl.gz
python scripts/cache_a_dataset_reusing.py mp_2026_gga_gap \
    --reuse /tmp/mp_gga_gap_full.pkl.gz \
    --max-sites 61 --no-sort-by-letter --n-jobs 16 \
    --scalar-columns band_gap dft_run_type \
        mp_dft_uncorrected_energy_per_atom mp_dft_energy_per_atom \
        mp_dft_formation_energy_per_atom mp_dft_energy_above_hull \
        max_force max_force_missing stress stress_missing \
        formation_energy_per_atom energy_above_hull
python scripts/tokenise_a_dataset.py mp_2026_gga_gap \
    yamls/tokenisers/lemat_bulk_ehull_sg_multiplicity.yaml --new-tokenizer
```

The first step carries `band_gap`, `formation_energy_per_atom` and
`energy_above_hull` automatically via `LEGACY_SCALAR_COLUMNS`; the second takes an
explicit `--scalar-columns` that *replaces* that default, so they are listed there.
`--no-sort-by-letter` is required because the reused records were built in pyxtal's
own site order — unlike `lemat_bulk_ehull` and `lemat_bulk_fmax1`, this dataset was
not built with `--sort-by-letter`. `--verify-reuse` (200 per split by default)
recomputes a sample and aborts on any mismatch.

`lemat_bulk_ehull_sg_multiplicity.yaml` carries `band_gap` through `no_processing`
as the model target alongside the `energy_above_hull` conditioning feature. A
`no_processing` name missing from a dataset is warned about and skipped rather than
raised, so the LeMat-Bulk caches, which have no band gap, tokenise exactly as before.

## Citation

```
@article{jain2013commentary,
  title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and others},
  journal={APL materials},
  volume={1},
  number={1},
  pages={011002},
  year={2013},
  publisher={American Institute of PhysicsAIP}
}
```
