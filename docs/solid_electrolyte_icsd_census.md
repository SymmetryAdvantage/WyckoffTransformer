# ICSD census for the Li solid-electrolyte family campaigns

> **Scope:** exact-element-set ICSD census for every chemical system sampled by the two completed Li solid-electrolyte generative campaigns: the 126-system OBELiX-origin campaign and the 19-system Liverpool Ionics extension. A hit must contain exactly the requested element set. Unary and binary subsystem records are excluded from ternary-and-higher requested systems, and supersystems are excluded as well.

> **Generated:** 2026-11-10 from campaign YAMLs pinned in this branch at `3d78f832141a9ec301a84aa0560043129ada5b0f`. The source metadata export is identified by SHA-256 `bcae4ddcaef7cce41529c4c98a64e76491523d2b1e51c16f82edf94284146ea6`.

## Executive result

The two YAMLs expand to **145 unique exact chemical systems**: **126** from the OBELiX-origin campaign and **19** from the Liverpool extension. The canonical, deduplicated valid-reference ICSD metadata export contains **484 exact-system candidate records** across **95 sampled systems**: **229 ordered** and **255 disordered**.

| campaign group | sampled exact systems | all exact-system records | ordered | disordered |
|---|---|---|---|---|
| OBELiX-origin campaign | 126 | 363 | 193 | 170 |
| Liverpool Ionics extension | 19 | 121 | 36 | 85 |

## Sources and boundary

- Campaign-system manifests: `yamls/campaigns/solid_electrolyte_families_hera_20261022.yaml` and `yamls/campaigns/solid_electrolyte_families_hera_20261103.yaml`.
- ICSD input: the supplied canonical `ICSD2025_valid_reference_unique_Xtal_family_dic.csv` metadata export. It is a **deduplicated valid-reference working corpus**, not a raw full ICSD release: it has 103,326 retained records, comprising 58,114 with `is_disorder=False` and 45,212 with `is_disorder=True`. Thus this census includes both ordered and disordered ICSD records present in that corpus, but does **not** claim raw-catalogue coverage of a separate approximately 200k-entry ICSD release.
- The raw source contains a `cif` column, but no CIF data or other structural payload is exported here. The published detail table retains only CollectionCode, reported formula, SWORD label, explicit ordered/disordered status, and a mineral name when the input header supplies one.
- The supplied input header has no mineral-name field. Accordingly, `mineral_name` is blank in the detailed table rather than inferred from a formula or structure label.

## Machine-readable tables

- [`sampled_chemsys.csv`](data/solid_electrolyte_icsd_census/sampled_chemsys.csv): every exact system expanded from the two campaign YAMLs, with group and family provenance.
- [`icsd_exact_chemsys_summary.csv`](data/solid_electrolyte_icsd_census/icsd_exact_chemsys_summary.csv): one row per sampled system, including zero-hit systems.
- [`icsd_exact_chemsys_records.csv`](data/solid_electrolyte_icsd_census/icsd_exact_chemsys_records.csv): one non-CIF metadata row per exact-system ICSD candidate.
- [`manifest.json`](data/solid_electrolyte_icsd_census/manifest.json): input hashes, row counts, field contract, and generation provenance.

## Exact-system summary

The full one-row-per-system table is in the linked CSV. This report retains the same campaign-table style while separating the original and extension scopes.

### OBELiX-origin campaign

| family | exact chemical system | all ICSD records | ordered | disordered |
|---|---|---|---|---|
| antiperovskite | Li-O-Br | 3 | 3 | 0 |
| antiperovskite | Li-O-Cl | 3 | 3 | 0 |
| antiperovskite | Li-O-I | 6 | 6 | 0 |
| antiperovskite | Li-S-Br | 0 | 0 | 0 |
| antiperovskite | Li-S-Cl | 0 | 0 | 0 |
| antiperovskite | Li-S-I | 0 | 0 | 0 |
| argyrodite | Li-Ge-S-Br | 0 | 0 | 0 |
| argyrodite | Li-Ge-S-Cl | 0 | 0 | 0 |
| argyrodite | Li-Ge-S-I | 0 | 0 | 0 |
| argyrodite | Li-P-S-Br | 7 | 0 | 7 |
| argyrodite | Li-P-S-Cl | 5 | 1 | 4 |
| argyrodite | Li-P-S-I | 4 | 2 | 2 |
| argyrodite | Li-Si-S-Br | 0 | 0 | 0 |
| argyrodite | Li-Si-S-Cl | 0 | 0 | 0 |
| argyrodite | Li-Si-S-I | 0 | 0 | 0 |
| fluoride | Li-Al-F | 2 | 2 | 0 |
| fluoride | Li-Ga-F | 1 | 1 | 0 |
| fluoride | Li-Hf-F | 1 | 1 | 0 |
| fluoride | Li-In-F | 2 | 1 | 1 |
| fluoride | Li-Mg-F | 0 | 0 | 0 |
| fluoride | Li-Nb-F | 2 | 2 | 0 |
| fluoride | Li-Sc-F | 2 | 2 | 0 |
| fluoride | Li-Ta-F | 1 | 1 | 0 |
| fluoride | Li-Y-F | 2 | 2 | 0 |
| fluoride | Li-Zr-F | 5 | 5 | 0 |
| garnet_perovskite | Li-La-Hf-O | 1 | 1 | 0 |
| garnet_perovskite | Li-La-Nb-O | 13 | 2 | 11 |
| garnet_perovskite | Li-La-Sn-O | 1 | 1 | 0 |
| garnet_perovskite | Li-La-Ta-O | 10 | 3 | 7 |
| garnet_perovskite | Li-La-Ti-O | 21 | 2 | 19 |
| garnet_perovskite | Li-La-Zr-O | 4 | 1 | 3 |
| halide | Li-Al-Br | 0 | 0 | 0 |
| halide | Li-Al-Cl | 2 | 1 | 1 |
| halide | Li-Al-I | 0 | 0 | 0 |
| halide | Li-Bi-Br | 0 | 0 | 0 |
| halide | Li-Bi-Cl | 0 | 0 | 0 |
| halide | Li-Bi-I | 0 | 0 | 0 |
| halide | Li-Er-Br | 1 | 0 | 1 |
| halide | Li-Er-Cl | 1 | 0 | 1 |
| halide | Li-Er-I | 0 | 0 | 0 |
| halide | Li-Ga-Br | 2 | 2 | 0 |
| halide | Li-Ga-Cl | 2 | 2 | 0 |
| halide | Li-Ga-I | 2 | 2 | 0 |
| halide | Li-Hf-Br | 0 | 0 | 0 |
| halide | Li-Hf-Cl | 0 | 0 | 0 |
| halide | Li-Hf-I | 0 | 0 | 0 |
| halide | Li-In-Br | 1 | 0 | 1 |
| halide | Li-In-Cl | 5 | 0 | 5 |
| halide | Li-In-I | 1 | 1 | 0 |
| halide | Li-La-Br | 0 | 0 | 0 |
| halide | Li-La-Cl | 0 | 0 | 0 |
| halide | Li-La-I | 0 | 0 | 0 |
| halide | Li-Lu-Br | 0 | 0 | 0 |
| halide | Li-Lu-Cl | 0 | 0 | 0 |
| halide | Li-Lu-I | 0 | 0 | 0 |
| halide | Li-Mg-Br | 3 | 2 | 1 |
| halide | Li-Mg-Cl | 4 | 0 | 4 |
| halide | Li-Mg-I | 0 | 0 | 0 |
| halide | Li-Nb-Br | 0 | 0 | 0 |
| halide | Li-Nb-Cl | 3 | 2 | 1 |
| halide | Li-Nb-I | 0 | 0 | 0 |
| halide | Li-Sc-Br | 0 | 0 | 0 |
| halide | Li-Sc-Cl | 2 | 0 | 2 |
| halide | Li-Sc-I | 1 | 1 | 0 |
| halide | Li-Ta-Br | 0 | 0 | 0 |
| halide | Li-Ta-Cl | 0 | 0 | 0 |
| halide | Li-Ta-I | 0 | 0 | 0 |
| halide | Li-Y-Br | 3 | 0 | 3 |
| halide | Li-Y-Cl | 3 | 0 | 3 |
| halide | Li-Y-I | 1 | 0 | 1 |
| halide | Li-Zr-Br | 0 | 0 | 0 |
| halide | Li-Zr-Cl | 0 | 0 | 0 |
| halide | Li-Zr-I | 0 | 0 | 0 |
| hydride | Li-Al-H | 0 | 0 | 0 |
| hydride | Li-B-H | 0 | 0 | 0 |
| hydride | Li-B-N-H | 0 | 0 | 0 |
| lgps | Li-Ge-P-S | 4 | 0 | 4 |
| lgps | Li-Si-P-S | 5 | 1 | 4 |
| lgps | Li-Sn-P-S | 3 | 1 | 2 |
| lisicon_oxide | Li-Al-O | 7 | 4 | 3 |
| lisicon_oxide | Li-B-O | 14 | 11 | 3 |
| lisicon_oxide | Li-Ge-O | 11 | 10 | 1 |
| lisicon_oxide | Li-Nb-O | 25 | 10 | 15 |
| lisicon_oxide | Li-P-O | 11 | 11 | 0 |
| lisicon_oxide | Li-Si-O | 9 | 9 | 0 |
| lisicon_oxide | Li-Ta-O | 14 | 8 | 6 |
| lisicon_oxide | Li-Zr-O | 8 | 4 | 4 |
| nasicon | Li-Ge-P-O | 1 | 1 | 0 |
| nasicon | Li-Hf-P-O | 2 | 0 | 2 |
| nasicon | Li-Sn-P-O | 2 | 2 | 0 |
| nasicon | Li-Ti-P-O | 7 | 5 | 2 |
| nasicon | Li-Zr-P-O | 6 | 2 | 4 |
| nitride | Li-Al-N | 1 | 1 | 0 |
| nitride | Li-B-N | 4 | 3 | 1 |
| nitride | Li-Ga-N | 1 | 1 | 0 |
| nitride | Li-Ge-N | 2 | 1 | 1 |
| nitride | Li-N-Br | 6 | 5 | 1 |
| nitride | Li-N-Cl | 3 | 2 | 1 |
| nitride | Li-N-I | 2 | 1 | 1 |
| nitride | Li-P-N | 6 | 5 | 1 |
| nitride | Li-Si-N | 4 | 2 | 2 |
| oxyhalide | Li-Al-O-Br | 0 | 0 | 0 |
| oxyhalide | Li-Al-O-Cl | 0 | 0 | 0 |
| oxyhalide | Li-Hf-O-Br | 0 | 0 | 0 |
| oxyhalide | Li-Hf-O-Cl | 0 | 0 | 0 |
| oxyhalide | Li-Nb-O-Br | 0 | 0 | 0 |
| oxyhalide | Li-Nb-O-Cl | 3 | 2 | 1 |
| oxyhalide | Li-Ta-O-Br | 0 | 0 | 0 |
| oxyhalide | Li-Ta-O-Cl | 0 | 0 | 0 |
| oxyhalide | Li-Zr-O-Br | 0 | 0 | 0 |
| oxyhalide | Li-Zr-O-Cl | 0 | 0 | 0 |
| phosphide | Li-Al-P | 4 | 2 | 2 |
| phosphide | Li-Ga-P | 3 | 1 | 2 |
| phosphide | Li-Ge-P | 6 | 3 | 3 |
| phosphide | Li-In-P | 2 | 1 | 1 |
| phosphide | Li-Si-P | 7 | 3 | 4 |
| phosphide | Li-Sn-P | 6 | 1 | 5 |
| thio | Li-Al-S | 1 | 1 | 0 |
| thio | Li-As-S | 3 | 3 | 0 |
| thio | Li-B-S | 5 | 2 | 3 |
| thio | Li-Ga-S | 3 | 2 | 1 |
| thio | Li-Ge-S | 2 | 2 | 0 |
| thio | Li-P-S | 10 | 4 | 6 |
| thio | Li-Sb-S | 8 | 5 | 3 |
| thio | Li-Si-S | 2 | 2 | 0 |
| thio | Li-Sn-S | 7 | 4 | 3 |

### Liverpool Ionics extension

| family | exact chemical system | all ICSD records | ordered | disordered |
|---|---|---|---|---|
| molybdate | Li-Cr-Mo-O | 5 | 1 | 4 |
| molybdate | Li-Fe-Mo-O | 3 | 2 | 1 |
| molybdate | Li-Mg-Mo-O | 1 | 0 | 1 |
| molybdate | Li-Mg-V-Mo-O | 1 | 0 | 1 |
| nasicon | Li-Hf-Ta-P-O | 0 | 0 | 0 |
| olivine | Li-Fe-P-O | 30 | 9 | 21 |
| olivine | Li-Mg-In-P-O | 3 | 0 | 3 |
| olivine | Li-Mg-P-O | 3 | 3 | 0 |
| phenakite | Li-Ga-Si-O | 5 | 4 | 1 |
| rocksalt | Li-Co-Cl | 4 | 2 | 2 |
| rocksalt | Li-F | 1 | 1 | 0 |
| rocksalt | Li-Fe-Cl | 8 | 2 | 6 |
| rocksalt | Li-I | 3 | 3 | 0 |
| rocksalt | Li-Mg-Ti-O | 9 | 0 | 9 |
| rocksalt | Li-Mg-Zr-O | 1 | 0 | 1 |
| rocksalt | Li-Ni-Cl | 0 | 0 | 0 |
| rocksalt | Li-Ti-O | 43 | 8 | 35 |
| rocksalt | Li-Ti-Ta-O | 0 | 0 | 0 |
| zircon | Li-Y-P-O | 1 | 1 | 0 |

## Reproduction

The report and data tables were generated without copying the input CIF column:

```bash
python scripts/build_solid_electrolyte_icsd_census.py \
  --icsd-csv /absolute/path/to/ICSD2025_valid_reference_unique_Xtal_family_dic.csv
```

The generator fails if the campaign groups overlap, if the declared 126/19/145 system counts do not reconcile, or if a matched source row has an element set different from its assigned sampled system.
