# Rebuilding the LeMat-Bulk datasets

The canonical order for turning the LeMat-Bulk archive into a trainable WyFormer dataset,
and what the variants already on disk were built with. If you have cloned this repository
and want to recreate the data, follow this file rather than reconstructing a command line
from the other documents — several of them describe how a particular dataset was built at
the time, which is not the same as how it should be built now.

The single most important step is **[3](#3-recover-the-materials-project-forces)**: 30,679
rows arrive from HuggingFace with an empty `forces` array, and a build that skips the
recovery either drops them or trains on a fabricated `max_force`. Both outcomes are
silently wrong, and both bias the dataset against Materials Project.

## Prerequisites

- The `data` and `cache` stores attached; see [data_store.md](data_store.md).
- `MP_API_KEY` in `.env` at the repository root, for step 3. It is read in process and
  never echoed. The task documents themselves come from a public S3 bucket and need no
  credentials, but identifying *which* task backs each row uses the API.
- Roughly 30 GB of free space in `data` and 20 GB in `cache` for one variant, plus 132 MB
  for the extracted task documents.
- Internet access to HuggingFace, `materialsproject-parsed` on S3, and the MP API.

## The pipeline

### 1. The archive

`data/lemat-bulk/raw/data.parquet` — HuggingFace `LeMaterial/LeMat-Bulk`, config
`compatible_pbe`, 5,335,299 rows. It carries per-atom `forces` (eV/Å) and `stress_tensor`
(**kBar, VASP sign**: positive means compressed).

### 2. Structures and energies

```bash
python scripts/prepare_cif.py                      # raw -> cif_prepared
python scripts/process_lemat.py                    # cif_prepared -> lemat_pbe.csv.gz
python -m wyckoff_transformer.formula_energy.hull_table --workers 16 \
    --input-file data/lemat-bulk/lemat_pbe.csv.gz \
    --output-file data/lemat-bulk/lemat_pbe_ehull.csv.gz          # 1 h 23 min on 16 workers
```

`hull_table` replaces the older `scripts/compute_e_hull.py`, which returned nothing for any
system containing Yb, anything past Po, or a chemsys of ten or more elements — half a
million rows of Yb and actinide chemistry that no cache before 2026-09-07 contained. See
[e_hull_definitions.md](e_hull_definitions.md).

### 3. Recover the Materials Project forces

```bash
python scripts/recover_mp_forces.py all --run full   # ~30 min, 132 MB of task documents
```

30,679 rows — every one from Materials Project, 22.1% of it — have an empty `forces` array
*and* an empty `stress_tensor`. No other row lacks either. They were never missing
upstream: LeMaterial reads the task document's top-level `output.forces`, which MP's
2013–2017 legacy tasks leave empty, while `calcs_reversed[0].output.ionic_steps[-1]` of the
same calculation keeps both. This step reads them back for 30,676 of the rows, matching on
geometry (within 1e-5 Å) *and* total energy (within 1e-5 eV), so the forces provably belong
to the calculation that produced the row's energy. Three rows stay ambiguous.

Skipping it is not neutral:

- **A `max_force <= X` cut drops all 30,679 for every X**, because NaN fails every
  comparison. That is what happened to the `lemat_bulk_ehull` family.
- **Keeping them with an imputed value trains on a fiction.** The source-median stand-in was
  0.0415 eV/Å against a true median of 0.088, and 356 of the rows actually exceed the
  1 eV/Å cut they were being kept inside.

Results land in `cache/mp_forces_recovery/runs/full/results.parquet`. The run checks itself
on 800 rows that were never missing, where it must reproduce the archived forces and stress
bit for bit.

### 4. Labels and splits

```bash
python scripts/build_lemat_bulk_fmax.py --name lemat_bulk_fmax1_stress
```

Reads the recovered forces by default (`--recovered-forces none` reproduces the older
behaviour) and writes `data/lemat-bulk/convergence_labels.parquet` on the way — one row per
structure with `max_force`, the stress invariants, and a `convergence_source` of `archive`,
`mp_task_doc` or `missing`. The split CSVs carry `energy_above_hull`, `delta_e_polymorph`,
`max_force`, `max_force_missing`, `stress_hydrostatic`, `stress_von_mises`,
`stress_missing` and `formation_energy_per_atom`.

Val and test ids are inherited from `cache/lemat_bulk_ehull/split_ids.json` so that
comparisons against earlier runs stay valid. Keep `--max-force 1.0`: see
[unconverged_relaxation_energy.md](unconverged_relaxation_energy.md) for why a tighter cut
is a provenance filter rather than a quality filter.

### 5. Cache and tokenise

```bash
python scripts/cache_a_dataset_reusing.py lemat_bulk_fmax1_stress \
    --reuse cache/lemat_bulk_fmax1/data.pkl.gz \
    --scalar-columns energy_above_hull delta_e_polymorph max_force max_force_missing \
        stress_hydrostatic stress_von_mises stress_missing formation_energy_per_atom \
    --observed-gene-minimum-target --max-sites 61 --n-jobs 8      # ~20 min with a reuse

python scripts/tokenise_a_dataset.py lemat_bulk_fmax1_stress \
    yamls/tokenisers/lemat_bulk_fmax1_sg_multiplicity.yaml --new-tokenizer
```

Every scalar column a model or a screen will read has to be named here, or it is built and
never reaches a tensor. `--max-sites 61` and the sort-by-Wyckoff-letter default are what
make the result comparable with the existing caches.

With no cache to reuse, `scripts/cache_a_dataset.py lemat_bulk_fmax1_stress --n-jobs 16
--sort-by-letter --scalar-columns ...` does the same from scratch in about six hours, but
has no site cap — filter the 4,581 structures above 61 Wyckoff sites first, or the padded
width goes from 62 to 361.

### Do not use `scripts/pipeline_lemat_20wyckoffs.py`

It is the 2026-05 path that produced the `lemat_bulk_ehull` family. It cuts at
`max_force <= 0.02`, which keeps 95.6% of Alexandria against 35.5% of the ICSD-backed
Materials Project rows, and silently drops every row whose forces are empty. It is kept for
provenance, not for use.

## What is on disk

Produced by `scripts/audit_lemat_variants.py`, which infers each variant's force cut and
checks the formerly-empty rows against the recovered values. Re-run it after building
anything; nothing in a split CSV or a cache records how it was made.

| variant | rows | `max_force` cut | recovered forces | stress labels |
| --- | ---: | --- | --- | :-: |
| `data/lemat_bulk_fmax1_stress` | 5,332,758 | ≤ 1.0 | **yes**, 30,317 rows | yes |
| `cache/lemat_bulk_fmax1_stress` | 5,327,500 | ≤ 1.0 | **yes**, 29,651 rows | yes |
| `data/lemat_bulk_fmax1` | 5,333,114 | ≤ 1.0 | no — 30,673 imputed at 0.04154436 | no |
| `cache/lemat_bulk_fmax1` | 5,327,846 | ≤ 1.0 | no — 29,997 imputed at 0.04154436 | no |
| `cache/lemat_bulk_fmax1_pilot` | 320,000 | ≤ 1.0 | holds none of those rows | no |
| `cache/lemat_bulk_ehull` | 4,207,723 | ≤ 0.02 | holds none of those rows | no |
| `cache/lemat_bulk_ehull_wp20` | 4,207,947 | ≤ 0.02 | holds none of those rows | no |
| `cache/lemat_bulk_ehull_pilot` | 110,000 | ≤ 0.02 | holds none of those rows | no |

Reading the table:

- **`*_stress` is current.** 30,317 of the 30,676 recovered rows survive into the splits:
  356 exceed the 1 eV/Å cut and 3 are the ambiguous ones. The cache loses a further 666 to
  the 61-site cap. Three rows keep `max_force_missing = 1`.
- **`lemat_bulk_fmax1` is superseded**, and is what run `19qbxo6l` trained on. It is not
  wrong so much as built before the recovery existed: 30,676 rows carry the single
  fabricated value 0.04154436, and 356 rows are in it that its own cut would have excluded.
  It is kept because the val and test ids of everything downstream descend from it.
- **The `ehull` family is legacy**, cut at 0.02 by the retired pipeline. Its caches store no
  `max_force` column at all, so the cut above was recovered by joining their ids back
  against `convergence_labels.parquet`. They hold none of the formerly-empty rows, which is
  the NaN drop rather than a decision.
- **`lemat_bulk_fmax1_pilot` is a 320,000-row subsample** of `lemat_bulk_fmax1` that
  contains none of the 30,676 rows. Chance would have put about 1,840 of them in it, so
  they were excluded deliberately; no script in the repository records how, and nothing
  depends on the answer.

Anything trained on a variant without recovered forces has seen 30,676 Materials Project
rows at roughly half their true `max_force`, or has not seen them at all. That matters for
a model conditioned on `max_force`; it does not otherwise change the structures.

## Checking a build

```bash
python scripts/audit_lemat_variants.py
python scripts/audit_lemat_variants.py --variant data/lemat_bulk_fmax1_stress
```

A correct build of the current dataset reports `<= 1.0`, `yes: 30317 rows; flag on 3` and
stress labels present. `build_lemat_bulk_fmax.py` additionally refuses to run if the
recomputed `max_force` disagrees with the structure CSV's on any row that already had one,
and logs the control comparison against the 800 never-missing rows.
