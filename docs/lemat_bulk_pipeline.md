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
(**kBar, VASP sign**: positive means compressed). Its 5,335,299 rows are the paper's
"LeMaterial (Compatible, PBE)" exactly ([arXiv:2511.05178](https://arxiv.org/abs/2511.05178),
Table 6). **`max_force` means something different in each source**: for Materials Project it
is the relaxation's own residual, for Alexandria and OQMD a re-evaluation at different
settings on the relaxed geometry. Even for MP it is not a convergence verdict: 87–90% of
its relaxations stopped on a positive `EDIFFG` (energy change), never on a force threshold.
Treat it as a provenance proxy, not a convergence label. The paper's Appendix L has MP and Alexandria the other way round; see
[unconverged_relaxation_energy.md](unconverged_relaxation_energy.md) §8 for the evidence.

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
same calculation keeps both. The LeMat-Bulk paper does not mention them, so anyone
reproducing its Fig. 11 from the released parquet is missing a fifth of Materials Project. This step reads them back for 30,676 of the rows, matching on
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
python scripts/build_lemat_bulk_fmax.py --name lemat_bulk_fmax1_stress \
    --split-ids none --exclude-ids data/lemat-bulk/over_61_sites.json
```

Reads the recovered forces by default (`--recovered-forces none` reproduces the older
behaviour) and writes `data/lemat-bulk/convergence_labels.parquet` on the way — one row per
structure with `max_force`, the stress invariants, and a `convergence_source` of `archive`,
`mp_task_doc` or `missing`. The split CSVs carry `energy_above_hull`, `delta_e_polymorph`,
`max_force`, `max_force_missing`, `stress_hydrostatic`, `stress_von_mises`,
`stress_missing` and `formation_energy_per_atom`.

Keep `--max-force 1.0`: see
[unconverged_relaxation_energy.md](unconverged_relaxation_energy.md) for why a tighter cut
is a provenance filter rather than a quality filter.

#### The split is drawn, not inherited

`--split-ids none` draws val and test uniformly at random. The default is to inherit them
from `cache/lemat_bulk_ehull/split_ids.json`, which is right only while the two datasets
cover the same population — and they do not. Those ids were drawn from a
`max_force <= 0.02` dataset, so inheriting them gave held-out sets in which:

| | train | val | test |
| --- | ---: | ---: | ---: |
| rows above 0.02 eV/Å | 10.8% | **0%** | **0%** |
| Materials Project | 2.66% | **0.79%** | **0.77%** |
| Alexandria | 86.5% | 93.2% | 93.1% |

The model was validated on a cleaner and markedly more Alexandria-heavy population than it
trained on, and on no recovered MP row at all. Under the uniform split every axis agrees to
within 0.1 pp — source mix 86.9 / 2.5 / 10.6, median `max_force` 0.00355, 10.3–10.5% above
0.02, and the recovered rows spread 28,519 / 557 / 572. The drawn ids are written to
`cache/<name>/split_ids.json` so a later variant can inherit *this* split.

Anything trained on the inherited split should be compared against models on the same
split, not against models trained on this dataset.

#### `--max-stress 500` is a corruption guard

It plays the role `MAX_ABS_E_FORM` plays for energies, and it is not a convergence cut.
Nine OQMD rows carry a hydrostatic stress of exactly 1e9 kBar and a von Mises of exactly
3e9 — sentinels, not measurements — and no `max_force` cut reaches them, because their
force is identically zero. The same 1e9 sentinel family is visible on the *force* axis in
Fig. 12 of the LeMat-Bulk paper ([arXiv:2511.05178](https://arxiv.org/abs/2511.05178)),
whose x-axis runs to 10⁹ eV/Å; inside `compatible_pbe` the force tail stops at
1.28 × 10⁴ eV/Å and the sentinels surface only in the stress. At 500 kBar the guard
removes 161 rows (0.003%), all of them
either sentinels or relaxations that ended 50 GPa away from the zero pressure they were
targeting; 92% of the rows it drops already sit more than 1 eV/atom above the hull.

Do not tighten it towards the physically interesting range. Residual stress is
provenance-laden — OQMD's median is +7.5 kBar against Materials Project's +0.02 — so a cut
at, say, 50 kBar would remove 3.45% of OQMD, 0.49% of MP and 0.08% of Alexandria, which is
the `max_force <= 0.02` mistake in a new variable. `--max-stress 0` disables it.

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
never reaches a tensor. The sort-by-Wyckoff-letter default is what makes the result
comparable with the existing caches.

#### Where the 61-Wyckoff-site cap is applied

**At cache construction, not in the split CSVs**, by `--max-sites` in
`cache_a_dataset_reusing.py`, which drops rows whose `site_symmetries` is longer than the
cap. It lives there because the site count is only known after symmetry determination,
which is what that step does — `build_lemat_bulk_fmax.py` never parses a structure.

Three consequences worth knowing:

- **The CSVs are a superset of the cache.** Nothing records the cap on either side, so the
  same split CSVs cached at a different `--max-sites` silently yield a different dataset.
- **`scripts/cache_a_dataset.py` applies no cap at all.** The 4,581 structures above 61
  sites then take the padded sequence width from 62 to 361, roughly sextupling resident
  memory and per-step cost, because every tensor is padded to the longest structure.
- **Uncapped rows would otherwise leak into val and test.** They would be drawn into the
  held-out sets, then vanish during caching, leaving both smaller than requested. The
  build above avoids this by passing `--exclude-ids data/lemat-bulk/over_61_sites.json`,
  the 5,258 ids a `--max-sites 61` cache drops (over-cap rows plus a handful pyxtal cannot
  handle), so val and test come out at exactly 100,000. That list is the set difference
  between the split CSVs and an existing cache; with no cache to hand, build once, derive
  it, and rebuild the splits.

The cap is 61 because that is what the first LeMat cache used, recoverable only because
that cache held exactly 224 fewer rows than the CSV it came from and exactly 224 of its
rows exceeded 61 sites.

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
| `data/lemat_bulk_fmax1_stress` | 5,327,342 | ≤ 1.0 | **yes**, 29,648 rows | yes |
| `cache/lemat_bulk_fmax1_stress` | 5,327,342 | ≤ 1.0 | **yes**, 29,648 rows | yes |
| `data/lemat_bulk_fmax1` | 5,333,114 | ≤ 1.0 | no — 30,673 imputed at 0.04154436 | no |
| `cache/lemat_bulk_fmax1` | 5,327,846 | ≤ 1.0 | no — 29,997 imputed at 0.04154436 | no |
| `cache/lemat_bulk_fmax1_pilot` | 320,000 | ≤ 1.0 | holds none of those rows | no |
| `cache/lemat_bulk_ehull` | 4,207,723 | ≤ 0.02 | holds none of those rows | no |
| `cache/lemat_bulk_ehull_wp20` | 4,207,947 | ≤ 0.02 | holds none of those rows | no |
| `cache/lemat_bulk_ehull_pilot` | 110,000 | ≤ 0.02 | holds none of those rows | no |

Reading the table:

- **`*_stress` is current**, and data and cache hold the same 5,327,342 rows: the 5,258 ids
  over the 61-site cap are excluded before the split (`--exclude-ids`), so `--max-sites 61`
  now drops nothing at cache time and val/test are exactly 100,000 each. 29,648 of the
  30,676 recovered rows survive the filters; three rows keep `max_force_missing = 1`.
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

A correct build of the current dataset reports `<= 1.0`, `yes: 29648 rows; flag on 3` and
stress labels present. `build_lemat_bulk_fmax.py` additionally refuses to run if the
recomputed `max_force` disagrees with the structure CSV's on any row that already had one,
and logs the control comparison against the 800 never-missing rows.
