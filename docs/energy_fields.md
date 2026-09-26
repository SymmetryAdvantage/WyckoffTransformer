# Energy fields: names, provenance, and what may be combined

A column name does not say which energy it holds. In this repository it never
did: `energy_above_hull` means raw PBE against LeMat-Bulk's own hull in one
dataset and MP2020-corrected against MP's hull in another. LeMat's
`energy_corrected` is not corrected at all. MP-20 spelled the hull distance
`e_above_hull`, and a whole tokeniser, `der_tokenizer_v1_e_above_hull`, existed
only to carry that spelling.

Since 2026-09-26 the meaning is recorded separately from the name, in three
places:

| what | where | module |
|---|---|---|
| the vocabulary: every energy label and what it means | code | `wyckoff_transformer.energy_fields` |
| each dataset's status and fields | `yamls/datasets/<name>.yaml` | `wyckoff_transformer.dataset_manifest` |
| a trained model's fields | `field_provenance.json` beside its weights, and its W&B config | `wyckoff_transformer.field_provenance` |

## 1. The vocabulary

An `EnergyField` is what a column means:

| part | values | notes |
|---|---|---|
| `quantity` | `energy`, `formation_energy`, `energy_above_hull`, `hull_formation_energy`, `delta_e_polymorph` | `hull_formation_energy` is the hull's own formation energy at a composition (formula_energy's `e_hull_at_composition`) |
| `extent` | `total`, `per_atom` | eV or eV/atom |
| `source` | an `EnergySource`, below | how the number was computed |
| `reference` | a registered entry set | required for every quantity except `energy`. Elemental references and hull both come from the set, so one id covers both. |
| `aggregate` | `none`, `gene_min`, `formula_min` | a statistic taken over rows before the value was stored |

An `EnergySource` records how the number was computed:
- **DFT:** `method: dft`, `dft_settings`, `correction`.
- **MLIP:** `method: mlip`, `mlip_model` (the exact checkpoint), and the
  `dft_settings` and `correction` of the data the model was trained on. The
  training-data scale is a property of the checkpoint, so validation checks that
  a field naming a model repeats it correctly.

The closed registries in `energy_fields.py` define every allowed value, and each
entry carries its meaning and evidence:

- **`DFT_SETTINGS`**
  - **`PBE_MP`**: Materials Project GGA/GGA+U settings. MP, MPtrj and LeMat-Bulk
    `compatible_pbe` all use it: MP, Alexandria and OQMD rows alike.
    - Measured 2026-09-26 on 2.15M LeMat-Bulk rows that MP2020 leaves
      uncorrected: MACE-MP-0b3 (trained on MPtrj) sits at a median +0.003
      (MP), −0.010 (Alexandria) and −0.035 eV/atom (OQMD) from LeMat's DFT.
    - The OQMD offset cannot be told apart from the model extrapolating to
      OQMD's prototype structures.
  - **`PBE_OMat24`**: OMat24's settings. These are **not** the MP scale:
    - ORB-v3-omat, UMA-omat and MACE-OMAT-0 all sit +0.05 to +0.07 eV/atom
      above LeMat-Bulk's DFT, on every source.
    - Three different architectures agreeing puts the offset in the training
      data.
- **`CORRECTIONS`**
  - `none`
  - `MaterialsProject2020Compatibility`
- **`MLIP_MODELS`**: the five checkpoints of the published LeMat-Bulk MLIP hulls.
- **`REFERENCES`**
  - **`lemat_bulk_pbe`**: all 5,335,299 LeMat-Bulk rows at their raw energies.
    - The retired `compute_e_hull.py` used the same set. It only left 589,250
      rows unlabelled.
    - So `lemat_bulk_fmax1` and `lemat_bulk_fmax1_stress` share this reference;
      they differ only in which rows they cover.
  - **`lemat_bulk_pbe_mp_oqmd`**: formula_energy's "shallow" subset, without
    Alexandria.
  - **`mp_gga_gga_u_2026.04.13`**: Materials Project's GGA/GGA+U hull at that
    release.
  - **`mp_cdvae_2021`**: MP-20's hull, as CDVAE pulled it.
  - **`lemat_bulk_mlip_hull/<split>@70d505bb`**: one split of the published
    LeMat-Bulk MLIP hulls.
  - **`dataset:<name>`**: a dataset's own rows. `delta_e_polymorph` is measured
    against the lowest polymorph among them.

There is no clipping flag. A distance to a hull that contains the structure is
non-negative by construction. LeMat's single negative row sits at −2.7e-15,
which is float noise. A genuinely negative value only arises for a structure
*outside* the reference set, and the reference id already says which case
applies.

To add a label, add it to the registry with its meaning and its evidence first.
A manifest cannot use one that is not there.

## 2. Canonical names

A column name states the quantity, its extent and any aggregate. It never
names a model, a tokeniser or a data source: that is what provenance is for.
`energy_fields.canonical_id` computes the name from the field, and manifest
validation refuses an energy field named anything else.

| field | name |
|---|---|
| energy, per atom | `energy_per_atom` |
| energy, per cell | `energy` |
| formation energy, per atom | `formation_energy_per_atom` |
| its minimum over the gene | `gene_min_formation_energy_per_atom` |
| its minimum over the formula | `min_formation_energy_per_atom` |
| distance to the hull | `energy_above_hull` |
| the hull's formation energy | `hull_formation_energy_per_atom` |
| energy above the lowest polymorph | `delta_e_polymorph` |

A dataset that holds two definitions of one quantity tells them apart with a
`qualifier` that names the difference in scale, for example
`energy_per_atom_uncorrected` in `mp_2026_gga_gap`.

`wyformer-cache-dataset` renames each source column to its canonical name as it
builds the cache. For example, MP-20's `e_above_hull` becomes
`energy_above_hull`. The old name stays as an alias:
- configs that ask for it still resolve, with a warning;
- tensor caches built before the rename are still found under it
  (`field_provenance.alias_stored_columns`).

The `formula_energy` tables and the `lemat-bulk` source files keep their
historical column names on disk (`e_form_min`, `energy_corrected`). Their
manifests map each one to its canonical field, and consumers resolve through the
manifest rather than trusting the name.

## 3. Dataset manifests

Each `yamls/datasets/<name>.yaml` holds:

```yaml
name: mp_20
status: current            # current | source | obsolete (then `reason:` is required)
description: >- ...
defaults:                  # applied to every energy field unless it overrides them
  source: {method: dft, dft_settings: PBE_MP, correction: MaterialsProject2020Compatibility}
  reference: mp_cdvae_2021
fields:
  energy_above_hull:       # the canonical name
    column: e_above_hull   # its name in the source files, if different
    energy: {quantity: energy_above_hull, extent: per_atom}
  band_gap:
    quantity: band_gap     # not an energy: a free label, no provenance required
```

A few more keys cover the less common cases:
- **`parent`**: a derived dataset inherits its parent's fields. The e_hull ≤ 0.1
  slice of `lemat_bulk_fmax1_stress` does this.
- **`tables`**: a dataset of several files has one table per file, each with its
  `path`. `formula_energy` is one; so is the `lemat-bulk` source.
- **`duplicates`**: a builder's copy of a column under a second name is listed
  here. The cache build checks it equals its original, then drops it.

**Status:**
- `current`: the only datasets to use for new work. As of 2026-09-26 these are
  `lemat_bulk_fmax1_stress`, `lemat_bulk_fmax1_stress_ehull01`, `mp_20`,
  `mp_2026_gga_gap` and `formula_energy`.
- `source`: a raw input that training datasets are built from, and that screens
  compare against (`lemat-bulk`). It is not trained on directly.
- `obsolete`: kept only to analyse runs already trained on it. A dataset with
  **no** manifest counts as obsolete too. Some obsolete manifests still label
  their fields (`lemat_bulk_fmax1`, the `lemat_bulk_ehull` family's
  `energy_above_hull`), so that models trained on them keep a provenance.

**What obsolescence does:**
- **New work refuses it:** training (`scripts/train.py`, and `train_from_config`
  for sweeps), `wyformer-cache-dataset`, `scripts/tokenise_a_dataset.py` and
  `scripts/slice_dataset_by_ehull.py`. Pass `--allow-obsolete-dataset` to go
  ahead anyway; the reason is logged.
- **Resuming does not:** a resumed training run is existing work, so it only warns.
- **Reading prints a warning,** once per process and dataset. That covers every
  cache or tensor load, every model loaded for its dataset, a protocol cohort
  whose generator was trained on one, and `GeneratedDataset`.

**Cache records:** a cache build records the manifest's definitions of the
fields it holds in its `build` options (`fields`, `manifest`). Loading the cache
for a model then refuses it (`check_cache_matches`) if the manifest has since
said something else about those fields. A cache built before the record existed
is trusted, with an info message.

## 4. Models

**At training** (`train_from_config`):
- every `condition_feature` and a `Scalar` `target_name` is resolved through the
  dataset's manifest, and a field the manifest does not declare is refused;
- the result is recorded as `field_provenance.json` in the run directory;
- it is also logged to the W&B run's config under `field_provenance`, to the
  `run_config_<id>` artifact, and as metadata on every `best_model_<id>`
  artifact; `scripts/push_to_hub.py` publishes it too.

It is deliberately **not** in `config.yaml`, which a resumed run must reproduce
exactly.

**At loading** (`WyckoffTrainer.from_config`, `cli.csp.load_trainer`):
- the recorded provenance is used if there is one;
- a run from before 2026-09-26 has its provenance inferred from its dataset's
  current manifest instead, with a warning, and `"recorded": false`;
- when a trained model is loaded together with its dataset, the dataset's fields
  as they are now are checked against the ones it recorded.

Configs keep their keys. A value may be the canonical name, an old alias, or a
bare quantity; a quantity resolves to the row's own value, not a gene or formula
minimum.

## 5. What may be combined

`energy_fields.check_compatible(expected, actual, context, allow=False)` is the
one place that decides:
- **Compatible:** the two fields agree on quantity, extent, all four parts of
  the source, and reference.
- **Aggregate is ignored:** a gene-minimum regressor may be screened against
  per-row energies.
- **Unknown provenance matches nothing:** an unlabelled dataset, or a file no
  manifest names, is refused.

Where it is enforced:

| where | what is compared |
|---|---|
| `wyformer-gene-screen`, ROE's predicted-hull filter | the regressor's target against the hull reference file's formation energy |
| `wyformer-dft-screen` | the reference against the gene regressor's target, the formula ensemble's target and the formula table |
| `formula_energy.prefilter`, `wyformer-screen` | the formula ensemble's target against the reference |
| `gene_energy_residuals` | the regressor against the dataset whose targets the residuals are measured on; the regressor loaded later against the one the residuals were measured on |
| `WyckoffTrainer.from_config` with datasets | a trained model's recorded fields against its dataset now |

The protocol records the definition rather than enforcing one:
- `manifest.json["hull"]["energy_field"]` says what its `e_above_hull` is;
- the generator's `field_provenance` and training dataset are recorded beside it.

A generator conditioned on DFT e_hull is *meant* to be scored against an MLIP
hull; recording both makes the difference visible instead of implicit.

On a mismatch, `IncompatibleEnergyFieldError` lists each part that differs, side
by side, with the meaning of each reference involved. Every CLI above takes
`--allow-incompatible-energy`, which turns the error into a loud warning.
`wyformer-dft-screen` still accepts its old flag name,
`--allow-unverified-energy-scale`.

## 6. Adding a dataset

1. Write `yamls/datasets/<name>.yaml`, with `status: current` and every energy
   column labelled.
2. Run `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest
   src/wyckoff_transformer/tests/test_dataset_manifest.py`. It validates every
   manifest and pins the set of current datasets, so add yours there.
3. Build the cache with `wyformer-cache-dataset <name>`. The columns are renamed
   and their definitions recorded.

To retire a dataset, set `status: obsolete` and say why in `reason:`. Nothing is
deleted.

## 7. Caches rebuilt for the rename (2026-09-26)

The caches were built on branch `energy-fields` from base `c92fa82`, with the
changes not yet committed, so their build records say `dirty: true`. Both were
verified against the caches they replaced; the replaced files are kept in the
store under `staging/energy-fields/previous/`.

- **`cache/mp_20`** (all splits):
  - `e_above_hull` is renamed `energy_above_hull`. Every row and every other
    value is identical to the 2026-09-24 build (`b6725db`).
  - `tensors/der_tokenizer_v1` was re-tokenised with its saved tokeniser. Its
    tensors are identical apart from the added `energy_above_hull`, and the
    tokeniser JSON is byte-identical, so runs already trained on it are
    unaffected.
- **`cache/mp_2026_gga_gap`**:
  - Now in Parquet, alongside the old `data.pkl.gz`.
  - The `mp_dft_*` columns take their canonical names, and the builder's
    duplicate copies under the bare names were verified equal and dropped.
  - Rows, energies and every other value are unchanged. Only container types
    differ (plain ints, sorted tuples), and the newer builder adds
    `site_symmetries_augmented`.
  - The `MaterialsProject2020Compatibility` label was checked on 600 test rows:
    - 91% have `energy_per_atom − energy_per_atom_uncorrected` equal to a
      pymatgen MP2020 recomputation, to float precision;
    - +U oxides and fluorides sit at a median −0.70 eV/atom;
    - the rest are rows where the recomputation, lacking MP's oxidation-state
      detection, applies an anion correction MP did not.

End to end:
- W&B run `02k5uhni` (pilot, 3 epochs) conditioned on `energy_above_hull` with
  `der_tokenizer_v1`. Its `field_provenance` was recorded in the run directory,
  the run config, the `run_config` artifact and every `best_model` version.
- Screening the MP-20 regressor `min_energy_adamw_wsd-20260925-035642` against
  the LeMat hull is refused, on `source.correction` and `reference`.
- The inputs `wyformer-dft-screen` uses today (`zpxeyxm8` and
  `runs/formula_energy/ensemble.pt`) pass, with their provenance inferred.
