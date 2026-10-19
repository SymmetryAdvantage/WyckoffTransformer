# Building the LeMat-Bulk dataset

The canonical order for turning LeMat-Bulk into a trainable WyFormer dataset. If you have
cloned this repository and want to recreate the data, follow this file rather than
reconstructing a command line from the other documents: several of them describe how a
particular dataset was built at the time, which is not the same as how it should be
built now.

The current dataset is **`lemat_bulk_fixed_fmax1_stress`**, built from
[`kazeevn/LeMat-Bulk-fixed`](https://huggingface.co/datasets/kazeevn/LeMat-Bulk-fixed)
at revision `bf8f0b31` (v4). That archive is upstream `LeMaterial/LeMat-Bulk`
`compatible_pbe` with its stale Materials Project records replaced, 131 broken rows
removed, and the forces and stress upstream left empty filled in
([lemat_stale_mp_energies.md](lemat_stale_mp_energies.md)). How the earlier variants were
built from upstream, `lemat_bulk_fmax1_stress` included, is in
[archive/lemat_bulk_upstream_pipeline.md](archive/lemat_bulk_upstream_pipeline.md).

## Prerequisites

- The `data` and `cache` stores attached; see [data_store.md](data_store.md).
- Internet access to HuggingFace. No MP API key: the Materials Project lookups are
  already in the archive.
- About 12 GB of free space in `data` and 14 GB in `cache`.
- A run takes about 5 h on zeus: 2 h for the archive and hull, 3 h for the cache and
  tokens, and another hour for the derived datasets.

## The pipeline

`D` below is `data/lemat-bulk-fixed` in the data store (`python -m
wyckoff_transformer.paths` prints where that is).

### 1. The archive

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('kazeevn/LeMat-Bulk-fixed', repo_type='dataset',
                  revision='bf8f0b3112a0c522ccd7f90da9748e9240ee80a5',
                  local_dir='$D/raw',
                  allow_patterns=['compatible_pbe/*', 'audit/*', 'README.md'])"
```

- **`raw/compatible_pbe/`:** 17 shards, 5,335,168 rows, 2.8 GB. Check them against the
  sha256 HF lists for the revision; the 2026-10-14 download matched on all 18 files.
- **`raw/audit/changes.parquet`:** one row per replaced, removed, flagged or force-filled
  entry, with the evidence in `cause`.
- **Forces and stress:** every row has per-atom `forces` (eV/Å) and a `stress_tensor`
  (**kBar, VASP sign**: positive means compressed), so no recovery step is needed.

**`max_force` means something different in each source.** For Materials Project it is the
relaxation's own residual; for Alexandria and OQMD it is a re-evaluation at different
settings on the relaxed geometry. Even for MP it is not a convergence verdict: 87–90% of
its relaxations stopped on a positive `EDIFFG` (energy change), never on a force
threshold. Treat it as a provenance proxy, not a convergence label; see
[unconverged_relaxation_energy.md](unconverged_relaxation_energy.md) §8.

The fix makes this more true, not less. For many replaced MP rows the record is now MP's
GGA+U static run on the old GGA-relaxed geometry. That geometry is not a GGA+U minimum,
so the static reports real forces of 1–4 eV/Å. 770 of the 17,998 replaced records are
above 1 eV/Å, against 366 before.

### 2. Structures and energies

```bash
python scripts/prepare_cif.py --input-dir $D/raw/compatible_pbe --output-dir $D/cif_prepared   # 17 min
python scripts/process_lemat.py --input-dir $D/cif_prepared --output-file $D/lemat_pbe.csv.gz  # 21 min
python -m wyckoff_transformer.formula_energy.hull_table --workers 16 \
    --input-file $D/lemat_pbe.csv.gz --output-file $D/lemat_pbe_ehull.csv.gz                   # 1 h 35 min
```

`process_lemat.py` reads the worker count from `NP`. The hull is the reference
`lemat_bulk_fixed_pbe@bf8f0b31`: every row at its raw PBE/PBE+U energy, one pymatgen
phase diagram per chemical system ([e_hull_definitions.md](e_hull_definitions.md)). Every
row gets a hull energy.

Check it on the entries the fix was for (eV/atom above the hull, upstream → fixed):

| entry | upstream | fixed |
| --- | ---: | ---: |
| `mp-1201` Sb₂Te₃ R-3m, the ground state | 1.018 | 0.000 |
| `mp-1079094` Sb₂Te₃ C2/m | 0.000 | 0.462 |
| `mp-1182783` Bi₂Te₃ C2/m | 0.000 | 0.634 |
| `mp-28305` K–Mo–O | 0.000 | 0.389 |
| `mp-725221` CN₄Cl | 6.828 | 0.975 |
| `mp-554674` Ga₃N₃OZn, kept | 0.052 | 0.052 |

### 3. Labels and splits

```bash
python scripts/build_lemat_bulk_fmax.py --name lemat_bulk_fixed_fmax1_stress \
    --structure-csv $D/lemat_pbe.csv.gz --energy-csv $D/lemat_pbe_ehull.csv.gz \
    --labels $D/labels.parquet --convergence-labels $D/convergence_labels.parquet \
    --raw-parquet $D/raw/compatible_pbe --recovered-forces none \
    --split-ids cache/lemat_bulk_fmax1_stress/split_ids.json \
    --exclude-ids data/lemat-bulk/over_61_sites.json                                 # 7 min
```

It writes `data/lemat_bulk_fixed_fmax1_stress/{train,val,test}.csv.gz` with
`energy_above_hull`, `delta_e_polymorph`, `max_force`, `max_force_missing`,
`stress_hydrostatic`, `stress_von_mises`, `stress_missing` and
`formation_energy_per_atom`. The script's docstring defines each label.

- **`--recovered-forces none`:** the archive already carries the recovered forces, and
  the script refuses to apply a recovery to rows that have forces. The convergence table
  should report 5,335,167 rows, all from `archive`; the one row it lacks is upstream's
  single row with no `immutable_id`, which no dataset can carry.
- **`--max-force 1.0`, the default:** keep it. A tighter cut is a provenance filter
  rather than a quality filter; see
  [unconverged_relaxation_energy.md](unconverged_relaxation_energy.md).
- **`--max-stress 500`, the default, is a corruption guard.** It plays the role
  `MAX_ABS_E_FORM` plays for energies, and it is not a convergence cut. Nine OQMD rows
  carry a hydrostatic stress of exactly 1e9 kBar and a von Mises of exactly 3e9, which
  are sentinels rather than measurements, and no `max_force` cut reaches them because
  their force is identically zero. Do not tighten it towards the physically interesting
  range: residual stress is provenance-laden (OQMD's median is +7.5 kBar against MP's
  +0.02), so a cut at 50 kBar would remove 3.45% of OQMD, 0.49% of MP and 0.08% of
  Alexandria.

#### The split is inherited

`--split-ids` takes val and test from `lemat_bulk_fmax1_stress`, whose ids were drawn
uniformly at random. Inheriting is right here because the two datasets cover the same
population: the fixed archive drops rows and adds none. Models on either dataset are
therefore scored on nearly the same held-out structures.

- Val keeps 99,988 of its 100,000 ids, and test 99,986.
- 8 of the 26 lost ids were removed from the archive.
- The other 18 are replaced MP records whose new task fails the 1 eV/Å cut.

Inherit only from a dataset covering the same population. A split inherited from a
differently filtered dataset gives a held-out set unlike the training set; the archived
pipeline has the case that taught this.

#### The exclusion list

`--exclude-ids` drops the 5,258 ids that a `--max-sites 61` cache dropped from
`lemat_bulk_fmax1_stress`, before the split is assigned. Otherwise they would be drawn
into val and test and vanish at cache time. The list is the parent's:

- 74 of its ids had their structure replaced, nearly all by the same crystal in another
  cell setting.
- A replacement that now exceeds the cap is dropped by `--max-sites` at cache time.

### 4. Cache and tokenise

```bash
wyformer-cache-dataset lemat_bulk_fixed_fmax1_stress \
    --observed-gene-minimum-target --max-sites 61 --n-jobs 18                          # 2 h 8 min

python scripts/tokenise_a_dataset.py lemat_bulk_fixed_fmax1_stress \
    yamls/tokenisers/der_tokenizer_v1.yaml --n-jobs 18 \
    --tokenizer-path cache/lemat_bulk_fmax1_stress/tokenisers/der_tokenizer_v1.json    # 42 min
```

Build from a clean commit: the cache records the commit and whether the tree was dirty
([data_store.md](data_store.md#what-built-it)).

- **Labels:** every numeric, boolean and string column of the split CSVs is carried into
  the cache as a per-structure label. `--observed-gene-minimum-target` adds
  `gene_min_formation_energy_per_atom` over all splits.
- **The 61-site cap is applied here, not in the CSVs.** `--max-sites` drops rows whose
  `site_symmetries` is longer than the cap, after symmetrising and before the labels
  are attached, because the site count is only known after symmetry determination. The
  CSVs are therefore a superset of the cache. Leaving `--max-sites` off applies no cap at
  all and takes the padded sequence width from 62 to 361, because every tensor is padded
  to the longest structure.
- **The tokeniser is the parent's saved `der_tokenizer_v1`**, loaded rather than rebuilt.
  The saved JSON comes out byte-identical, so the tensors are token-compatible with every
  LeMat `der_tokenizer_v1` checkpoint. `Field band_gap not found` in the log is expected:
  the config lists a field LeMat does not have.

### 5. Distribute

```bash
scripts/store_sync.sh iapetus  push cache/lemat_bulk_fixed_fmax1_stress data/lemat_bulk_fixed_fmax1_stress        # dry run
scripts/store_sync.sh iapetus  push --go cache/lemat_bulk_fixed_fmax1_stress data/lemat_bulk_fixed_fmax1_stress
scripts/store_sync.sh aspire2a push --go cache/lemat_bulk_fixed_fmax1_stress data/lemat_bulk_fixed_fmax1_stress
```

That is 13.6 GB of cache and 1.5 GB of split CSVs; the derived datasets of step 6 go the
same way (`cache/lemat_bulk_fixed_fmax1_stress_ehull01`,
`cache/lemat_bulk_fixed_fmax1_stress_icsd`, `data/formula_energy_fixed`). Compare sha256 on both ends
afterwards. The remote checkouts also need the commit that adds the dataset manifests,
or they refuse the dataset. `data/lemat-bulk-fixed` itself, the archive and per-row
tables (about 10 GB), is on zeus only.

### 6. Derived datasets

Three datasets are derived from the cache and the hull, each in minutes, without
re-running pyxtal. Each reuses its upstream counterpart's saved tokeniser, and each saved
tokeniser comes out byte-identical.

```bash
# e_hull <= 0.1 eV/atom slice: tensors and records filtered, tokeniser copied     (5 min)
python scripts/slice_dataset_by_ehull.py --source-dataset lemat_bulk_fixed_fmax1_stress \
    --target-dataset lemat_bulk_fixed_fmax1_stress_ehull01 --tokeniser-name der_tokenizer_v1

# icsd_backed column from data/mp_provenance.csv.gz, then tokenise                (50 min)
python scripts/add_icsd_provenance.py lemat_bulk_fixed_fmax1_stress lemat_bulk_fixed_fmax1_stress_icsd
python scripts/tokenise_a_dataset.py lemat_bulk_fixed_fmax1_stress_icsd \
    yamls/tokenisers/der_tokenizer_v1_icsd.yaml --n-jobs 12 \
    --tokenizer-path cache/lemat_bulk_fmax1_stress_icsd/tokenisers/der_tokenizer_v1_icsd.json

# per-formula tables, the MP + OQMD ("shallow") world and its answer key          (16 min)
python -m wyckoff_transformer.formula_energy.dataset --energy-csv $D/lemat_pbe_ehull.csv.gz \
    --force-csv $D/lemat_pbe.csv.gz --out data/formula_energy_fixed/formula_table.parquet
python -m wyckoff_transformer.formula_energy.answer_key --energy-csv $D/lemat_pbe_ehull.csv.gz \
    --force-csv $D/lemat_pbe.csv.gz --deep-table data/formula_energy_fixed/formula_table.parquet \
    --out-dir data/formula_energy_fixed --workers 10
```

- **`lemat_bulk_fixed_fmax1_stress_ehull01`:** 1,602,967 / 31,099 / 31,327 rows. That is
  1.7% more than the upstream slice, because corrected hull vertices pull rows under the
  cut.
- **`lemat_bulk_fixed_fmax1_stress_icsd`:** the parent's rows, 43,407 of them ICSD-backed in
  train.
- **`formula_energy_fixed`:** 2,754,253 formulas, 37 fewer than `formula_energy`, with the
  same formula-level split on the formulas both have.
  - The shallow world is the reference `lemat_bulk_fixed_pbe_mp_oqmd@bf8f0b31`.
  - As upstream, 17,387 of its 706,756 rows get no hull energy.
  - Pass `--force-csv` to both commands. Without it `max_force`, a covariate in both
    tables, is read from the upstream archive.

## Training the models

`yamls/models/lemat_fixed/` has the four models to retrain on these datasets: the e_hull
CFG backbone, the gene energy regressor, the unconditional backbone and the
chemical-system model. Each file's header names the run it reproduces and what changed.

- **The CFG backbone is one run.** The model it replaces was trained as a run plus a
  warm-started continuation; `ehull_adamw_wsd_5x_cfg_long.yaml` is the same peak-lr time
  and anneal as a single schedule.
- **The chemical-system model's horizon is the unconditional backbone's.** The run it
  replaces had not converged.
- **The xl unconditional model trains at half the peak learning rate**
  (`unconditional_5x_ehull01_h3x_xl_lr7p5e-4.yaml`). See below.

All four were trained on aspire2a as 4-GPU chains from the worktree `lemat-fixed-sota`.

| model | config | W&B run | as of 2026-10-19 |
| --- | --- | --- | --- |
| CFG backbone | `ehull_adamw_wsd_5x_cfg_long` | `ehull_adamw_wsd_5x_cfg_long-20261015-021519` | training |
| energy regressor | `min_energy_adamw_wsd_h7x_xl` | `min_energy_adamw_wsd_h7x_xl-20261015-185731` | training (the first run, `-20261015-031005`, collapsed) |
| unconditional | `unconditional_5x_ehull01_h3x_xl_lr7p5e-4` | submitted 2026-10-18, PBS 25866146 | training |
| chemical system | `chemsys_sg_uncond_adanmw_wsd_long` | `chemsys_sg_uncond_adanmw_wsd_long-20261015-044354` | finished, best val 11.24 |

### The xl models can switch their encoder off

The two models without conditioning, the xl unconditional backbone and the xl regressor,
have learnable LayerNorm gains. At peak lr 1.5e-3 the gradient can drive the last encoder
layer's output gain to zero. The encoder then passes nothing on, and the heads fit
marginals: the collapsed regressor predicts the target mean (train MAE 0.6296, exactly
that of a constant), and the unconditional model sits at val ~35.9. The AdaLN models, the
CFG backbone and the chemical-system model, have no learnable gain to lose.

- **It is not the data.** Token tensors of every structure the fix left unchanged are
  bit-identical between the upstream and fixed slices. The unchanged xl unconditional config
  collapsed on the upstream slice too, with current code (`-20261018-034428`). Its
  2026-09-30 run at `5ecb25e` survived, but sat at val 27.4 at epoch 1645, against
  19.1 at half the lr. Whether that run was lucky or the code has changed is not settled.
- **It is not weight decay.** With LayerNorm gains and biases exempt
  (`optimisation.optimiser.exclude_norm_and_bias_from_weight_decay`, kept as an opt-in)
  the run collapsed the same way. Decay only makes a collapse permanent, by driving the
  unused gains of the other layers to zero.
- **Half the lr avoids it.** The unconditional backbone reached val 19.12 at epoch 1645
  at 7.5e-4. The regressor recovered on a plain `--fresh` restart at 1.5e-3, so it keeps
  its config, but is exposed to the same failure.
- **It shows within one validation period**, and a run in it never recovers. Check
  val at the first validation point of any new xl run: about 35.9 (unconditional) or
  0.63 (regressor) means collapsed.

### Protocol evaluation

Run with `scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh`, protocol defaults
(1,000 genes, `orb_conserv_inf`), and the fixed dataset as the novelty reference:

```bash
bash scripts/platforms/aspire2a/protocol_wandb_in_pbs.sh <run-id> \
    --lemat-cif-csv data/lemat_bulk_fixed_fmax1_stress -- --reference-cache cache/lemat_bulk_fixed_fmax1_stress
```

The reference's `gene_fingerprints.pkl.gz` (4,823,711 fingerprints) was built on zeus
in 8 min and synced to all three machines; jobs cannot write the read-only aspire2a store.

| per sampled gene | `chemsys_sg_uncond_adanmw_wsd_long-20261015-044354` | upstream `chemsys_sg_uncond_adanmw_wsd-20260925-030937` |
| --- | ---: | ---: |
| free: MetaSUN | 0.363 | 0.353 |
| free: metastable | 0.472 | 0.456 |
| free: SUN | 0.007 | 0.005 |
| fixed symmetry: MetaSUN | 0.235 | 0.220 |
| gene novelty | 0.801 | 0.806 |
| valid and unique structures | 0.925 | 0.931 |

Every difference is within about one binomial standard error (0.015 at 1,000 genes), so
the 15x longer horizon bought no measurable protocol gain. The two are judged against
different novelty references: the fixed dataset and the upstream one, 512 rows apart.

## What is on disk

| dataset | rows | built from | status |
| --- | ---: | --- | --- |
| `lemat_bulk_fixed_fmax1_stress` | 5,326,830 | LeMat-Bulk-fixed @`bf8f0b31` | current: use for new work |
| `lemat_bulk_fixed_fmax1_stress_ehull01` | 1,665,393 | its e_hull <= 0.1 slice | current |
| `lemat_bulk_fixed_fmax1_stress_icsd` | 5,326,830 | it, plus `icsd_backed` | current |
| `formula_energy_fixed` | 2,754,253 formulas | `data/lemat-bulk-fixed/lemat_pbe_ehull.csv.gz` | current |
| `lemat_bulk_fmax1_stress` | 5,327,342 | upstream LeMat-Bulk, recovered forces | current in its manifest, for the runs trained on it, as are its `_ehull01` and `_icsd` children and `formula_energy`; its hull contains the stale MP energies |

The split CSVs live in `data/<name>/` and the caches in `cache/<name>/`, in the data store
and the cache on zeus, iapetus and aspire2a.

`yamls/datasets/<name>.yaml` records each dataset's status, which training, caching and
tokenising enforce, and the hull reference of each energy field; see
[energy_fields.md](energy_fields.md). The two hulls are different references
(`lemat_bulk_fixed_pbe@bf8f0b31` and `lemat_bulk_pbe`), so energies from one are refused
against the other.

The gene indices (`gene_keys.npz`, `gene_fingerprints.pkl.gz`, `gene_ehull_index.pkl.gz`)
and the defaults of the evaluation protocol and CrySPR's template index are still built
on `lemat_bulk_fmax1_stress`.
Neither the MLIP energies nor the variants older than `lemat_bulk_fmax1_stress` have been
rebuilt on the fixed archive. The older variants are described in
[the archived pipeline](archive/lemat_bulk_upstream_pipeline.md#what-was-on-disk-2026-10-14).

## Checking a build

A correct build reproduces these numbers:

| | |
| --- | --- |
| `lemat_pbe.csv.gz` rows | 5,335,168 |
| rows without a hull energy | 0 |
| convergence sources | `archive`: 5,335,167 |
| rows after the filters | 5,326,830 |
| train / val / test | 5,126,856 / 99,988 / 99,986 |
| cached rows, symmetry failures, over-long rows | 5,326,830, 0, 0 |
| `tokenisers/der_tokenizer_v1.json` | byte-identical to `lemat_bulk_fmax1_stress`'s |

```python
from wyckoff_transformer.dataset_cache import build_info
build_info("cache/lemat_bulk_fixed_fmax1_stress", "train")   # commit, dirty, options
```

On the 99,683 val rows the fix left untouched, `site_symmetries`, `elements`,
`spacegroup_number` and `max_force` equal `lemat_bulk_fmax1_stress`'s.
`energy_above_hull` moved on 8% of those rows, and by more than 10 meV/atom on 2.5%,
because the hull vertices changed.
