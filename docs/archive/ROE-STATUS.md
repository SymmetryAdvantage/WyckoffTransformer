# Rules of engagement: session status

Summary of Claude Code session `8c17cf3d-b38e-422f-93aa-41cf701769d9`
(2026-09-21 14:30 UTC to 2026-09-22 06:07 UTC, host iapetus). Written 2026-09-23.
The session's work is on `main` in commits `de73401` and `773726d`, which follow
`9145354`.

## 1. Naming and design

- The four inference modes have naval names. Every term that already exists
  keeps its name: `e_hull`, formation energy, convex hull, novelty, uniqueness,
  SUN, MetaSUN, relaxation, gene.

  | # | Mode | Pipeline |
  |---|---|---|
  | 1 | **broadside** | generate → reconstruct (CrySPR) |
  | 2 | **fire-discipline** | generate → novelty/uniqueness screen → reconstruct |
  | 3 | **fire-control** | generate → screen → predicted `e_hull` → reconstruct |
  | 4 | **torpedo-run** | sample chemical system → conditioned generate → predicted `e_hull` → screen → reconstruct |

- The design doc is `docs/rules_of_engagement.md`. It is in `docs/`, not
  `data/rules_of_engagment.md` as requested, because `data/` holds datasets.
- The new package is `wyckoff_transformer.roe`: `cohort.py`, `components.py`,
  `plan.py`, `builtin.py`, `report.py`, `cli.py` and tests. It adds the CLI
  `wyformer-roe`.
  - A mode names slots; a run fills each slot with a component. The built-in
    components are thin adapters over existing code: `screen_genes`,
    `score_genes`, the protocol stages and `system_prior`.
  - A cohort marks genes and never drops them. Filters write verdict columns and
    a `kept` mask, so every rate can be restated per *sampled* gene.
  - Broadside is charged for its duplicates without relaxing them. Its cost is
    exact and its yield is a lower bound.
  - **DiffCSP++ is not implemented.** Asking for it raises an error.
    `--no-reconstruct` writes the gene file for an external run instead.
- Fire-control keeps the **1000 lowest predicted `e_hull`** by default rather
  than applying a zero threshold. A pilot found only 10 of 187 novel genes (5.3%)
  predicted at or below the hull, so a zero threshold would need about 30k draws.
  `--energy-select threshold` is still available.
- Differences between modes are read from `metasun_per_sampled_gene`
  (`e_hull ≤ 0.1`). SUN is reported but not argued from: 1000 reconstructions per
  arm is too few to resolve it.

## 2. Tensor-space novelty and uniqueness screen

The screen lives in `evaluation/gene_hash.py` and is now the default backend.

- The key is a 128-bit hash of the **whole set** of augmentation variants, not a
  min-over-orbit canonical form.
- It was checked against the full reference (5,327,342 rows):

  | | fingerprint set | key table |
  |---|---|---|
  | distinct genes | 4,826,004 | **4,826,004** |
  | memory | ~17 GB | **77 MB** |
  | load, then screen 300 genes | ~2 min | **1.2 s** |

- Tests cover a partition identity over 3000 random genes, invariances, stability
  under `PYTHONHASHSEED`, the table's edge cases, backend agreement, and three
  `needs_cache` reference tests.

## 3. Defect found in the Wyckoff augmentation (fixed in `de73401`)

- **Defect.** `data.py::pyxtal_notation_to_sites` paired the **original**
  site-symmetry symbol with the **relabelled** enumeration index.
- **Where it applies.** In 26 orthorhombic space groups the normaliser relabelling
  changes the oriented symbol (`2..` → `.2.`): 16, 17, 20–25, 35, 42, 44, 47–50,
  59, 65–74.
- **Effect.** `record_to_augmented_fingerprint` both false-merged genes and, much
  more often, missed merges. Examples:
  - sg 68: `1e,1c` and `1e,1d` were merged, although no relabelling relates them.
  - sg 16: `1t` and `1o` were not merged, although an axis permutation relates
    them.
- **What was still correct.** The relabellings do form a group in all 230 space
  groups. The defect is purely in the plumbing.
- **Literature.** This is known crystallography: ITA A1 uses I222 (sg 23) as its
  example. `pymatgen`'s prototype labels, AFLOW and spglib avoid the problem by
  carrying letters rather than symbols. PyXtal does not document the provenance
  of `wyckoff_sets.json`; it matches Bilbao WYCKSETS field for field.
- **The fix.**
  - `data.py::augmented_sites` now emits `site_symmetries_augmented`, aligned with
    `sites_enumeration_augmented` and deduplicated and sorted as pairs.
  - Nine consumers were switched to the pair: the four fingerprints in
    `novelty.py`, `evaluation/core.py` (novelty during training),
    `evaluation/protocol.py`, `gene_hash`, `gene_energy`, `gene_likelihood`,
    `cryspr/template.py`, and `wyckoff_processor.get_feature_from_augmented_series`
    (the training feature path).
  - A record that has the old column but not the new one now raises an error.
- **`site_symmetry_ops` (question c).** It was not independently broken. It is
  keyed on `(sg, site_symmetries)` and is correct once the symbol is relabelled.
  The engineered fields that were actually broken were `harmonic_site_symmetries`
  and `harmonic_cluster`. `multiplicity` was right because it is a normaliser
  invariant.
- **New tests.** They compare against the letter-orbit ground truth rather than
  against the fingerprint itself. They cover equivariance, both hand-checked
  cases, the group property, the list of 26 symbol-changing groups, variant order
  under `PYTHONHASHSEED`, the migration script, and agreement between PyXtal's
  `site_symm` and the package table.
- The full write-up is in `docs/wyckoff_augmentation_audit.md`.

## 4. Tokeniser changes (`de73401`, `773726d`)

- **Obsolete marker.** Tokeniser configs can now carry an `obsolete: <reason>`
  field.
  - `train.py` refuses an obsolete config unless given
    `--allow-obsolete-tokeniser`.
  - Re-tokenising and inference only warn.
  - 30 configs are marked: 21 for the augmentation defect and 9 for the harmonic
    fields. The original yamls were restored and left otherwise unchanged.
- **`yamls/tokenisers/der_tokenizer_v1.yaml`** is new. It combines
  `lemat_bulk_fmax1_sg_multiplicity` and `lemat_bulk_ehull_sg_multiplicity_ssops`,
  and adds two things neither had:
  - `site_symmetries` is augmented alongside `sites_enumeration`;
  - `site_symmetry_ops_id` is augmented-engineered (recomputed per variant).

  `multiplicity` is not augmented. Two outdated restrictions were lifted: the
  3-level-only augmented-engineer path, and `train.py`'s limit of one augmented
  field.
- **Removed:** `harmonic_site_symmetries`, `harmonic_cluster`,
  `engineers/sites_enumeration.json` (the inverse mapping from cluster to
  enumeration), the spherical-harmonic signature code, and
  `generation_modes.HarmonicCluster`. That is 897 KB of package data. The nine
  configs that use these fields can still load engineers from their own run
  directories but **can no longer be retrained**.
- Tests at merge: **1238 passed, 50 skipped, 0 failures**.

## 5. The experiment: stopped with no results

- **Setup.**
  - Backbone: `unconditional_5x_ehull01-20260915-151250`.
  - Energy model: `min_energy_adamw_wsd-20260912-115957`.
  - All three arms share one pool of 10,000 genes, and each gets 1000
    reconstructions.
  - The run configs were saved to disk and the runs log nothing to W&B.
  - Driver: `scripts/platforms/iapetus/run_rules_of_engagement.sh`.
  - Outputs: `/mnt/hdd/kna/wyformer/roe/{pool,broadside,driver.log}`.
- **Progress.** Relaxation ran at about 12.6 trials/min, roughly 3.5 h per arm.
- **Stopped** on 2026-09-21 at 17:36 UTC, with broadside at 634 of 2663 trials.
  The augmentation fix would otherwise have given the three arms different
  fingerprints. Fire-discipline and fire-control never started.
- **Resuming.** The pool and broadside's trials are kept. Broadside's engaged set
  does not depend on the screen, so `--resume` can pick those trials up.

## Open items

1. **The cache has been rebuilt; the derived novelty caches have not.**
   - LeMat-Bulk was re-cached and re-tokenised on 2026-09-22 (the user reported
     this on 2026-09-23).
   - The split files in `cache/lemat_bulk_fmax1_stress/` (`{train,val,test}.parquet`,
     from 21:21 local time) contain `site_symmetries_augmented`.
   - `tensors/der_tokenizer_v1.safetensors` is 13 GB and dates from 22:48.
   - **Rebuilt on 2026-09-23:** `gene_keys.npz` (199 s) and
     `gene_fingerprints.pkl.gz` (975 s), from the corrected Parquet splits.
     - The fix lowers the distinct gene count from 4,826,004 to **4,823,981**
       (−2,023) over the same 5,327,342 rows. The net change is toward more
       merging, which fits the audit's finding that missed merges dominated.
     - The key table and the fingerprint set give the same count, so the two
       backends still agree.
     - The pre-fix files are kept beside them with the suffix
       `.pre-augmentation-fix`.
     - Build log: `cache/lemat_bulk_fmax1_stress/rebuild_derived.log`.
   - **Rebuilt on 2026-09-23:** `gene_ehull_index.pkl.gz`, using
     `scripts/analyse_guidance_sweep.py index`.
     - Its builder first had to be ported from the superseded `data.pkl.gz` to
       the Parquet cache (uncommitted).
     - It holds 4,823,981 fingerprints, the same count as the set above.
     - The pre-fix file is kept with the suffix `.pre-augmentation-fix`.
     - Build log: `rebuild_ehull_index.log`.
   - The superseded `data.pkl.gz` is still present.
2. **Resume the experiment.** It is unblocked now that `gene_fingerprints` and
   `gene_keys` are rebuilt: broadside with `--resume`, then fire-discipline and
   fire-control. Torpedo-run
   has never been run. Training on `der_tokenizer_v1` is no longer blocked.
3. **Decide on retraining.** Every existing checkpoint was trained on the
   corrupted augmentation. 10.0% of model output falls in the 26 affected groups;
   the share in the training corpus has not been measured.
4. **Measure the novelty impact.** It is unknown how many novelty verdicts against
   the 4.8M reference flip under the corrected fingerprint. On an adversarial
   sample, missed merges dominate, which would **overstate** novelty and MetaSUN
   in every stored protocol artifact.
5. **Fix the doc.** The STATUS banner in `docs/rules_of_engagement.md` says
   broadside, fire-discipline and fire-control "have been run". That is false:
   the experiment was stopped and none of the arms finished. The banner needs
   correcting.
6. The branch `fix/wyckoff-augmentation-site-symmetry` was fast-forwarded into
   `main` and can be deleted.
