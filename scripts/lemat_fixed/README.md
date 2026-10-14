# Pipeline

Run in order, each with the work directory as the first argument
(`build_fixed.py WORK OUT`). Paths to the upstream HF snapshot and the MP S3
manifests are hard-coded at the top of the scripts. `mpclient.py` reads `MP_API_KEY`
from a `.env` file.

1. `load_meta.py`: per-row metadata from upstream `compatible_pbe`.
2. `join.py`: joins in the MLIP energies of `LeMat-Bulk-MLIP-Hull-All`.
3. `dl_collections.py`: the MP thermo `GGA_GGA+U` collection. Its unsorted core pull is
   superseded by step 4.
4. `dl_core.py`: MP material docs for the LeMat MP ids.
5. `dl_tasks.py`: task metadata for every GGA/GGA+U static and relaxation.
6. `dl_aliases.py`: LeMat ids MP has merged into other materials, mapped to their
   parent material.
7. `identify.py`: the upstream source task per row, MP's current task, and the
   provenance group.
8. `s3_extract.py`: candidate tasks' full outputs from `s3://materialsproject-parsed`.
9. `review_rows.py`: full upstream rows under review.
10. `decide.py`: geometry matching and the replace/keep rules.
11. `label_geo.py`: StructureMatcher labels for whole-structure replacements.
12. `check_contacts.py`: shortest-bond sanity check on every replacement.
13. `s3_steps.py`: last-ionic-step forces/stress for replacement tasks whose output
    fields are empty. Reads `need_steps.txt`, the new task ids of replaced rows with
    empty forces.
14. `build_fixed.py`: writes the shards and `audit/changes.parquet`. It also fills
    upstream-empty forces/stress from `scripts/recover_mp_forces.py`'s
    `cache/mp_forces_recovery/runs/full/results.parquet`. Removals come only
    from `removal_causes.py`, each with its evidence, plus MP-deprecated materials with
    measured non-standard-POTCAR offsets.

The evidence for the calculation defects comes from two scripts:
- `traj_find.py`: Alexandria relaxation trajectories from `LeMaterial/LeMat-Traj`.
- `siblings.py`: identical-structure duplicates across MP, Alexandria and OQMD.

Alexandria's Bader charges come from its OPTIMADE API, and the MP task documents
from S3.
