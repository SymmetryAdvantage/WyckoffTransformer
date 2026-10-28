# Handoff: SE new-families campaign, after Phase 8 on the mined candidates

Written 2026-10-28 on hera. The session before pushed the Phase 2 candidates ("DB
candidates") through the plan's Phase 8 checks; Phase 8 ran on hera. **On 2026-10-28 the
user decided that no DFT or AIMD inputs are prepared.** Nothing is to be generated. No
task is open (see [Open](#open)).

## Read first

1. `CLAUDE.local.md` → `docs/platforms/hera/agent_brief.md` and `environment.md`; `AGENTS.md`.
2. [`docs/se_new_families_campaign.md`](docs/se_new_families_campaign.md): the status block
   and "Phase 8". It is the record and must stay current.
3. [`docs/se_new_families_campaign_plan.md`](docs/se_new_families_campaign_plan.md), Phases 7–8.
4. [`docs/e_hull_definitions.md`](docs/e_hull_definitions.md) and
   [`docs/energy_fields.md`](docs/energy_fields.md) before touching any LeMat energy.

User rules that apply here (beyond AGENTS.md):
- **PDFs: send them whole to Claude** (the Read tool on the file). Never use local
  pdf→text tools. This is also saved as a memory.
- **Do not connect to ASPIRE2A from any host until the user says so** (also a memory).
- **No DFT or AIMD inputs** for these candidates (the user, 2026-10-28).
- Run code with `.venv/bin/python` from the checkout root; storage goes through
  `wyckoff_transformer.paths`; W&B runs online by default (`symmetry-advantage`).
- Commit or push only when asked. `yamls/` is immutable once results exist. New keys go in
  with a dated comment before the stage that reads them runs.
- When reporting results, give the date, commit and W&B run.

## Status

- **Branch `se-new-families`. Phase 8 is uncommitted on top of `8c0e0a3`.** Nothing has
  been committed or pushed since. The changes:
  - new `src/wyckoff_transformer/transport/electrochem.py`;
  - `scripts/se_new_families_campaign.py` (stages `mine-md-members`, `phase8-entries`,
    `phase8-window`, `phase8-check`, `phase8-prototypes`, `phase8-report`);
  - `src/wyckoff_transformer/tests/test_transport.py` (58 pass);
  - the config's `phase8:` block. It was added before any Phase 8 stage ran. A comment
    of 2026-10-28 marks `r2scan_clusters` and `aimd_members` as no longer read;
  - `docs/se_new_families_campaign.md` and `docs/platforms/hera/environment.md`;
  - this file.
- **W&B run**
  [`se-new-families-hera-20261025`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/se-new-families-hera-20261025):
  - Phase 8 is the artifact `-phase8:v0` plus the tables `phase8_report` and
    `phase8_interfaces`, uploaded 2026-10-27 13:19 UTC with `code_dirty: true`.
  - `-benchmark`, `-gates` and `-mining` stayed at v0.
  - v0 still holds `phase8/dft_bundle/` and `phase8/aimd_package/`. Both came from a
    `phase8-bundle` stage, now removed. They were withdrawn on 2026-10-28 and deleted
    locally, together with `scripts/platforms/aspire2a/vasp/`.
  - `report.csv` was regenerated after the code change and is byte-identical to v0's.
- Nothing is running; both GPUs are idle.
- **Phase 4 (generation) still waits for the authors' decision on G1. Do not generate.**

Run directory: `~/.local/share/wyformer/runs/se-new-families-hera-20261025/`, Phase 8 in
`phase8/`:

| path | what |
|---|---|
| `report.csv` | one row per member of the 7 tiered clusters (34): the plan's Phase 8 fields |
| `windows.csv`, `interfaces.csv`, `joins.csv`, `e_hull.csv` | Richards 2016 windows, eqs. 2/4/5 against the electrode panel, joins; MP2020 and raw |
| `structures.csv`, `prototypes.csv` | ORB vs LeMat structure check; ICSD structure types |
| `md/` | MD on further members (`candidates.csv`, trajectories) |
| `entries.parquet` | 72,778 LeMat-Bulk-fixed rows with MP2020 energies (not uploaded) |

## What Phase 8 found

- **Cluster 12, borate-sulfates (Li4B4Sb2SO16, Li6Sn2B4SO16; tychite-derived):**
  - Theoretical MP entries, and the only new framework in the electrolyte tier.
  - MP2020 windows are 3.44–4.33 V and 2.10–3.66 V; Li and graphite reduce both hard.
  - MD on the Sn member reached only the coating tier.
  - It is a high-voltage material, if anything.
- **Cluster 9, Li2HN:** a known conductor; window 0.24–0.63 V.
- **Cluster 3, Cmcm Li4MX4 (Li4GeO4 type):**
  - MD on further members put Li4TiS4 `mp-756811` (0.038 eV/atom) in the electrolyte
    tier (1.7e-4 S/cm).
  - No Li4TiS4 is ICSD-backed, and no report was found.
  - Window 1.71–2.07 V.
- **Clusters 10, 16, 21, 28:** known types, coating tier, narrow windows. In Li2ZrN2 and
  Li2HfN2, the only members stable against Li metal, Li does not move.

## Open

- **A commit** of the Phase 8 work, if the user wants one.
- **Phase 4**, after the authors decide on G1.

## Caveats to keep in view

- **MD ranks; it does not predict σ.** Each run is one ordered cell, ORB softens the PES
  (Deng 2025), and σ_300K is extrapolated from 500 K with E_a = 0.5 eV. Li2HN and
  Li6Sn2B4SO16 ran on ORB-modified cells (structure check).
- **Windows are thermodynamic limits** of a placed-on-hull composition. They ignore
  kinetic stabilisation and depend only on the composition.
- **MP2020 puts the polysulfides LiS4, LiS5 and Li2S13 and the superoxide-classified LiO8 on
  the hull.** Some oxidation products in the tables are these artefacts; the raw
  windows are the check.
- **G1 is a formal pass** resting on anion-only frameworks. "New" means only no match to
  a fast OBELiX framework.

## Gotchas met in the last two sessions

- **Use LeMat-Bulk-fixed, not upstream.** Upstream `lemat-bulk/lemat_pbe.csv.gz` has stale
  MP energies. `data/lemat-bulk-fixed/` on hera was copied from zeus on 2026-10-27
  (`docs/platforms/hera/environment.md`); `lemat_pbe_ehull.csv.gz`'s sha256 is pinned in
  the config.
- **pymatgen's `InterfacialReactivity` reads `entry.name` as a formula.** Keep PDEntry
  names default and put ids in `attribute`.
- **MP API:**
  - the thermo type is `"r2SCAN"` (`"R2SCAN"` returns nothing);
  - a `chemsys` filter returns exactly that system;
  - MP now returns new-style ids (`mp-aaabrpyi`). Map them to legacy ids by
    StructureMatcher against `materials/core` structures.
- **The OQMD REST API** returned 502s or unfiltered data. LeMat's ICSD-backed rows
  (`phase8-prototypes`) replaced it.
- **MP2020 needs the structure** (sulfate vs sulfide, peroxide vs oxide) and the run type.
  U follows the most electronegative element (`electrochem.mp_run_parameters`).
- `tail` here rejects `tail -3`; use `tail -n 3`.
- `nohup setsid … &`: `$!` is not the Python process group. Find the PGID with `ps -eo
  pid,pgid,cmd`, record it, and stop with `kill -- -PGID`.
- **Two GPU stages at once run out of memory.** `run_pool` retries rows with an `"error"`
  key on the next run of a stage.
- `md_summary` splits a run key on its last `_`, so keys must not contain other `_`.
- CPU-only tests: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q
  src/wyckoff_transformer/tests/test_transport.py`. Lint with `.venv/bin/ruff check`.
- A long-running stage keeps the code it started with. Rerun analysis stages after a fix.
- SSH from hera to zeus or iapetus: use `SSH_AUTH_SOCK=` so a job never depends on the
  login agent.
