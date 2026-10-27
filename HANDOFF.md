# Handoff: SE new-families campaign, Phase 8 on the mined candidates

Written 2026-10-27 on hera, at the end of the session that ran Phases 0–2. The next task,
in the user's words: **"let's start with pushing the DB candidates through the plan's
Phase 8 checks"**. The "DB candidates" are the compounds mined from LeMat in Phase 2 that
reached an MD tier. Nothing is to be generated (see [Status](#status)).

## Read first

1. `CLAUDE.local.md` → `docs/platforms/hera/agent_brief.md` and `environment.md`; `AGENTS.md`.
2. [`docs/se_new_families_campaign_plan.md`](docs/se_new_families_campaign_plan.md), Phases 7–8.
3. [`docs/se_new_families_campaign.md`](docs/se_new_families_campaign.md): the status block,
   "Gate G1", "Phase 2" and "Results". It is the record and must stay current.
4. [`docs/e_hull_definitions.md`](docs/e_hull_definitions.md) and
   [`docs/energy_fields.md`](docs/energy_fields.md) before touching any LeMat energy.

User rules that apply here (beyond AGENTS.md):
- **PDFs: send them whole to Claude** (the Read tool on the file). Never use local
  pdf→text tools. This is also saved as a memory.
- Run code with `.venv/bin/python` from the checkout root; storage goes through
  `wyckoff_transformer.paths`; W&B runs online by default (`symmetry-advantage`).
- Commit or push only when asked. `yamls/` is immutable once results exist. New keys
  went in with a dated comment before the stage that reads them ran.
- When reporting results, give the date, commit and W&B run.

## Status

- **Branch `se-new-families`, base `b31fb2d`.** On 2026-10-27 the work was committed in
  one commit on top of `b31fb2d`, at the user's request, and not pushed:
  `docs/se_new_families_campaign.md`, `scripts/se_new_families_campaign.py`,
  `src/wyckoff_transformer/transport/` (with the new `screens.py`),
  `src/wyckoff_transformer/tests/test_transport.py` (43 pass),
  `yamls/campaigns/se_new_families_hera_20261025.yaml` and this file. The W&B upload ran
  before that commit, on the uncommitted tree.
- **Gates.**
  - G1b passes, 18 of 20.
  - G1c passes, MD AUROC 0.957.
  - G1 passes only formally, for the empty-framework E_2D. On frameworks with cations,
    no descriptor passes.
- **Phase 2** is done: 5,707 survivors, 518 hits, 153 clusters, MD on the 20 best clusters
  with cations.
- **Phase 3** has no target: no gene-level descriptor passed G1.
- **Phase 4 (generation) waits for the authors' decision on G1. Do not generate.**
- Nothing is running; both GPUs were idle at handoff.
- **W&B run**
  [`se-new-families-hera-20261025`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/se-new-families-hera-20261025):
  artifacts `-benchmark`, `-gates` and `-mining`, uploaded 2026-10-27 with
  `code_dirty: true`. Each `upload` run makes new artifact versions, so run it once at the
  end.

Run directory: `~/.local/share/wyformer/runs/se-new-families-hera-20261025/`.

| path | what |
|---|---|
| `mine/screen.parquet` | every Phase 2 flag per compound (gap, roles, coordination, known SE) |
| `mine/hits.parquet`, `mine/clusters.csv` | the 518 hits with their `cluster`; clusters ranked |
| `mine/md/candidates.csv`, `md_analysis.csv`, `trajectories/*.npz` | Phase 2 MD |
| `mine/framework_dimension.parquet` | framework dimension (diagnostic) |
| `bench/relax/…` | ORB-relaxed CIFs; `relaxed_structures(root)` maps `index` → `relaxed_cif` |
| `g1c/references.csv` | the MD scale (Li6PS5Cl 2.2e-4 S/cm, LGPS 1.4e-4, Li3InCl6 4.8e-5) |
| `~/.local/share/wyformer/data/lemat-bulk/lemat_pbe.csv.gz` | LeMat PBE: `immutable_id, full_formula, chemsys, energy_corrected, max_force, cif` (1 GB) |

## The candidates

The 7 clusters that reached a tier. σ is σ_300K − 1 s.e.m. from ORB MD. The ★ member is
the one MD ran on; e_hull is LeMat PBE, in eV/atom.

| rank | tier | σ (S/cm) | members (e_hull) |
|---|---|---|---|
| 12 | electrolyte | 2.6e-4 | ★Li4B4Sb2SO16 mp-768968 (0.100), Li4B4Sb2SO16 agm003282527 (0.099), Li6Sn2B4SO16 mp-769001 (0.065) |
| 9 | electrolyte | 2.3e-4 | ★Li2HN mp-35695 (0.087), Li2HN mp-34465 (0.066), Li2HN agm003173795 (0.064) |
| 3 | coating | 9.6e-5 | ★Li4GeSe4 agm006046064 (0.020), Li4SnSe4 agm006046394 (0.017), Li4TiS4 mp-756811 and agm003222569 (0.038) |
| 16 | coating | 3.4e-5 | ★LiLaSbO agm005202614 (0.062), LiErTeO oqmd-3271894 (0.054), LiLuTeO oqmd-4527247 (0.067) |
| 10 | coating | 3.0e-5 | ★LiYTe2 agm003097245 (0.020) and 17 more, among them Li2ZrN2 mp-3216 (ICSD), Li2HfN2, Li2TiN2, Li2SnN2, Li2CdO2, LiLaTe2, LiErS2 |
| 21 | coating | 1.6e-6 | ★Li3BiS3 agm003214583 (0.049), Li3BiS3 mp-753677 (0.050) |
| 28 | coating | 1.2e-5 | ★Li4TiS4 agm003244046 (0.025) |

Already known from single web searches (not yet a literature check):
- **Li2HN** (lithium imide) is a known Li⁺ conductor, about 2.5–3e-4 S/cm at room
  temperature ([PCCP 2012](https://pubs.rsc.org/en/content/articlelanding/2012/cp/c2cp23636b)),
  and above 1e-3 S/cm after a low-temperature synthesis
  ([2025](https://pubs.rsc.org/en/content/articlehtml/2025/eb/d5eb00058k)). It is a
  blind rediscovery. In the 500 K MD its H also moves.
- **Li4GeSe4 and Li4SnSe4** belong to the selenido-LISICON family, which has been made and
  measured (Kaib et al., Chem. Mater., about 2e-5 S/cm at 20 °C for the best member).
- **Li4B4Sb2SO16:** only the computed MP entry (OSTI 1298577) was found. Its PBE e_hull
  sits at the screen's 0.1 limit.
- **LiYTe2's cluster** holds the ICSD compound Li2ZrN2. The anonymised matcher groups
  nitrides, oxides and tellurides with one framework; the cluster is not one chemistry.
- Only 14 of the 518 hits are ICSD-backed; 431 come from Alexandria.

## Phase 8 for compounds that are already DFT entries

The plan's Phase 8 was written for generated structures. For mined ones:
- **The PBE relaxation exists already.** Each candidate is a relaxed LeMat PBE entry, and
  its e_hull is LeMat's. Re-relaxing with PBE adds little. What is new is r2SCAN, which
  tests whether the e_hull survives (it matters most for Li4B4Sb2SO16), and a check that
  the ORB structure MD ran on still matches the DFT one (StructureMatcher between the
  `relaxed_cif` and the LeMat `cif`).
- **The report fields** (plan, Phase 8): framework and members; e_hull (LeMat PBE in place
  of "free and fixed-symmetry"); descriptors and MD tiers; the window; the join flag and
  the coordination flags (`rarest_environment_count` is in `screen.parquet`).

### Suggested order

1. **ICSD and literature check, per family.** Cheap; do it first, because it can remove
   candidates (Li2HN is already known). Sources:
   - LeMat's ICSD provenance (`icsd_backed`; cache `lemat_bulk_fixed_fmax1_stress_icsd`);
   - MP and OQMD entry pages;
   - web searches, reading papers as whole PDFs.

   Record the outcome per family in the campaign doc.
2. **Windows, interface reactions and the join flag** (hera, CPU; nothing like it exists
   in the repo yet).
   - Method: Richards 2016, a grand-potential phase diagram open in Li over LeMat entries
     in each chemical system (`lemat_pbe.csv.gz`; filter by `chemsys` once and cache
     it). That gives the reduction and oxidation limits against Li/Li⁺. Then pymatgen's
     `InterfacialReactivity` (`pymatgen.analysis.interface_reactions`, installed)
     against Li metal and a cathode.
   - **Decisions for the user:**
     - (a) Which electrodes are "intended". Richards used Li metal and LiCoO2; Co is
       outside the palette, but LeMat has Co entries.
     - (b) **Energy corrections.** `energy_corrected` is raw PBE(+U), with no MP2020
       anion or +U correction (`docs/e_hull_definitions.md`). Richards' windows used
       MP-compatible energies, and oxidation limits for O, S and N compounds are
       sensitive to this.
   - Implement it as a driver stage (e.g. `phase8-window`) in the existing style. Put
     new config keys under a new `phase8:` block with a dated comment.
3. **DFT, only where it is really needed.** The user (2026-10-27): *"If you really
   [need] DFT, we can run VASP on ASPIRE2a."* So the route is decided: VASP on ASPIRE2A,
   via `~/NSCC-VASP-computer` there (MongoDB + FireWorks, MP settings;
   `docs/platforms/aspire2a/experiment_orchestration.md`, `docs/generation.md`, "DFT
   relaxation"). QE on zeus is out: its pslibrary set lacks most of these elements, and
   its energies are not on LeMat's footing.
   - **Justify each calculation first.** Steps 1, 2 and the structure check need no DFT:
     the candidates already have LeMat PBE relaxations and energies. The case for DFT
     is r2SCAN energies for the electrolyte-tier new family:
     - the two Li4B4Sb2SO16 entries (PBE e_hull 0.099–0.100) and Li6Sn2B4SO16 (0.065);
     - the competing phases of Li–B–Sb–S–O and Li–Sn–B–S–O that `mp_20260928_r2scan_gap`
       does not cover (r2SCAN energies for 122,756 MP entries, `R2SCAN_MP`,
       uncorrected; never mix them with PBE).

     Anything beyond that (coating-tier families, gaps) needs a stated reason.
   - **Access:** hera cannot reach ASPIRE2A. There is no SSH host entry, and
     `asp2a-login-nus02` does not resolve. Prepare a self-contained bundle on hera: the
     structures, the list of calculations and the settings. Then ask the user how it
     gets there and who submits. Do not assume an allocation or submit anything without
     asking.
4. **AIMD, by the co-authors.** Do not run it here. Prepare a package for the best ~10
   members: LeMat CIF, ORB-relaxed CIF, the MD supercell matrix (in `md.jsonl`), the ORB MD
   numbers, and the trajectories (W&B `-mining` artifact).

Optional, from Phase 7's scope ("up to 3 structures per candidate family"): `mine-md` ran
only each cluster's best member. Worth adding a members option:
- Li6Sn2B4SO16 mp-769001 has a lower e_hull (0.065) than the Li4B4Sb2SO16 that was run.
- Li2HN has two other orderings in its cluster.
- Li4SnSe4 is the lowest member of rank 3.

One MD run took 22 min–3.4 h with 6 sharing the two GPUs. Phase 7b (doping) applies only
to new frameworks; Li4B4Sb2SO16 is the one plausible case.

The output goes into a new "Phase 8" section of `docs/se_new_families_campaign.md`,
organised by family, plus the new files in `UPLOADS["mining"]` or a new artifact. Upload
once at the end.

## Caveats to keep in view

- **G1 is a formal pass.** It rests on anion-only frameworks. E_2D's line selects
  frameworks of isolated units (22% within the line) and anion-only ones (41%), but only
  1.5% of 3D frameworks. Oxides are nearly out of its reach.
- **MD ranks; it does not predict σ.** Each run is one ordered cell, ORB softens the PES
  (Deng 2025), and σ_300K is extrapolated from 500 K with E_a = 0.5 eV.
- **The framework filter tests only the slowest non-Li species,** so H moving in Li2HN
  passes it.
- **"New" so far means only no match to a fast OBELiX framework.**

## Gotchas met in the last session

- `nohup setsid … &`: `$!` is not the Python process group. Find the PGID with `ps -eo
  pid,pgid,cmd`, record it, and stop with `kill -- -PGID`.
- **Two GPU stages at once run out of memory.** That cost 431 `mine-pes` rows. `run_pool`
  retries rows with an `"error"` key on the next run of a stage.
- `md_summary` splits a run key on its last `_`, so keys must not contain other `_`.
  LeMat ids use `-`.
- CPU-only tests: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q
  src/wyckoff_transformer/tests/test_transport.py`. Lint with `.venv/bin/ruff check`.
- A long-running stage keeps the code it started with. Rerun analysis stages after a fix
  (G1c first printed a stale 0.700 this way).
- SSH from hera to zeus or iapetus: use `SSH_AUTH_SOCK=` so a job never depends on the
  login agent.
