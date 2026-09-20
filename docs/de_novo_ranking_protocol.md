# The de novo ranking protocol

A development-time evaluation for comparing WyFormer variants by MetaSUN, at
roughly 80% of the cost of a faithful LeMat-GenBench run — 2.4 trials per gene
against 3, over the same four relaxation stages, on one MLIP rather than three.
Implemented as `wyformer-protocol` (`wyckoff_transformer.cli.protocol`) for a
gene file on disk, and `wyformer-protocol-wandb`
(`wyckoff_transformer.cli.protocol_wandb`) to run it end to end against a W&B
run and log the results back.

This is a *ranking* instrument, not a leaderboard predictor. It is deliberately
biased in ways that cancel between arms; the final submission still goes through
the full protocol with the three-model ensemble.

The reasoning behind every default — the power analysis, the stage design, the
trial schedule, the potential and hull choices — is in [the rationale
notes](archive/de_novo_ranking_protocol_rationale.md).

## How many genes

**Generate 1000 genes per arm.** That is the MetaSUN cohort size the power
analysis calls for (945, rounded up), and it is also `wyformer-generate`'s
default `--firm-n-samples`. Larger readouts need more — ~1,900 to develop
against `e_hull ≤ 0.05`, and ~10,000 for SUN (`e_hull ≤ 0`), which is far out of
reach at this scale — which is why development is done against the 0.1 eV/atom
threshold and SUN is only reported. See [the rationale
notes](archive/de_novo_ranking_protocol_rationale.md#develop-against-the-01-evatom-threshold-report-sun).

## Quick start

Against a gene file:

```bash
uv run wyformer-protocol generated/<run>/wyckoff_genes.json.gz \
    --output-dir generated/<run>/protocol \
    --devices cuda:0,cuda:1
```

CPU instead of GPU: `--cores 20`. The two are mutually exclusive, and they
apply to the relaxation only -- the PyXtal draws are always made on the CPU,
`--pyxtal-cores` of them at a time.

Trials follow each gene's positional degrees of freedom by default
(`--n-trials "0:1,2:2,*:3"`); `--n-trials 1` forces a constant budget,
`--no-rattle` drops the rattle stage and `--no-release-symmetry` the
unconstrained stage before it. All three are recorded in `manifest.json`.

## The four stages

Each stage needs different hardware, so each one is separately runnable and
none of them waits on another's resource:

| stage | what it needs | cost | output |
|---|---|---|---|
| `screen` | one core, 4.8M fingerprints in RAM (~17 GB) | ~2 min cached, ~11 min cold | `screen.json` |
| `generate` | every CPU core, no potential | seconds per draw | `pyxtal.extxyz`, `pyxtal.csv` |
| `relax` | the MLIP, on GPU | ~2-8 s per trial per K20c-class GPU | `relaxations.csv`, `structures.csv`, `structures_fixed_symmetry.csv`, `cifs/`, `cifs_fixed_symmetry/` |
| `score` | the hull parquet and LeMat-Bulk geometry in RAM | 8–20 min on 2026-09-15 (zeus, shared load): mostly the relaxed-structure symmetry detection, then ~3 min for the novelty reference | `funnel.json`, updated `structures*.csv` |

Two further stages are optional, and neither is part of `--stage all`. Both are
alternative *sources of starting structures* rather than steps of the cascade,
and neither changes a default.

`template` adds one [template-matched start](cryspr_template_starts.md) per gene
— a training structure's lattice and coordinates instead of a PyXtal draw — as
an extra trial, to be relaxed and scored alongside the random ones.

`prescreen` narrows instead of adding: with `generate --trial-multiplier 10` it
relaxes ten times the usual draws on a cheap potential under fixed symmetry,
drops the ones that landed on the same structure, and hands `relax --relax-from
prescreen` the schedule's *usual* number of lowest-energy survivors — so the
expensive relaxation count is unchanged and only the choice of start improves.
That, and the two-stage `--prerelax-mlip nep89`, are the two
[NEP89 variants](de_novo_ranking_protocol_nep89_variants.md).

```bash
wyformer-protocol genes.json.gz --output-dir run/ --stage screen
wyformer-protocol genes.json.gz --output-dir run/ --stage generate --pyxtal-cores 12
wyformer-protocol genes.json.gz --output-dir run/ --stage relax --devices cuda:0,cuda:1
wyformer-protocol genes.json.gz --output-dir run/ --stage score
```

**PyXtal is generated before the relaxation, not inside it.** A draw takes
milliseconds to seconds while a trial's relaxation takes seconds to minutes, so
interleaving them costs little on CPU -- but it idles a GPU, and PyXtal
rejection-samples, so a gene it cannot satisfy does not fail: it spins.
`--pyxtal-timeout` (300 s) bounds one draw and `--relax-timeout` (300 s) one
trial -- 1800 s before commit 86935ef, which is what every protocol artifact up
to 2026-09-12 was relaxed with, so pass `--relax-timeout 1800` to extend one of
those runs -- and an abandoned one is recorded as such rather than silently missing.

**The unit of work is a trial, not a gene.** Trial budgets differ by a factor
of three, so a pool keyed by gene leaves workers idle at the end; both stages
therefore queue `(gene, trial)` pairs, and both append a row per pair as it
finishes. `--resume` (on by default) then restarts at the trial granularity,
and `--no-resume` starts over.

**A broken worker is not a failed trial.** A CUDA error such as `unspecified
launch failure` poisons the worker's context, and every later trial on it
then fails in milliseconds. One card on iapetus did this to 750 of the 2324
trials of `ehull-ssops-20260904-235534`, whose MSUN then read a third lower
than a comparable model's while its MSUN per relaxed structure was the same.
The pool stages (generate, prescreen, basinhop, relax) therefore run under
`wyckoff_transformer.cli.worker_pool`, which:

- stops giving work to a worker that reported a GPU error, lets the other
  workers finish what they are running, and retries the trial on a fresh pool;
- retries the trials of a worker that crashed or hung past its timeout
  (twice `--relax-timeout`, plus 600 s: a trial can wait for another to finish
  before a worker takes it). A crash that could have come from any of the
  trials running at the time charges none of them; they are then run one at a
  time, so that a trial which keeps crashing is the only one charged;
- retires a device that faults in three consecutive pool rounds;
- stops a pool without waiting on a worker that never finished initialising,
  which would otherwise block the stage after its last trial;
- writes a trial it still could not answer (three attempts, or no device left)
  with an error that `--resume` recognises and re-runs.

Each stage records what it had to do in `manifest.json` (`relax_worker_faults`,
`relax_pool_breaks`, `relax_hung_workers`, `relax_trials_retried`,
`relax_trials_unanswered`, `relax_retired_devices`, with the same fields for
`pyxtal_`, `prescreen_` and `basinhop_`). A stage that leaves unanswered trials
raises once its outputs are written, and `score` refuses a run with any, so
that such a run is not scored and uploaded as the model's numbers. The fix is
`--resume`; `--allow-incomplete` scores what there is.

**A resumed row must come from the inputs it is resumed with.** Stage logs are
keyed by `(gene index, trial)` and nothing else. `protocol_ehull5x-20260904-213346`
v1 and v2 were produced by re-running `wyformer-protocol-wandb` without
`--no-resume`: it sampled a fresh gene file over the old one and then resumed the
old draws and relaxations as the new genes' trials. 2392 of the 2804 successful
relaxations, and 789 of the 998 structures scored, belong to a gene other than
the one they were scored as (audit of 2026-09-15). Every output therefore records
its lineage under `lineage` in `manifest.json`: an `id` of its own and the
`parent` it was built from. That parent is the gene cohort's digest for
`screen.json` and `pyxtal.csv`, and the id of the source log for `prescreen.csv`,
`basinhop.csv` and `relaxations.csv`. An output started afresh gets a new id.
With that record:

- a stage refuses to resume a log whose parent is not what it is building from
  now (`StaleOutputError`). A `generate --no-resume` therefore makes a later
  `relax --resume` refuse: PyXtal is not seeded, so the re-made draws keep their
  keys but not their structures;
- a log written before lineage was recorded is checked by content: every row's
  composition must be its gene's (or its draw's), and the log is adopted if so;
- `score` refuses outputs whose recorded lineages do not chain, and the W&B
  wrapper uploads nothing for it;
- `wyformer-protocol-wandb` refuses to sample a gene file at all into a
  directory that holds stage logs, unless given `--no-resume` (start over) or
  `--skip-generate` (resume the cohort already there). The later check would
  come too late: by then the gene file those logs belong to is overwritten.

`--limit 12` gives a smoke test on the first twelve genes. Outputs:

| file | contents |
|---|---|
| `screen.json` | validity, uniqueness with counts, gene novelty |
| `pyxtal.extxyz` | every generated draw, tagged with its gene and trial |
| `pyxtal.csv` | per trial: PyXtal status (`ok`/`failed`/`timeout`), formula, DoF, seconds |
| `relaxations.csv` | per trial: status, energy, device, seconds, kept CIFs (both free and fixed symmetry), and error if failed |
| `structures.csv` | per gene: lowest-energy trial chosen after rattling and symmetry release (free), plus validity, uniqueness, novelty, `e_above_hull`, `dof_positional`, `n_trials`, `gene_novel`, `novel_by_sampled_gene`, `relaxed_fingerprint_resolved`, `relaxed_fingerprint_changed` |
| `structures_fixed_symmetry.csv` | per gene: lowest-energy trial relaxed under fixed symmetry (pre-rattling), scored identically to `structures.csv` |
| `funnel.json` | hierarchical report containing `gene`, `fixed_symmetry`, and `free` sections, with cascade rates per *sampled* gene |
| `manifest.json` | MLIP, checkpoint, trial schedule, rattle, devices, timeouts, hull provenance, novelty reference (`reference_cache`, `novelty_reference`; the screen's under `lineage`) |
| `cifs/`, `cifs_fixed_symmetry/` | kept CIFs: `cifs/` holds free structures post-rattling, `cifs_fixed_symmetry/` holds fixed-symmetry structures pre-rattling |
| `cryspr/` | per-trial relaxation logs and `rattle.json` |

A gene with no structure carries the reason in `structures.csv`'s and
`structures_fixed_symmetry.csv`'s `error` column, and the trial that produced
it is in `relaxations.csv`. This matters: a cohort whose potential fails to load
and one whose genes PyXtal cannot draw both read as `has_structure = 0` in the
funnel and nowhere else.

## Evaluating a W&B run

`wyformer-protocol-wandb` takes a run id, generates a fresh gene cohort from its
checkpoint, runs the full cascade, and writes the funnel metrics into the run's
summary and every protocol output into one versioned artifact.

```bash
uv run wyformer-protocol-wandb <run-id> \
    --output-dir generated/<run-id>/protocol \
    --condition energy_above_hull=0 \
    --devices cuda:0,cuda:1 --workers-per-device 2
```

Drop `--condition` for an unconditional run.

- **Genes are generated, not read.** 1000 by default (`--n-genes`), sampled from
  the run's saved space-group distribution exactly as `wyformer-generate` does.
  The gene file is written to the output directory and included in the artifact,
  so the cohort a run was scored on is recoverable.
- **Conditioning targets default to 0 for stability channels.** Datasets are
  not loaded here, so conditioning cannot be sampled from the training
  distribution. Features `energy_above_hull`, `delta_e_polymorph`, and
  `max_force` default to 0 if not specified via CLI. Other features must be
  passed with `--condition NAME=VALUE` (or `--condition-value` for a
  single-channel model). An unconditional run takes neither.
- **`--temperature` re-draws the cohort, nothing else.** It rescales the logits
  of every generated cascade field; the start token still comes from the run's
  saved space-group distribution, so the space-group marginal is held fixed.
  The value, and how much of the raw draw was formally valid at it, go into
  `manifest.json` as `sampling_temperature` / `formal_gene_validity`. What a
  sweep over it costs and buys is in [the temperature
  sweep](temperature_sweep.md).
- **The run's model files must be reachable.** `runs/<run-id>/` is used if it
  already holds `best_model_params.pt`, `wyckoff_processor.json` and
  `spacegroup_distribution.json`; otherwise they are downloaded from the run.
- **What lands on the run.** Summary metrics are written into `run.summary` with a hierarchical `protocol/` layout:
  - `protocol/gene/`: Gene-level metrics from the initial screen: `sampled`, `valid_gene`, `unique_gene`, `novel_gene`, `validity_rate`, `uniqueness_rate`, `novelty_rate`, and `vun_per_sampled_gene`.
  - `protocol/fixed_symmetry/`: Structure-based metrics evaluated on the lowest-energy structures relaxed under fixed symmetry (pre-rattling): `structure`, `valid_structure`, `unique_structure`, `novel_structure`, `metastable`, `stable`, `metastable_among_novel`, `stable_among_novel`, `sun_per_sampled_gene`, `metasun_per_sampled_gene`, etc.
  - `protocol/free/`: Structure-based metrics evaluated on the lowest-energy structures chosen after symmetry release and rattling: `structure`, `valid_structure`, `unique_structure`, `novel_structure`, `metastable`, `stable`, `metastable_among_novel`, `stable_among_novel`, `sun_per_sampled_gene`, `metasun_per_sampled_gene`, etc.

  `screen.json`, `pyxtal.extxyz`, `pyxtal.csv`, `relaxations.csv`, `structures.csv`, `structures_fixed_symmetry.csv`, `funnel.json`, `manifest.json`, `cifs/`, and `cifs_fixed_symmetry/` go into an artifact named `protocol_<run-id>` of type `protocol_eval`. `--no-upload` runs everything and skips only the write-back; `--wandb-entity` / `--wandb-project` override where the run is looked up.

The same hardware, trial-schedule, MLIP and reference flags as `wyformer-protocol`
are accepted and passed straight through.

- **Re-scoring without re-relaxing.** `--from-artifact --stages score` downloads
  the run's existing `protocol_<run-id>` artifact into `--output-dir` and runs
  only the score stage on it — no cohort is generated, nothing is relaxed. The
  refreshed `funnel.json`, `structures.csv`, and `structures_fixed_symmetry.csv` go
  back as a new artifact version and `run.summary` is overwritten. Use it after a change to how
  novelty or the hull is judged. `--from-artifact v2` pins a version instead of
  taking the latest. After a change of *reference* it is `--stages screen,score`:
  gene novelty is decided in the screen, and `score` refuses a `screen.json`
  judged against a reference other than its own `--reference-cache` (see
  [The novelty reference](#the-novelty-reference)). Re-screening is safe on a
  relaxed run — validity, uniqueness and the representatives depend on the genes
  alone. A re-score that fails uploads nothing.

## The cascade

```
sampled → valid gene → unique gene (keep counts)
        → PyXtal + 1–3 trials × 4-stage CrySPR (2 symmetric, free, rattle)
        ┬─→ fixed symmetry (pre-rattling) ─→ valid structure → unique structure ─┬─→ metastable / stable
        │                                                                        └─→ novel structure → MetaSUN / SUN
        └─→ free (post-rattling)          ─→ valid structure → unique structure ─┬─→ metastable / stable
                                                                                 └─→ novel structure → MetaSUN / SUN
```

**Structure metrics are computed separately along two tracks:**
1. **Fixed symmetry (pre-rattling)**: Structures relaxed through the two symmetric stages (`0_fix_cell` and `2_sym_cell+pos`), keeping the sampled space group and Wyckoff orbit constraints intact. Evaluated and scored into `structures_fixed_symmetry.csv`, `cifs_fixed_symmetry/`, and `protocol/fixed_symmetry/`.
2. **Free (post-rattling)**: Structures chosen after symmetry release (`2b_free_cell+pos`) and rattling (`3_rattle`), which can break symmetry and escape symmetric stationary points. Evaluated and scored into `structures.csv`, `cifs/`, and `protocol/free/`.

Both tracks undergo the same structure scoring pipeline:
- CIF reading and structural validity (pymatgen check).
- Structure deduplication via `StructureMatcher` (`unique_structure`).
- Novelty evaluation against the joint LeMat-Bulk candidate reference (`novel_structure`).
- Hull energy evaluation with the MLIP hull (`metastable`: $e_{\text{hull}} \le 0.1$ eV/atom, `stable`: $e_{\text{hull}} \le 0.0$ eV/atom).
- Final success rates: `sun_per_sampled_gene` and `metasun_per_sampled_gene`.

**`metastable` / `stable` are measured before the novelty filter**, over every
unique structure with a hull energy; `metastable_among_novel` / `stable_among_novel`
and the `metasun` / `sun` rates add novelty back. So `metastable ≥
metastable_among_novel`, and MetaSUN is the novel-only slice of the metastable
count.

**Uniqueness deduplicates but keeps counts.** Every rate stays per sampled gene;
a duplicate belongs once in the numerator and once per sample in the
denominator. Reporting over the deduplicated set instead would make uniqueness
≈1.0 by construction.

**Every unique gene is relaxed**, gene-known ones included — novelty is decided
by `StructureMatcher` on the relaxed structure, not by the fingerprint alone, so
a gene already in LeMat-Bulk can still relax into a novel structure, and its
`e_above_hull` is needed to keep the energy distribution unbiased. Full argument
in [the rationale
notes](archive/de_novo_ranking_protocol_rationale.md#every-unique-gene-is-relaxed-gene-known-ones-included).

### Novelty and uniqueness are two-stage

Both use `NoveltyFilter` and `filter_by_unique_structure` from
`evaluation/novelty.py`: the augmented Wyckoff fingerprint first, then
`StructureMatcher` on whatever shares it.

The fingerprint alone is not a verdict. Two structures with the same space
group and the same elements on the same Wyckoff orbits differ in their free
coordinates and lattice parameters, so a gene that occurs in LeMat-Bulk can
still relax into a structure that is not in it. The screen therefore produces
*candidates* for the matcher rather than a decision.

**Novelty is judged against two fingerprints, not one.** The sampled gene's,
and the *relaxed* structure's own — recomputed by PyXtal symmetry detection in
the score stage. Relaxation, the rattle stage especially, can move a structure
off the orbit set PyXtal placed it on, so a gene PyXtal drew onto a
LeMat-Bulk-known fingerprint can relax into something genuinely new, and a gene
drawn onto an unknown one can relax onto a known structure. A structure is
`novel_structure` iff no LeMat-Bulk entry sharing *either* fingerprint matches
it; MetaSUN and SUN use this. `funnel.json` reports the two crossings —
`gene_known_became_novel` and `gene_novel_became_known` — against the sampled
gene's novelty. When symmetry detection fails on a relaxed structure
(`relaxed_fingerprint_resolved = false`), only the sampled fingerprint is used
for that gene.

That is what makes the reference affordable. LeMat-Bulk has 5.3M entries and
the matcher needs a `Structure` per candidate, which is far too much to hold;
but only entries whose fingerprint collides with a generated one can ever reach
it. On `upi73i4k`'s 2500 genes, 627 fingerprints collide, over **795** reference
structures — a median of 1 candidate each and never more than 6 (measured
against `lemat_bulk_ehull`). So the reference is built per run: one streaming
pass over the Wyckoff cache for the colliding `immutable_id`s, then one chunked
pass over `lemat_pbe.csv.gz` for their geometry.

**A colliding entry without geometry is refused, not dropped.** An entry missing
from `lemat_pbe.csv.gz`, or whose CIF does not parse, cannot be matched against,
and a structure whose only candidates were such entries would be scored novel.
`build_novelty_reference` therefore raises `UnresolvedReferenceError` rather than
scoring around the hole. No such entry exists today: every one of the 5,327,342
`immutable_id`s of `lemat_bulk_fmax1_stress` (and of the 4,207,723 of
`lemat_bulk_ehull`) is in the 5,335,299-row export (checked 2026-09-15).

### The novelty reference

Novelty is judged against **`cache/lemat_bulk_fmax1_stress`, all three splits**
— the current LeMat-Bulk variant ([lemat_bulk_pipeline.md](lemat_bulk_pipeline.md)).
Both halves of it are recorded in `manifest.json`: the screen's reference under
`lineage` → `screen.json` → `reference`, with the size of its fingerprint set;
the score stage's as `reference_cache`, `reference_splits`, `lemat_cif_csv` and
`novelty_reference` (with how many fingerprints collided, over how many
entries). `score` refuses (`StaleOutputError`) a `screen.json` whose recorded
reference is not its own `--reference-cache` and `--reference-splits`, or that
records none, so that gene novelty and structure novelty in one funnel always
come from the same reference. The fingerprint set is cached beside the
reference it was computed from (`--reference-fingerprint-cache` defaults to
`gene_fingerprints.pkl.gz` in the reference's directory, with the splits in the
name when they are not all three); before 2026-09-15 it was one fixed path, so
passing another `--reference-cache` alone silently reused `lemat_bulk_ehull`'s
fingerprints.

**Protocol artifacts scored before 2026-09-15 used `lemat_bulk_ehull`**
(changed on top of commit `600a2ab`). That variant lacks 1.12M of the current
variant's rows — everything above `max_force` 0.02 eV/Å, the Materials Project
rows with empty forces, Yb and actinide chemistry — so a generated structure
matching one of them counted as novel. Gene novelty, `novel_structure`,
MetaSUN, SUN and the novelty crossings of such an artifact are **not comparable**
with a run scored against `lemat_bulk_fmax1_stress` until it is re-scored
(`--from-artifact --stages screen,score`); validity, uniqueness, `metastable`
and `stable` do not depend on the reference. Re-scored on 2026-09-15 with
`--from-artifact --stages screen,score`, `lemat_bulk_ehull` → `lemat_bulk_fmax1_stress`
(free readout unless marked; every other funnel entry unchanged):

| run | artifact | gene novelty | novel structure | MetaSUN | SUN | MetaSUN, fixed symmetry |
|---|---|---|---|---|---|---|
| `e9ywwsie` | v4 → **v5** | 0.672 → 0.663 | 0.665 → 0.653 | 0.278 → 0.268 | 0.006 → 0.005 | 0.161 → 0.153 |
| `ehull-ssops-20260904-235534` | v2 → **v3** | 0.643 → 0.631 | 0.622 → 0.612 | 0.289 → 0.281 | 0.006 → 0.006 | 0.197 → 0.190 |
| `ehull5x-20260904-213346` | v3 → **v4** | 0.580 → 0.557 | 0.594 → 0.579 | 0.268 → 0.255 | 0.014 → 0.012 | 0.185 → 0.172 |
| `19qbxo6l` | v3 → **v4** | 0.673 → 0.665 | 0.588 → 0.580 | 0.193 → 0.188 | 0.007 → 0.007 | 0.103 → 0.099 |
| `e_all_adamw_wsd-20260909-001225` | v1 → **v2** | 0.698 → 0.690 | 0.608 → 0.603 | 0.210 → 0.206 | 0.003 → 0.002 | 0.107 → 0.103 |
| `relational_e_all_adamw_wsd-20260909-234259` | v0 → **v1** | 0.725 → 0.686 | 0.672 → 0.636 | 0.197 → 0.169 | 0.011 → 0.006 | 0.110 → 0.085 |
| `upi73i4k`\* | v3 → **v4** | 0.670 → 0.660 | 0.638 → 0.633 | 0.256 → 0.250 | 0.005 → 0.005 | 0.161 → 0.154 |

\* Not a pure re-score; see below.

The relational run loses the most: 2.8 points of MetaSUN and half its SUN. Its
generated structures are disproportionately ones the current variant has and
`lemat_bulk_ehull` did not, so its MetaSUN lead over `e_all_adamw_wsd` went
from −0.013 to −0.037. In every run the re-screen kept validity, uniqueness and
the counts exactly, and no gene went from known to novel.

**`upi73i4k` also had its gene 860 re-relaxed.** `score` refused v3
(`IncompleteStageError`): all three trials of gene 860 (K24Cl36H90O132, 282
atoms) had failed with a CUDA out-of-memory error on a 2 GiB card. v3 was scored
on 2026-09-12, before that check existed, and counted the gene as having no
structure. On 2026-09-16 those three trials were re-run on zeus with `relax
--resume` (one worker on each RTX 6000 Ada, `--relax-timeout 1800`, the same
draws from `pyxtal.extxyz`), then `screen` and `score` against
`lemat_bulk_fmax1_stress`. The other 2365 trials were kept. So v4 differs from v3
by that one gene as well as by the reference: `structure` 997 → 998,
`valid_structure` 904 → 905. The gene relaxed to a valid, novel structure at 0.101
eV/atom above the hull, just outside `metastable`, so `metastable` and `stable`
are unchanged.

## The settings, in brief

The reasoning for each is in [the rationale
notes](archive/de_novo_ranking_protocol_rationale.md).

| setting | default | one-line reason |
|---|---|---|
| cohort size | 1000 genes/arm | MetaSUN power at α=0.05, 80%, +20% relative |
| readout | develop against `e_hull ≤ 0.1`, report SUN | SUN is 10× costlier to resolve for the same shift |
| relaxation stages | fix-cell → sym cell+pos → free → rattle | only the rattle can leave a symmetric stationary point |
| trials | `0:1,2:2,*:3` by positional DoF | a 2nd trial provably changes nothing at 0 DoF; a flat budget is a DoF-dependent handicap |
| PyXtal tol factor | 1.3 | a loose initial cell + compressive relaxation is what recovers structures |
| MLIP | `orb_conserv_inf` | energy and hull must come from the same potential; ORB's checkpoint is identifiable and verified |
| reference hull | the full published `LeMat-Bulk-MLIP-Hull` split | which hull produced a number is not recoverable afterwards |

## No LeMat-GenBench dependency

The scoring half is implemented here rather than imported — LeMat-GenBench is
not on PyPI and its pinned `torch_scatter` wheels hold torch at 2.6, which
cannot coexist with our `torch ==2.11.0`.

| module | replaces |
|---|---|
| `evaluation/hull_energy.py` | `preprocess.reference_energies.get_energy_above_hull` |
| `evaluation/structure_validity.py` | `metrics.validity_metrics.OverallValidityMetric` |
| `evaluation/oxidation_state.py` | `utils.oxidation_state` (vendored verbatim) |
| `evaluation/structure_novelty.py` | `metrics.novelty_new_metric` (reference half) |

Novelty is *not* a port: it is our own `evaluation/novelty.py`, which predates
the benchmark and answers the same question with `StructureMatcher` rather than
with a hash.

`tests/test_genbench_equivalence.py` pins the ported half against the originals
and skips when LeMat-GenBench is absent:

```bash
uv sync --group genbench-oracle
LEMAT_GENBENCH_PATH=/path/to/lemat-genbench uv run pytest -k equivalence
```

See [the rationale notes](archive/de_novo_ranking_protocol_rationale.md#why-the-scoring-half-is-reimplemented-not-imported)
for the equivalence results and the one-hot-encoding trap.

## Data it needs

| path | what | note |
|---|---|---|
| `cache/lemat_bulk_fmax1_stress/data.pkl.gz` | LeMat-Bulk in the Wyckoff representation, 5,327,342 rows over train/val/test; the novelty reference | built by [the LeMat pipeline](lemat_bulk_pipeline.md); `--reference-cache` overrides |
| `cache/lemat_bulk_fmax1_stress/gene_fingerprints.pkl.gz` | 4,826,004 distinct gene fingerprints of all three splits (3,959,797 in `lemat_bulk_ehull`) | built on first `screen`, beside the reference |
| `data/lemat-bulk/lemat_pbe.csv.gz` | LeMat-Bulk CIFs, by `immutable_id` | the geometry `StructureMatcher` needs; `--lemat-cif-csv` overrides |

The hull parquet is fetched from HuggingFace and cached there. The first `screen`
takes ~11 minutes (2026-09-15, zeus): ~1.5 minutes unpickling the 5.3M-row
reference, the rest fingerprinting it. Cached, ~2 minutes, spent loading the
fingerprint set. `score` unpickles the reference once more to find the colliding
entries (~25 GB resident while it does).

## Known limitations

- **The relaxed fingerprint is a fingerprint, not a full re-analysis.** The
  score stage now re-fingerprints the relaxed structure and judges novelty
  against both it and the sampled gene's (see *Novelty and uniqueness are
  two-stage*), so a structure that relaxed onto or off a known orbit set is
  caught. What is still fingerprint-first is the *candidate* set: only
  LeMat-Bulk entries sharing one of the two fingerprints reach the matcher, so
  a match that neither fingerprint points at is missed. PyXtal symmetry
  detection also fails on some relaxed cells; those fall back to the sampled
  fingerprint alone (`relaxed_fingerprint_resolved = false`).
- **The matcher runs at pymatgen's defaults** (`ltol=0.2, stol=0.3,
  angle_tol=5`, primitive cell, scaled). A looser tolerance would find more
  matches and lower the novelty rate.
- **Single MLIP, not the ensemble.** The leaderboard averages ORB+MACE+UMA, each
  against its own hull.
- **The leaderboard is pre-relaxation.** It scores structures exactly as
  submitted, so whatever CrySPR leaves is what counts. That is the argument for
  relaxing the final submission with the ensemble models themselves.

## See also

- [The rationale notes](archive/de_novo_ranking_protocol_rationale.md) — why every default is what it is
- [The NEP89 variants](de_novo_ranking_protocol_nep89_variants.md) — two-stage NEP89→ORB, and wide-then-narrow
- [Improving de novo quality](archive/de_novo_quality_plan.md) — what to change in the model
- [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md) — where the trial and stage numbers come from
- [CrySPR reconstruction report](cryspr_reconstruction_report.md) — the rattle stage, the DoF breakdown, and the 79% reconstruction ceiling
- [Sampling temperature](temperature_sweep.md) — what the readouts do as the sampler is sharpened or flattened
- [Conditioning on the chemical system and the space group](chemical_system_conditioning.md) — what the accumulated protocol artifacts say about where the budget should go
- [Every `e_hull` in this repository](e_hull_definitions.md) — the six definitions, which agree, and which must never be mixed
