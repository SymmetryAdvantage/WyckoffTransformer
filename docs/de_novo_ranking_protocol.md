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
| `screen` | one core, 4M fingerprints in RAM | ~2 min cached, ~8 min cold | `screen.json` |
| `generate` | every CPU core, no potential | seconds per draw | `pyxtal.extxyz`, `pyxtal.csv` |
| `relax` | the MLIP, on GPU | ~2-8 s per trial per K20c-class GPU | `relaxations.csv`, `structures.csv`, `cifs/` |
| `score` | the hull parquet and LeMat-Bulk geometry in RAM | ~1 min | `funnel.json` |

There is a fifth, optional stage. `template` adds one
[template-matched start](cryspr_template_starts.md) per gene — a training
structure's lattice and coordinates instead of a PyXtal draw — as an extra
trial, to be relaxed and scored alongside the random ones. It is not part of
`--stage all` and changes no default.

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
`--pyxtal-timeout` (300 s) bounds one draw and `--relax-timeout` (1800 s) one
trial, and an abandoned one is recorded as such rather than silently missing.

**The unit of work is a trial, not a gene.** Trial budgets differ by a factor
of three, so a pool keyed by gene leaves workers idle at the end; both stages
therefore queue `(gene, trial)` pairs, and both append a row per pair as it
finishes. `--resume` (on by default) then restarts at the trial granularity,
and `--no-resume` starts over.

`--limit 12` gives a smoke test on the first twelve genes. Outputs:

| file | contents |
|---|---|
| `screen.json` | validity, uniqueness with counts, gene novelty |
| `pyxtal.extxyz` | every generated draw, tagged with its gene and trial |
| `pyxtal.csv` | per trial: PyXtal status (`ok`/`failed`/`timeout`), formula, DoF, seconds |
| `relaxations.csv` | per trial: status, energy, device, seconds, kept CIF, and the error if it failed |
| `structures.csv` | per gene: the lowest-energy trial, plus validity, uniqueness, novelty, `e_above_hull`, `dof_positional`, `n_trials`, `gene_novel`, `novel_by_sampled_gene`, `relaxed_fingerprint_resolved`, `relaxed_fingerprint_changed` |
| `funnel.json` | the whole cascade, as rates per *sampled* gene, plus `gene_known_became_novel` / `gene_novel_became_known` |
| `manifest.json` | MLIP, checkpoint, trial schedule, rattle, devices, timeouts, hull provenance |
| `cifs/`, `cryspr/` | relaxed structures, relaxation logs, per-trial `rattle.json` |

A gene with no structure carries the reason in `structures.csv`'s `error`
column, and the trial that produced it is in `relaxations.csv`. This matters:
a cohort whose potential fails to load and one whose genes PyXtal cannot draw
both read as `has_structure = 0` in the funnel and nowhere else.

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
- **A conditional run needs its target.** Datasets are not loaded here, so the
  conditioning cannot be sampled from the training distribution — pass it with
  `--condition energy_above_hull=0` (repeat once per feature, or
  `--condition-value` for a single-channel model). An unconditional run takes
  neither.
- **The run's model files must be reachable.** `runs/<run-id>/` is used if it
  already holds `best_model_params.pt`, `wyckoff_processor.json` and
  `spacegroup_distribution.json`; otherwise they are downloaded from the run.
- **What lands on the run.** Every key in `funnel.json` is flattened into
  `run.summary` under a `protocol/` prefix (`protocol/metasun_per_sampled_gene`,
  `protocol/valid_gene_rate`, `protocol/gene_known_became_novel`, …). `screen.json`, `pyxtal.extxyz`,
  `pyxtal.csv`, `relaxations.csv`, `structures.csv`, `funnel.json`,
  `manifest.json` and `cifs/` go into an artifact named
  `protocol_<run-id>` of type `protocol_eval`. `--no-upload` runs everything
  and skips only the write-back; `--entity` / `--project` override where the
  run is looked up.

The same hardware, trial-schedule, MLIP and reference flags as `wyformer-protocol`
are accepted and passed straight through.

- **Re-scoring without re-relaxing.** `--from-artifact --stages score` downloads
  the run's existing `protocol_<run-id>` artifact into `--output-dir` and runs
  only the score stage on it — no cohort is generated, nothing is relaxed. The
  refreshed `funnel.json` and `structures.csv` go back as a new artifact
  version and `run.summary` is overwritten. Use it after a change to how
  novelty or the hull is judged. `--from-artifact v2` pins a version instead of
  taking the latest.

## The cascade

```
sampled → valid gene → unique gene (keep counts)
        → PyXtal + 1–3 trials × 4-stage CrySPR (2 symmetric, free, rattle)
        → valid structure → unique structure ─┬─→ e_hull ≤ 0.1 (metastable) → ≤ 0 (stable)
                                              └─→ novel structure (sampled + relaxed fingerprint)
                                                  → e_hull ≤ 0.1 (MetaSUN) → ≤ 0 (SUN)
```

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

That is what makes the reference affordable. LeMat-Bulk has 4.2M entries and
the matcher needs a `Structure` per candidate, which is far too much to hold;
but only entries whose fingerprint collides with a generated one can ever reach
it. On `upi73i4k`'s 2500 genes, 627 fingerprints collide, over **795** reference
structures — a median of 1 candidate each and never more than 6. So the
reference is built per run: one streaming pass over the Wyckoff cache for the
colliding `immutable_id`s, then one chunked pass over `lemat_pbe.csv.gz` for
their geometry.

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
| `cache/lemat_bulk_ehull/data.pkl.gz` | LeMat-Bulk in the Wyckoff representation | existing |
| `cache/lemat_bulk_ehull/gene_fingerprints.pkl.gz` | 3.96M gene fingerprints | built on first `screen` |
| `data/lemat-bulk/lemat_pbe.csv.gz` | LeMat-Bulk CIFs, by `immutable_id` | the geometry `StructureMatcher` needs; `--lemat-cif-csv` overrides |

The hull parquet is fetched from HuggingFace and cached there. The first `screen`
takes ~8 minutes, almost all of it unpickling the 4.2M-row reference; cached, ~2
minutes.

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
- [Improving de novo quality](archive/de_novo_quality_plan.md) — what to change in the model
- [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md) — where the trial and stage numbers come from
- [CrySPR reconstruction report](cryspr_reconstruction_report.md) — the rattle stage, the DoF breakdown, and the 79% reconstruction ceiling
- [Every `e_hull` in this repository](e_hull_definitions.md) — the six definitions, which agree, and which must never be mixed
