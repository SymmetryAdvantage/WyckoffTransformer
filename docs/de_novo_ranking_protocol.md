# The de novo ranking protocol

A development-time evaluation for comparing WyFormer variants by MetaSUN, at
roughly a fifth of the cost of a faithful LeMat-GenBench run. Implemented as
`wyformer-protocol` (`wyckoff_transformer.cli.protocol`).

This is a *ranking* instrument, not a leaderboard predictor. It is deliberately
biased in ways that cancel between arms; the final submission still goes through
the full protocol with the three-model ensemble.

## Quick start

```bash
uv run wyformer-protocol generated/<run>/wyckoff_genes.json.gz \
    --output-dir generated/<run>/protocol \
    --cores 20
```

GPU instead of CPU: `--devices cuda:0,cuda:1 --workers-per-device 2`. The two
are mutually exclusive.

Stages are separable, which matters because `relax` is the only expensive one:

```bash
wyformer-protocol genes.json.gz --output-dir run/ --stage screen   # ~2 min cached
wyformer-protocol genes.json.gz --output-dir run/ --stage relax --cores 20
wyformer-protocol genes.json.gz --output-dir run/ --stage score    # ~1 min
```

`--limit 12` on `relax` gives a smoke test. Outputs:

| file | contents |
|---|---|
| `screen.json` | validity, uniqueness with counts, gene novelty |
| `structures.csv` | per gene: energy, validity, uniqueness, novelty, `e_above_hull` |
| `funnel.json` | the whole cascade, as rates per *sampled* gene |
| `manifest.json` | MLIP, checkpoint, trials, devices |
| `cifs/`, `cryspr/` | relaxed structures and relaxation logs |

## The cascade

```
sampled → valid gene → unique gene (keep counts)
        → PyXtal + 1 trial × 2-stage CrySPR
        → valid structure → unique structure → novel structure
        → e_hull ≤ 0.1 → ≤ 0
```

**Uniqueness deduplicates but keeps counts.** Every rate stays per sampled gene;
a duplicate belongs once in the numerator and once per sample in the
denominator. Reporting over the deduplicated set instead would make uniqueness
≈1.0 by construction.

**Every unique gene is relaxed**, gene-known ones included. An earlier version
skipped them, on the grounds that a gene already in LeMat-Bulk cannot contribute
to SUN. Three reasons to relax them anyway, and they cost only ~20% more
compute:

1. **Novelty needs them.** A known gene is a candidate for `StructureMatcher`,
   not a verdict; skipping it decides novelty by fingerprint alone.
2. **The `e_above_hull` distribution becomes unbiased.** Scoring only the
   gene-novel genes conditions the energy distribution on novelty, and the
   genes it drops are exactly the ones that reproduce real materials — so the
   reported mean is biased upwards by an unknown amount.
3. **It is a CrySPR control.** These genes came from LeMat-Bulk structures, so
   whether the reconstruction recovers them measures reconstruction quality on
   a set where the right answer is known. See the [reconstruction
   study](cryspr_reconstruction_study.md).

### Novelty and uniqueness are two-stage

Both use `NoveltyFilter` and `filter_by_unique_structure` from
`evaluation/novelty.py`: the augmented Wyckoff fingerprint first, then
`StructureMatcher` on whatever shares it.

The fingerprint alone is not a verdict. Two structures with the same space
group and the same elements on the same Wyckoff orbits differ in their free
coordinates and lattice parameters, so a gene that occurs in LeMat-Bulk can
still relax into a structure that is not in it. The screen therefore produces
*candidates* for the matcher rather than a decision.

That is what makes the reference affordable. LeMat-Bulk has 4.2M entries and
the matcher needs a `Structure` per candidate, which is far too much to hold;
but only entries whose fingerprint collides with a generated one can ever reach
it. On `upi73i4k`'s 2500 genes, 627 fingerprints collide, over **795** reference
structures — a median of 1 candidate each and never more than 6. So the
reference is built per run: one streaming pass over the Wyckoff cache for the
colliding `immutable_id`s, then one chunked pass over `lemat_pbe.csv.gz` for
their geometry.

**This replaces BAWL, which was silently vacuous.** The score stage used to hash
each structure with BAWL and test membership in
`data/unique_fingerprints.parquet`. That parquet does not hold BAWL hashes: all
4,719,106 of its entries are LeMat-GenBench *augmented Wyckoff* fingerprints
(`AUG_12_('Ba', '4j', 1):1_...`), which no BAWL hash can ever equal. Novelty
therefore came out at exactly 1.0 for every structure ever scored. In
LeMat-GenBench that file belongs to `novelty_new_metric.AugmentedNovelty`; its
BAWL `NoveltyMetric` builds its own reference by hashing LeMat-Bulk from
HuggingFace at run time.

## Why these settings

### Develop against the 0.1 eV/atom threshold, report SUN

Genes per arm to detect a +20% relative change at α=0.05, 80% power, computed
from `upi73i4k`'s own `e_above_hull` distribution:

| readout | p₀ | n/arm |
|---|---|---|
| `e_hull ≤ 0` (SUN) | 0.041 | **10,136** |
| `e_hull ≤ 0.05` | 0.180 | 1,916 |
| **`e_hull ≤ 0.1` (MetaSUN)** | **0.303** | **945** |
| mean `e_hull` | — | 14,848 |
| Mann–Whitney | — | 930 |

SUN is 10× more expensive to resolve than MetaSUN for the same underlying
shift. And a continuous readout does *not* help: the `e_hull` tail is heavy
(sd 0.709 against a mean of 0.308), so the threshold indicator is already near
optimal. There is no free power in the statistic — the leverage is all in cost
per gene.

These p₀ values come from the MACE/LeMat-GenBench run that predates the ORB
default. The ORB numbers over the same genes are lower — 0.259 at ≤0.1 and
0.015 at ≤0 over valid structures — which moves the required n but not the
ordering, so the conclusion stands and the table has not been recomputed.

### One trial, two stages

Counting relaxations as cost, and modelling fewer trials as extra per-gene noise
drawn from the observed best-of-1 vs best-of-3 penalty distribution:

| protocol | p(≤0.1) | n/arm | relax/gene | total work |
|---|---|---|---|---|
| 3 trials × 3 stages | 0.303 | 948 | 9 | 8530 |
| 3 trials × 2 stages | 0.303 | 948 | 6 | 5686 |
| 2 trials × 2 stages | 0.278 | 1020 | 4 | 4079 |
| **1 trial × 2 stages** | 0.226 | 1226 | 2 | **2452** |

One trial costs 30% more genes and saves 4.5× per gene: **3.5× cheaper for equal
power**.

Dropping the third relaxation stage is safe for the *energy* — it moves by more
than 1 meV/atom in 0.4% of trials ([CrySPR trial and stage
spread](cryspr_trial_and_stage_spread.md)). It is also safe for *novelty*: stage
1 fingerprints the sampled gene, which the relaxation cannot change at all, and
stage 2 compares geometries that stage 3 barely moves. Measured on 398 random
`upi73i4k` genes, stage-2 and stage-3 spglib space groups agree **398/398** at
symprec 0.01.

### ORB by default

`--mlip` is restricted to the six splits of `LeMaterial/LeMat-Bulk-MLIP-Hull`,
and raises otherwise. `e_above_hull` is only meaningful when the structure
energy and the hull come from the same potential; per-atom offsets between
models over the 204,976 shared entries:

| pair | mean | sd | p90 abs |
|---|---|---|---|
| mace_mp − uma | −0.057 | 0.106 | 0.157 |
| mace_omat − uma | 0.002 | 0.017 | 0.023 |

The effect size we are chasing is 0.023 eV/atom, so a model/hull mismatch is
2.5× the signal. ORB is the default because:

- it tracks the three-model ensemble mean twice as closely as MACE (residual sd
  0.036 vs 0.071 eV/atom, offsets removed) — UMA is equally close, so this alone
  does not decide;
- its checkpoint is identifiable from source. `orb-models` hard-codes
  `orb-v3-conservative-inf-omat-20250404.ckpt` as a default argument, identical
  from v0.5.1 to v0.7.0. A test asserts our recorded URL is what the installed
  package loads;
- `verify_hull_energies("orb_conserv_inf")` reproduces the published energies to
  **90 µeV/atom**, so the pairing is confirmed, not assumed.

**Open item — the `mace_mp` hull's checkpoint is unidentified.** LeMat-GenBench
built it by calling `mace_mp()` with no `model` argument, whose meaning changed
in mace-torch 0.3.10. Neither candidate reproduces the published energies:
MACE-MP-0a-medium is off by mean +1.0 meV/atom (max 22), MACE-MPA-0-medium by
+4.6 (max 14); float32 and float64 agree to 1e-6, so dtype is not the cause.
Treat anything from that hull as carrying a few meV/atom of unexplained
systematic error. Recorded in `HULL_MLIPS["mace_mp"].note`.

## No LeMat-GenBench dependency

The scoring half is implemented here rather than imported. LeMat-GenBench is not
on PyPI, and its pinned `torch_scatter==2.1.2+pt26cu124` wheels hold torch at
2.6, which cannot coexist with our `torch ==2.11.0`. For a development-time eval,
portability and consistency beat tracking their changes.

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
on 12 real CrySPR structures: validity verdicts and charge deviations match
exactly, and `e_above_hull` agrees to 1e-9. Those tests are the only reference
to LeMat-GenBench and skip when it is absent:

```bash
uv sync --group genbench-oracle
LEMAT_GENBENCH_PATH=/path/to/lemat-genbench uv run pytest -k equivalence
```

A trap worth knowing: LeMat-GenBench contains two incompatible one-hot encodings.
`preprocess.reference_energies` uses 119 slots indexed by atomic number;
`fingerprinting.encode_compositions` uses 118 indexed by `Z−1`. Only the first
matches the stored composition matrices — the second raises a shape error against
them.

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

- **Novelty is judged on the sampled gene, not the relaxed one.** Stage 1 uses
  the gene WyFormer emitted; relaxation can lower the symmetry, so a structure
  whose *relaxed* fingerprint is in LeMat-Bulk while its sampled one is not will
  be called novel without the matcher ever seeing it. Bounding this needs the
  relaxed structure re-fingerprinted, which the score stage does not yet do.
- **The matcher runs at pymatgen's defaults** (`ltol=0.2, stol=0.3,
  angle_tol=5`, primitive cell, scaled). A looser tolerance would find more
  matches and lower the novelty rate.
- **Single MLIP, not the ensemble.** The leaderboard averages ORB+MACE+UMA, each
  against its own hull.
- **The leaderboard is pre-relaxation.** It scores structures exactly as
  submitted, so whatever CrySPR leaves is what counts. That is the argument for
  relaxing the final submission with the ensemble models themselves.

## See also

- [Improving de novo quality](de_novo_quality_plan.md) — what to change in the model
- [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md) — where the trial and stage numbers come from
