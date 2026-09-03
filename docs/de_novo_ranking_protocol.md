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
| `structures.csv` | per gene: energy, validity, BAWL hash, `e_above_hull` |
| `funnel.json` | the whole cascade, as rates per *sampled* gene |
| `manifest.json` | MLIP, checkpoint, trials, devices |
| `cifs/`, `cryspr/` | relaxed structures and relaxation logs |

## The cascade

```
sampled → valid gene → unique gene (keep counts)
        → gene-novel?  no  → count into the denominator, stop
                        yes → PyXtal + 1 trial × 2-stage CrySPR
        → valid structure → BAWL-unique → BAWL-novel → e_hull ≤ 0.1 → ≤ 0
```

Two ideas make it cheap.

**Free filters run before the expensive one.** A gene that is not novel cannot
contribute to SUN or MetaSUN by definition, so it is counted into the
denominator and never relaxed. On `upi73i4k` that skips 627 of 2500 genes — 25%
of the relaxation budget, for free.

**Uniqueness deduplicates but keeps counts.** Every rate stays per sampled gene;
a duplicate belongs once in the numerator and once per sample in the
denominator. Reporting over the deduplicated set instead would make uniqueness
≈1.0 by construction.

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
spread](cryspr_trial_and_stage_spread.md)) — and also for *novelty*, which was
the open question, since BAWL hashes include a space-group label. Measured on
398 random `upi73i4k` genes, stage-2 and stage-3 spglib space groups agree
**398/398** at symprec 0.01.

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
| `evaluation/bawl.py` | `material_hasher.hasher.bawl` |
| `evaluation/bawl_reference.py` | `data.reference_fingerprint_loader` |

One new runtime dependency, `structuregraph-helpers`, which keeps the
Weisfeiler-Lehman hash itself out of our code.

`tests/test_genbench_equivalence.py` pins all of it against the originals on 12
real CrySPR structures: BAWL hashes are string-identical for both variants,
validity verdicts and charge deviations match exactly, and `e_above_hull` agrees
to 1e-9. Those tests are the only reference to LeMat-GenBench and skip when it is
absent:

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
| `data/unique_fingerprints.parquet` | 4.72M BAWL fingerprints | copy from LeMat-GenBench's `data/augmented_fingerprints/`; `$BAWL_REFERENCE_PARQUET` overrides |

The hull parquet is fetched from HuggingFace and cached there. The first `screen`
takes ~8 minutes, almost all of it unpickling the 4.2M-row reference; cached, ~2
minutes.

## Known limitations

- **Gene novelty is not BAWL novelty.** The screen filters on the augmented
  Wyckoff fingerprint; the benchmark scores the relaxed structure's BAWL hash.
  Aggregate rates agree closely on `upi73i4k` — 0.7492 gene-novel against 0.740
  BAWL-novel — but that is not per-structure agreement, and genes routed to the
  free branch may have relaxed into novel structures. Fine for ranking while the
  transfer rate is stable across variants; not a leaderboard estimate.
- **Single MLIP, not the ensemble.** The leaderboard averages ORB+MACE+UMA, each
  against its own hull.
- **The leaderboard is pre-relaxation.** It scores structures exactly as
  submitted, so whatever CrySPR leaves is what counts. That is the argument for
  relaxing the final submission with the ensemble models themselves.

## See also

- [Improving de novo quality](de_novo_quality_plan.md) — what to change in the model
- [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md) — where the trial and stage numbers come from
