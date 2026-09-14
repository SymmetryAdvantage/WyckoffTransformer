# The de novo ranking protocol: why these settings

Historical and motivation notes for the [de novo ranking
protocol](../de_novo_ranking_protocol.md). Everything here is *why the defaults
are what they are* — the power analysis that sizes the cohort, the stage design,
the trial schedule, the potential and hull choices, and the provenance of the
element exclusions. The user guide carries the settings themselves; this file
carries the argument, and is kept for the record rather than for day-to-day use.

The numbers were measured against `upi73i4k` (a 2500-gene run,
`--condition-value 0.0`) unless stated otherwise, and predate later changes to
the defaults where noted.

## Develop against the 0.1 eV/atom threshold, report SUN

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

The default cohort of **1000 genes per arm** is this table's MetaSUN row (945)
rounded up to a round number, and it also matches `wyformer-generate`'s own
`--firm-n-samples` default.

## The stages: two symmetric, one free, then a rattle

```
1_fix-cell        FixSymmetry, cell fixed   warm-up, so a random cell is not
                                            dragged by badly placed atoms
2_sym_cell+pos    FixSymmetry, cell free    the bulk of the energy drop
3_no-sym_cell+pos no constraint, cell free  the rattle's baseline: converged,
                                            unconstrained, and usually a no-op
4_rattle_no-sym   no constraint, cell free  perturb, re-relax, keep if it wins
```

Four stages, but about **3.2 relaxation-equivalents**: stage 3 takes zero
optimiser steps in 78.2% of trials, so it costs roughly a fifth of a stage. That
is the accounting used in the cost tables below.

**Stage 3 cannot break symmetry, and the rattle can.** For a symmetry-invariant
potential the force and stress components along symmetry-lowering modes vanish
*identically* at a symmetric stationary point, so a structure that converged in
stage 2 has nothing to descend. Measured over the 9990 trials of the
[reconstruction study](../cryspr_reconstruction_report.md), stage 3 took **zero
optimiser steps in 78.2%** of them. A finite perturbation is the only thing that
leaves such a point:

| | |
|---|---|
| perturbation | 0.05 Å per-atom rattle + symmetrised cell strain, sd 0.01 |
| acceptance | keep only if `ΔE < −1 meV/atom` |
| accepted | **33.1%** of trials, median drop **−186.5 meV/atom** |
| ground states recovered | **+99** matching trials out of ~5000 |

The acceptance margin is what makes the stage safe to run unconditionally: a
converged structure can never be traded for numerical noise, so the stage costs
one relaxation and can only improve the answer. It is on by default
(`--no-rattle` to drop it), and every trial writes `rattle.json` with its ΔE and
verdict, because the acceptance rate is what the trial schedule below has to be
re-measured against.

**Stage 3 stays, as the rattle's baseline.** Taken on its own it is nearly free
of consequence — it moves the energy by more than 1 meV/atom in 0.4% of trials,
and stage-2 and stage-3 spglib space groups agree **398/398** at symprec 0.01 on
398 random `upi73i4k` genes ([trial and stage
spread](../cryspr_trial_and_stage_spread.md)) — but what it buys is not its own
energy drop. It is what the perturbation is applied to and what the acceptance
test compares against, so `ΔE < −1 meV/atom` asks *did the rattle find a better
basin* rather than *did it finish a relaxation the constrained stages had left
undone*. Rattling straight off stage 2 would confound those two, and would
compare a symmetry-broken structure against a symmetry-constrained energy. It
also still catches the trials that hit `steps_limit` under constraint, and being
a no-op 78% of the time it costs about a fifth of a stage.
`--no-release-symmetry` drops it.

## Trials by positional degrees of freedom

A trial is a *random draw of the free internal coordinates*, so what a second
trial can possibly buy is set by how many there are. `dof_positional` is the sum
over the gene's Wyckoff orbits of each orbit's degrees of freedom; lattice
parameters are deliberately not counted, because stages 1 and 2 relax the cell
from wherever it started.

Measured on the 2423 genes of `upi73i4k` with all three trials surviving, pairing
the per-gene best-of-*k* energy penalty (MACE, 3 trials) with the ORB
`e_above_hull` of the same gene — `e_hull` is affine in the energy at fixed
composition, so a per-atom energy penalty maps onto it one-to-one:

| positional DoF | share of genes | p(≤0.1) 1 trial | 2 trials | 3 trials | trials agree to 1 meV |
|---|---|---|---|---|---|
| **0** | 0.202 | **0.410** | **0.410** | **0.410** | 97.6% |
| 1–2 | 0.226 | 0.305 | 0.365 | 0.381 | 79.4% |
| 3–5 | 0.283 | 0.215 | 0.336 | 0.374 | 51.4% |
| 6–10 | 0.177 | 0.144 | 0.260 | 0.305 | 41.6% |
| >10 | 0.111 | 0.126 | 0.215 | 0.256 | 34.4% |

Two things fall out of that table.

**At zero positional DoF a second trial changes nothing at all** — not "little":
p is identical to three decimals, and 97.6% of those genes have every trial land
within 1 meV/atom. A fifth of the cohort is in that bin. Spending trials there
is pure waste, and *not* spending them pays for a second trial everywhere else.

**Best-of-1 imposes a handicap that grows with DoF** — the shortfall against
best-of-3 is 0.000 / 0.077 / 0.159 / 0.160 / 0.130 across the bins. That is the
argument for a schedule rather than a flat budget: a flat one trial does not
merely add noise that cancels between arms, it penalises whichever arm generates
higher-DoF genes. Since the changes queued up for comparison (a coordinate head,
symmetry conditioning, stability conditioning) all move the DoF mix, a
DoF-dependent handicap of up to 16 points is a confound, not noise. Shifting 10
points of an arm's mass from the 0 bin to the >10 bin moves measured MetaSUN by
~1.3 points on its own — a quarter of the 20% relative effect the protocol is
sized to detect.

The default schedule is therefore **1 trial at 0 positional DoF, 2 up to 2 DoF,
3 above that** (`--n-trials "0:1,2:2,*:3"`; bins are inclusive upper bounds, `*`
catches the rest, and a bare integer still means a constant). Against the
alternatives, with n/arm sized for a 20% relative change at α=0.05, 80% power,
and cost counted in relaxation-equivalents at 3.2 stages per trial:

| schedule | p(≤0.1) | trials/gene | n/arm | total work |
|---|---|---|---|---|
| flat 1 (the old default) | 0.252 | 1.00 | 1234 | **3948** |
| 0:1, \*:2 | 0.330 | 1.80 | 831 | 4783 |
| flat 2 | 0.330 | 2.00 | 831 | 5321 |
| **0:1, 2:2, \*:3 (default)** | **0.353** | **2.37** | **746** | **5653** |
| flat 3 | 0.357 | 3.00 | 733 | 7038 |

Reproduce with `scripts/analyse_trials_by_dof.py`.

Flat 1 is still the cheapest way to buy a given amount of *power* — raising p
shrinks n/arm sublinearly, so trials never pay for themselves on variance alone.
What the schedule buys for its 43% more total work than flat 1 — the n/arm
reduction from 1236 to 744 is already netted off in that figure — is the removal
of the DoF-dependent bias. At every quality level it strictly beats the flat
budget that matches it: it reaches 99% of flat-3's p at 0.80× flat-3's cost, and the
intermediate `0:1,*:2` reaches flat-2's p at 0.90×. Drop to `0:1,*:2` (−16%
work, largest remaining per-bin gap 4.4 points instead of 1.6) when the arms are
known to have the same DoF mix and compute is tight.

Two honest caveats. The penalty distribution is MACE's while `e_hull` is ORB's,
as in the original sizing table; and it was measured **without** the rattle
stage, which does part of the same job — 33% of trials accept a median 186
meV/atom improvement, which is the same order as a second trial's mean gain
(0.13–0.17 eV/atom). The two therefore overlap by an unknown amount, and the
schedule is likely to be *generous* now rather than tight. `structures.csv`
carries `dof_positional` and `n_trials` per gene, and each trial carries
`rattle.json`, precisely so the table above can be recomputed from the next
multi-trial run with the rattle on.

## Every unique gene is relaxed, gene-known ones included

An earlier version skipped gene-known representatives, on the grounds that a gene
already in LeMat-Bulk cannot contribute to SUN. Three reasons to relax them
anyway, and they cost only ~20% more compute:

1. **Novelty needs them.** A known gene is a candidate for `StructureMatcher`,
   not a verdict; skipping it decides novelty by fingerprint alone.
2. **The `e_above_hull` distribution becomes unbiased.** Scoring only the
   gene-novel genes conditions the energy distribution on novelty, and the
   genes it drops are exactly the ones that reproduce real materials — so the
   reported mean is biased upwards by an unknown amount.
3. **It is a CrySPR control.** These genes came from LeMat-Bulk structures, so
   whether the reconstruction recovers them measures reconstruction quality on
   a set where the right answer is known. See the [reconstruction
   study](../cryspr_reconstruction_study.md).

## PyXtal's tolerance factor stays at 1.3

`Tol_matrix(prototype="atomic", factor=1.3)` biases how close two atoms may
*start*. Two corrections to how this used to be described, both measured on
2026-09-10:

- **The multiplier is the covalent-radius *mean*, not the sum.** `Tol_matrix`
  returns `0.5 × (rₐ + r_b)` per pair, so factor 1.3 asks for contacts no
  shorter than `0.65 × (rₐ + r_b)` — 35% *shorter* than a covalent bond, not 30%
  longer. Read as the sum, 1.3 would forbid every real bonded crystal.
- **It is a bias, not a floor.** 45% of the structures PyXtal returns violate it
  (worst 0.60 in tolerance units, below PyXtal's own default of 1.0), and only 2
  of 272 violations were an atom against its own periodic image. PyXtal checks
  distances while placing atoms, not exhaustively on the finished cell.

Neither changes the conclusion. All 750 ORB-relaxed oracle references clear the
floor, the tightest by 14%, so it does not exclude the answer; and the
relaxation repairs the crowded draws, landing at a median contact of 1.82
against the ground truth's 1.83. It is retained, and the oracle studies are the
reason rather than inertia.

With the lattice free — which is how this protocol samples — a 1.3 floor makes
PyXtal draw a **loose** cell, median 1.68× the target volume, and the
variable-cell relaxation then contracts it to 1.34×. That compressive annealing
is what works: 50.8% recovery at 5 trials, 60.8% at 10, with PyXtal failing to
generate in only 1.2% of trials ([relaxed-cell
oracle](../cryspr_oracle_relaxed_cell_report.md), [lattice-free CN
pilot](../cryspr_lattice_free_cn_pilot_report.md)).

The case against 1.3 comes from a different setup and does not transfer. Pinning
the cell to the ground-truth lattice at 1.3 chokes generation — 39.8% failures,
recovery down to 16.2% — which is what motivated dropping to 1.0 (with a 0.9
fallback, since 1.0 alone fails at high DoF). But 1.0 inside a fixed equilibrium
cell jams the atoms instead: recovery 19.0%, against 50.8% for the loose 1.3
baseline. That study's own recommendation for the lattice-free decoder is to
scale the initial volume up and *keep* `factor=1.3`.

And every measurement this protocol rests on used it — the trial and stage
spread, the reconstruction ceiling, the DoF table above — so changing it would
invalidate the trial schedule those numbers set.

## ORB by default

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

## The reference hull is the whole one LeMat publishes

`e_above_hull` is computed against `data/<mlip>-00000-of-00001.parquet` of
`LeMaterial/LeMat-Bulk-MLIP-Hull`, downloaded from HuggingFace and cached. Row
counts are pinned in `PUBLISHED_HULL_ENTRIES`, the loaded count is checked
against them, and the source path, dataset revision and entry count are written
into `manifest.json` under `hull`. Passing a local parquet instead logs a
warning, and anything short of the published count logs that it is not the full
reference: which hull produced a number is not recoverable from the number
afterwards.

| split | entries | what it is |
|---|---|---|
| `orb_conserv_inf` | 194,240 | every LeMat-Bulk entry within 1 meV/atom of ORB's own hull |
| `mace_mp` | 204,976 | …of MACE-MP's |
| `uma` | 173,441 | …of UMA's |
| `mace_omat` | 168,458 | …of MACE-OMAT's |
| `orb_direct_20` | 163,735 | …of ORB-direct's |
| `dft` | 144,127 | …of the PBE hull |

**The 1 meV/atom threshold is LeMaterial's, and it is lossless for the phase
diagram.** A hull vertex sits at exactly `e_above_hull = 0`, so no positive
threshold can cut one; what it drops is entries strictly inside the hull, which
`PhaseDiagram` ignores anyway. It is the only filter the reference carries: 89
elements are present, each with an elemental reference entry, and no entry has
ten or more elements. Ra, Rn, At, Po, Am and Cm are absent because LeMat-Bulk
has no such structures, not because anything filtered them.

## Where the exclusions in the *other* hull came from

There *were* element exclusions in this repository's hull code — not on this
path, and not from `lemat-genbench`. `scripts/compute_e_hull.py` refused three
classes of system outright:

```python
if "Yb" in chemsys_in_set:                      return None, None
if NA_ELEMENTS.intersection(chemsys_in_set):    return None, None   # Po..Og
if len(chemsys_in_set) >= 10:                   return None, None
```

Provenance: they arrived with the whole CrySPR script family in commit
`e943883` ("lm related update", 2025-11-18, shuyayamazaki), and the later
rewrite `107d89b` kept the tests while dropping the comments that explained
them. Those comments were the only statement of intent there has ever been:

- `NA_ELEMENTS` (Z ≥ 84, Po through Og) — *"elements that are radioactive or not
  well-supported in common databases"*;
- `len(chemsys) >= 10` — *"Qhull, used by pymatgen, can have issues with
  high-dimensional systems"*;
- `Yb` — *"skip … certain elements like Yb or rare elements"*, i.e. no reason
  given. The usual reason to single Yb out is the Materials Project's
  `Yb_2`/`Yb_3` pseudopotential ambiguity, but that is inference, not what the
  commit says.

`lemat-genbench` contains no such list — grep it for `Yb` or `NA_ELEMENTS` and
nothing matches. Its only reference restriction is the `threshold=0.001`
argument of `get_energy_above_hull`, which is the same published slice described
above, plus a `missing reference energies` error path for formation energies of
elements with no chemical potential.

What those exclusions affected was the **training labels**: `e_hull` is what the
conditioning datasets carry, and requiring it to be non-NaN dropped 589,250 of
5,335,299 LeMat-Bulk rows — 579,217 of them for the Z ≥ 84 clause alone, while
the ten-element clause could never fire. The script was replaced by
`formula_energy/hull_table.py` and the archive relabelled without the exclusions
on 2026-09-07, recovering 589,249 of those rows while reproducing every
pre-existing label bit for bit ([what every `e_hull` in this repo
means](../e_hull_definitions.md), [dirty-data
conditioning](../dirty_data_conditioning.md)). They never touched a generated
structure's score — the published ORB hull scores Yb, U, Th and Pa compositions
without complaint (1211 Yb entries in it), and `funnel.json` now reports
`no_hull_energy`, the count of surviving structures the hull could not reach at
all, so a silent exclusion on this side would be visible.

The provenance is recorded here because it was worth knowing where a filter with
no stated reason came from, and because the same question is worth asking of the
next one.

## Why the scoring half is reimplemented, not imported

LeMat-GenBench is not on PyPI, and its pinned `torch_scatter==2.1.2+pt26cu124`
wheels hold torch at 2.6, which cannot coexist with our `torch ==2.11.0`. For a
development-time eval, portability and consistency beat tracking their changes.

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

**This replaces BAWL, which was silently vacuous.** The score stage used to hash
each structure with BAWL and test membership in
`data/unique_fingerprints.parquet`. That parquet does not hold BAWL hashes: all
4,719,106 of its entries are LeMat-GenBench *augmented Wyckoff* fingerprints
(`AUG_12_('Ba', '4j', 1):1_...`), which no BAWL hash can ever equal. Novelty
therefore came out at exactly 1.0 for every structure ever scored. In
LeMat-GenBench that file belongs to `novelty_new_metric.AugmentedNovelty`; its
BAWL `NoveltyMetric` builds its own reference by hashing LeMat-Bulk from
HuggingFace at run time.
