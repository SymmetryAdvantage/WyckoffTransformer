# Template-matched starts

> **Implemented and measured.** The variant described here is
> `wyformer-protocol --stage template` and
> `wyckoff_transformer.cryspr.template`; the two evaluations below were run on
> the cohorts named in each section. It is candidate 3 —
> ["prototype retrieval as the prior and the baseline"](archive/pyxtal_dof_reduction_study.md#3-prototype-retrieval-as-the-prior-and-the-baseline)
> — of the dof reduction study, and it is *not* on by default.

## What it is worth, in one paragraph

On the 400-structure oracle cohort it is transformative: **one template start
recovers 81.5% of the targets against 60.8% for ten random ones**, and costs a
third of the optimiser steps. On the de novo protocol it is worth less, and
*where* it is worth less is the whole story: added to run `e9ywwsie` it moves
MetaSUN 0.281 → 0.293 overall, but that is **0.351 → 0.374 on the genes whose
fingerprint LeMat-Bulk does not have** and **0.108 → 0.094 on the genes it
does**. On a gene the training set already names, the template start lands on
the very structure that makes the gene known, wins the trial on energy, and
displaces a novel one. Withholding it there — which is a change to the
retrieval, not to the search — gives 0.297 and costs nothing.

**It is off by default**, because on a mixed cohort the gain is small. It earns
its place on gene-novel cohorts, and outright where reproducing a known
structure is the objective: CSP against a named target, and generating geometry
training targets for a learned realizer.

## What it does

`single_pyxtal` hands a gene to `pyxtal.from_random`, which guesses the cell
volume, the cell shape and every free Wyckoff coordinate by rejection sampling.
The [dof reduction study](archive/pyxtal_dof_reduction_study.md) established
what that guessing costs: the true cell *and* the true coordinates together
recover the target at every degree of freedom, while either alone recovers
about 0.17 above 10 positional dof.

A gene does not come with its own coordinates. A training structure on the same
Wyckoff orbits does. So, per gene:

1. compute the gene's **anonymous Wyckoff fingerprint** — space group plus the
   multiset of occupied orbits, elements dropped
   (`record_to_anonymous_fingerprint`) — and take every LeMat-Bulk entry that
   shares it;
2. keep the candidate whose **chemical formula is closest** to the gene's;
3. start the relaxation from that structure's **lattice and coordinates**, with
   the gene's elements written onto its orbits;
4. where no candidate exists, fall back to the **ordinary randomised start**.

Nothing is trained and nothing is fitted. The whole method is a lookup, an
assignment problem and a relabelling.

### Why the *anonymous* fingerprint

Matching on the augmented fingerprint — elements included — would restrict
templates to structures the gene is already a chemical substitution of. That is
both far rarer (0.251 of `upi73i4k` genes against 0.668 for the anonymised key,
[dof reduction study](archive/pyxtal_dof_reduction_study.md)) and precisely the
case where the generated structure is least likely to be novel. What transfers
between two structures on the same orbits is *geometry*, and geometry is what
the sampler is bad at.

Note that this key is looser than the one that study measured coverage with.
`analyse_gene_priors.py` anonymises but keeps the **partition** of orbits into
species — which sites share an element, though not which element. This one
drops that too, so a gene with two orbits of the same element can be templated
on a structure that has two different elements there. The assignment step below
is what makes that safe.

### Choosing among the candidates

Two structures on the same orbits hold the same number of atoms, so "closest
formula" can be made exact: `composition_distance` is the cost of the cheapest
one-to-one substitution of the template's atoms into the gene's, per atom.
Atoms of an element both share are cancelled first and cost nothing, so only
the difference between the two formulas enters the assignment. Element
dissimilarity is the normalised **Mendeleev-number** gap, which walks each group
of the periodic table before moving to the next — the alkali metals occupy 8–12
and the halogens 97–102 — so chemically interchangeable elements are near
neighbours. Atomic number would rank Li closer to Be than to Na, which is the
wrong answer here.

The pool is not small: a common fingerprint is shared by tens of thousands of
entries, so candidates are first ranked by the atoms their formula already
shares with the gene (one dictionary pass) and only the best 64 reach the
assignment. Ties — and identical formulas tie at zero often — are broken by
**hull distance**: of two prototypes for one composition, the more stable one is
the better guess at where the atoms sit.

### Writing the gene onto the template

The template's Wyckoff letters are re-derived with `pyxtal.from_seed` and
checked against the gene's; a tolerance that does not reproduce the space group
and the orbit multiset is rejected and the next one tried, and a candidate that
never agrees is skipped in favour of the next candidate. That check is what
makes the loose key safe: the anonymous fingerprint is a set over *equivalent
enumerations*, so two records can share it while their letters differ by an
affine-normaliser relabelling, and coordinates only transfer under the identity
relabelling.

Orbits of the same letter are interchangeable, so within each letter the gene's
elements are assigned to the template's by the cheapest total element distance.
An oxygen orbit in the template goes to the gene's oxygen, or failing that to
its most oxygen-like element — not to whichever orbit the gene happens to list
first.

The lattice is the template's, **unscaled**. That is the point of the method and
also its main exposure: a template of much smaller atoms hands over a cell that
is too tight. The measurements below say how often that actually happens.

## How to use it

The template start is an extra **stage** of `wyformer-protocol`, not a
replacement for `generate`, and it is not part of `--stage all`. It appends one
template-matched draw per gene, filed as trial `TEMPLATE_TRIAL` (1000), to
whatever draws `generate` produced; `relax` then treats it as one more trial and
`score` picks the lowest-energy one as usual.

```bash
# on a fresh run
wyformer-protocol genes.json.gz --output-dir run/ --stage screen
wyformer-protocol genes.json.gz --output-dir run/ --stage generate
wyformer-protocol genes.json.gz --output-dir run/ --stage template
wyformer-protocol genes.json.gz --output-dir run/ --stage relax --devices cuda:0,cuda:1
wyformer-protocol genes.json.gz --output-dir run/ --stage score

# on a run that has already been relaxed and scored: --resume relaxes only the
# template draws and re-runs none of the random ones
wyformer-protocol genes.json.gz --output-dir run/ --stage template
wyformer-protocol genes.json.gz --output-dir run/ --stage relax --devices cuda:0 --resume
wyformer-protocol genes.json.gz --output-dir run/ --stage score
```

The stage builds `cache/lemat_bulk_ehull/anonymous_wyckoff_index.parquet` on
first use — one pass over the 4.2M-row Wyckoff cache, about two minutes, 104 MB
— and loads it thereafter. Selecting a template for a thousand genes then costs
about a minute, and reading their geometry one streaming pass over the LeMat-Bulk
CIF export.

`--template-candidates` (default 4) sets how many candidates are carried per
gene. More than one matters because the closest can resist symmetry detection,
and reading its geometry costs a pass over the export either way.

## Implementation

| piece | where |
|---|---|
| the index, the formula match, the rebuild | `src/wyckoff_transformer/cryspr/template.py` |
| the protocol stage | `wyformer-protocol --stage template` (`cli/protocol.py`) |
| unit tests | `src/wyckoff_transformer/cryspr/tests/test_template.py` |
| oracle study (experiment 1) | `scripts/run_template_reconstruction_study.py` |
| protocol comparison (experiment 2) | `scripts/analyse_template_protocol.py` |

`TemplateIndex` is LeMat-Bulk reduced to four columns — the fingerprint hash,
the Wyckoff letters, the composition and the hull distance — because that is
everything the *ranking* needs. Geometry is fetched only for the handful of
candidates that are actually chosen, through
`structure_novelty.load_reference_structures`, which is the same streaming read
the score stage's novelty reference uses.

The fingerprint hash is a `blake2b` digest rather than Python's `hash()`: the
fingerprint contains strings, whose hashing is salted per process, so a
persisted index would not survive the interpreter that built it.

## Experiment 1 — the 400-structure oracle cohort

400 LeMat-Bulk hull structures with total DoF ≥ 6, the cohort every other
oracle arm is measured on
([coordination study](archive/cryspr_oracle_coordination_study_report.md)),
relaxed with ORB-v3 conservative-inf through the same four stages at the same
`fmax = 0.02`. The random arms are the existing
`generated/cryspr_reconstruction_study` trials, read verbatim — `base5` and
`base10` reproduce the published 50.8% and 60.8% exactly, which is the check
that nothing about the comparison drifted. **The only new relaxations are the
386 template starts, one per structure.**

The target is itself a LeMat-Bulk entry, so two exclusions keep the arm honest:
its own `immutable_id`, and any candidate `StructureMatcher` says *is* the
target. The second one is not a formality. It removed 110 candidates over 88 of
the 400 structures — LeMat-Bulk merges several source databases, so the target
is routinely present a second time under another id — and it took the number of
templates with an *identical* formula from 91 down to 6. Without it this study
would have measured how often a structure matches itself.

| arm | trials | recovery | ceiling | share within 10 meV/atom of the target |
|---|---:|---:|---:|---:|
| `base1` — one random start | 1 | 27.2% | 27.2% | 31.5% |
| `base5` — five random starts | 5 | 50.8% | 51.8% | 59.0% |
| `base10` — ten random starts | 10 | 60.8% | 63.8% | 70.2% |
| **`tmpl1` — one template start** | **1** | **81.5%** | 81.5% | 85.0% |
| `tmpl_strict` — template, no fallback | ≤1 | 81.0% | 81.0% | 84.5% |
| `tmpl_base4` — template + four random | 5 | 82.8% | 85.0% | 88.0% |

`tmpl1` is the method as specified: the template start where a training
structure shares the gene's orbits (386 of 400) and the ordinary random start
where none does (14). `tmpl_strict` drops that fallback, so the 14 count as
failures; the gap between the two is the whole contribution of clause (d), and
it is 0.5 points.

**One template start beats ten random ones by 21 points and one random one by
54.** It also beats them on 136 structures that `base5` misses while losing 13
that `base5` finds. A fifth trial spent on a template rather than a random draw
(`tmpl_base4` against `base5`) is worth 32 points.

### It helps most exactly where the sampler fails

| positional DoF | n | `base1` | `base5` | `base10` | `tmpl1` |
|---|---:|---:|---:|---:|---:|
| 3–5 | 115 | 44.3% | 73.9% | 80.9% | **93.0%** |
| 6–10 | 162 | 31.5% | 56.8% | 66.0% | **85.2%** |
| >10 | 123 | 5.7% | 21.1% | 35.0% | **65.9%** |

| sites | n | `base5` | `base10` | `tmpl1` |
|---|---:|---:|---:|---:|
| ≤10 | 103 | 57.3% | 65.0% | **85.4%** |
| 11–20 | 206 | 59.2% | 68.4% | **85.9%** |
| 21–40 | 60 | 26.7% | 41.7% | **70.0%** |
| >40 | 31 | 19.4% | 32.3% | **61.3%** |

The bin the [dof reduction study](archive/pyxtal_dof_reduction_study.md) called
hopeless — above 10 positional dof, where a random draw recovers 21% at five
trials — recovers 66% from a single template start. Triclinic, at 0.0% in every
random arm including ten trials, recovers 4 of its 8 structures.

### And it costs less than a random start, not more

| | template start | random start |
|---|---:|---:|
| force at the start, median | 0.50 eV/Å | 18.87 eV/Å |
| BFGS steps over the four stages, median | 68 | 204 |
| BFGS steps, mean | 90 | 251 |

A template start begins near a minimum, so the relaxation is a third of the
length. The 386 starts took 21 minutes of wall clock on five workers over two
K20c cards and a GTX 750 Ti, against 3.3 s per start; the equal-budget
`tmpl_base4` arm is therefore *cheaper* than `base5`, not merely equal.

### The lattice does not need rescaling

The dof study's version of this proposal rescales the template's cell to a
predicted volume. On this cohort it does not need to: the relaxation changes the
template cell's volume by a median factor of **1.002** (5th–95th percentile
0.86–1.31), so the template's cell is already the right size. That is the
opposite of what a random draw hands over — a cell 1.68× the target that the
relaxation has to contract — and the reason the compressive-annealing argument
does not apply here.

Only **11 of 386** template starts (2.8%) break the
`Tol_matrix(factor=1.3)` distance floor, and those are the failures: they
recover 27.3% against 85.6% for the rest. Repairing that 2.8% — by rescaling
just those, or by the soft-sphere solve of
[proposal 4](proposed_pyxtal_generation_fixes.md#proposal-4--replace-rejection-sampling-with-a-continuous-solve)
— is worth about 1.6 points and is the obvious next increment.

### Where it fails

| formula distance | n | `base5` | `tmpl1` |
|---|---:|---:|---:|
| exact (0) | 6 | 16.7% | **0.0%** |
| ≤ 0.005 | 246 | 53.7% | **92.7%** |
| ≤ 0.02 | 70 | 52.9% | **82.9%** |
| > 0.02 | 64 | 46.9% | **59.4%** |

The trend is what one would expect except at the top. The six templates with an
*identical* formula recover nothing — and that is not noise about small numbers,
it is the definition biting: a template with the same space group, the same
orbits and the same formula that is nevertheless a different structure is a
**competing polymorph**, so its free coordinates are the other mode of a bimodal
distribution. `mp-6999` (PS4Sc, 18 positional dof) is templated on
`agm2000110430`, the same P₂S₈Sc₂ in the same orbits, and lands in the wrong
basin. Duplicate exclusion is what leaves only these behind; a chemically
*nearby* formula is a better template than the same one.

The 14 structures with no template at all are hard for everyone: `base5`
recovers 21.4% of them, against 50.8% over the cohort, and their median
positional dof is 12.5.

## Experiment 2 — the de novo protocol, run `e9ywwsie`

1000 sampled genes, 999 unique, ORB-v3 conservative-inf at `fmax = 0.05`. The
run's 2402 random-start relaxations are reused verbatim; the `template` stage
added one start per gene, and **only those were relaxed**. 858 of the 999 genes
(85.9%) had a template, higher than the 66.8% coverage the dof study measured
because the key here is the fully anonymous one.

Four arms, all over the same genes:

| arm | what it relaxes | relaxations | relaxation time |
|---|---|---:|---:|
| `random1` | the run's trial 0 only | 995 | 20.2 ks |
| `template_first` | the template start, or the run's random trials where there is none | 1262 | 25.7 ks |
| `random` | the run as published | 2402 | 56.9 ks |
| `union` | everything, lowest energy kept | 3258 | 66.6 ks |

All four are **re-scored here**, `random` included. The run's published funnel
was written before commit `9bf7073`, which changed both what counts as novel
and what `metastable` means, so comparing against it would have credited the
score stage's revision to the arm. (For the record, the published funnel says
MetaSUN 0.303 and SUN 0.014; re-scored, the same relaxations give 0.281 and
0.011.)

| per 1000 sampled genes | `random1` | `template_first` | `random` | `union` |
|---|---:|---:|---:|---:|
| valid, unique structure | 0.923 | 0.925 | 0.924 | 0.925 |
| **novel** structure | **0.704** | 0.661 | 0.687 | 0.669 |
| unique and e_hull ≤ 0.1 | 0.384 | 0.457 | 0.477 | **0.509** |
| unique and e_hull ≤ 0 | 0.028 | 0.042 | 0.032 | **0.043** |
| **MetaSUN** | 0.206 | 0.239 | 0.281 | **0.293** |
| **SUN** | 0.008 | 0.013 | 0.011 | **0.013** |
| median e_above_hull | 0.133 | 0.108 | 0.099 | **0.089** |

Paired over the 1000 genes (McNemar on MetaSUN membership):

| comparison | genes gained | genes lost | p |
|---|---:|---:|---:|
| `random1` → `template_first` | 72 | 39 | 0.0022 |
| `random` → `union` | 21 | 9 | 0.043 |
| `random` → `template_first` | 23 | 65 | <0.0001 |

Three things follow, and the second is the one that matters.

**One template start beats one random start.** At a matched budget of about one
relaxation per gene, MetaSUN goes from 0.206 to 0.239, SUN from 0.008 to 0.013,
and the median `e_above_hull` from 0.133 to 0.108 — 72 genes gained against 39
lost, p = 0.0022. The oracle study's finding survives contact with real
generated genes.

**It does not beat *three* random starts.** Against the protocol's own 2.4-trial
schedule, `template_first` loses: 0.239 against 0.281, 65 genes lost against 23
gained. A template start is one point on the manifold, and the schedule's second
and third trials are still buying what a single good point cannot. Replacing the
random budget with a template start is a way to spend 45% of the relaxation time
for 85% of the MetaSUN, not a way to win.

**Added on top, it is a small and cheap gain.** `union` costs 17% more
relaxation time than `random` — template trials are cheap, a median of 4.2 s
against 13.0 s — and moves MetaSUN from 0.281 to 0.293 and SUN from 0.011 to
0.013, with 21 genes gained against 9 lost (p = 0.043). The template trial was
the lowest-energy one for 325 of the 856 genes that had one.

### Novelty is what it costs — but only on genes LeMat-Bulk already has

Every arm that uses a template start is less novel overall: 0.704 → 0.661 at one
start per gene, 0.687 → 0.669 when added to the full budget. Going from `random`
to `union` adds 32 structures below 0.1 eV/atom and only 12 of them survive the
novelty filter.

Splitting by the *sampled gene's* own novelty says where those 20 went. Scoring
each of the 3258 trials separately rather than each gene
(`analyse_template_protocol.py ceiling`) gives, per sampled gene of each subset:

| | gene-novel (712 sampled) | gene-known (288 sampled) |
|---|---:|---:|
| `random` delivers | 0.351 | 0.108 |
| `union` delivers | **0.374** | **0.094** |
| template trials that are themselves MetaSUN | 179 | 16 |
| template trials metastable but **known** | 13 | 208 |
| genes lost to selection, `random` → `union` | 3 → 5 | 11 → 18 |

The two columns are opposite. On a gene whose fingerprint LeMat-Bulk does not
have, the template start is a clean gain: 179 of its trials are novel *and*
below 0.1 eV/atom, only 13 are metastable-but-known, and MetaSUN rises by 2.3
points. On a gene whose fingerprint LeMat-Bulk *does* have, the template start is
retrieving the structure that makes the gene known — 208 of 287 such trials are
metastable and known — and because it wins the trial on energy it evicts the
novel structure a random draw had found. MetaSUN there **falls**, by 1.4 points.

That mechanism is forced rather than incidental. Selection is by lowest energy,
and at fixed composition `e_above_hull` is affine in the total energy, so the
lowest-energy trial is also the lowest-`e_hull` trial. The only way a gene can
lose MetaSUN at the selection step is a lower-energy trial that is **not novel**
displacing a higher-energy one that is — which is precisely what retrieving a
training structure does on a gene the training set already contains.

The fix is on the retrieval side and needs no new relaxation: withhold the
template start where the gene's own *augmented* fingerprint is in LeMat-Bulk, or
equivalently exclude from its candidates the entries that share it. Composed out
of the trials already on disk that arm delivers **0.297**, against 0.293 for
`union` and 0.281 for `random`.

### How much room is left

The arms say what the search delivered; they cannot say what it could have
delivered, because a gene whose best structure is genuinely above the hull looks
exactly like one whose good structure was never sampled. The per-trial scores
separate the two. *Delivered* is the verdict on the lowest-energy trial;
*ceiling* is whether **any** trial of that gene is valid, novel and at or below
0.1 eV/atom. (Reconstructing `random` and `union` this way reproduces the
funnel's 0.281 and 0.293 exactly, which is the check on the per-trial pass.)

Over the 712 gene-novel sampled genes, per sampled gene of that subset:

| trials per gene | delivered | ceiling |
|---|---:|---:|
| 1 random | 0.249 | 0.249 |
| 2 random | 0.319 | 0.319 |
| the full random schedule (2.64) | 0.351 | 0.355 |
| plus the template start (3.44) | **0.374** | 0.381 |

Two things follow.

**The search is nowhere near done.** The curve is still rising by five to seven
points per increment of trials and shows no sign of flattening; the second
random trial alone is worth seven points. Whatever the true ceiling for these
genes is, this cohort has not approached it, and the template start moves the
*bound* rather than closing the gap to it.

**Selection is not the problem — on novel genes.** The any-trial ceiling sits
0.4 to 0.7 points above what the lowest-energy rule delivers, five genes in 712.
That is the reconstruction study's "coverage ≈ delivery" again: almost nothing
novel and stable is found and then discarded. All the selection loss is on the
gene-known subset, where it grows from 11 genes to 18 once the template start is
added — the eviction described above.

Selecting the lowest-energy trial *among the novel* ones attains the ceiling
exactly, and would give **0.316** over all 1000 genes against the 0.293 that
`union` reports. That is a change of readout, not of search, and it is not free
of interpretation: the structure it would report is metastable with respect to a
known polymorph the same run also found.

**What this ceiling is not.** It is a lower bound in three ways: it is bounded
by the 3258 trials that were actually run, it uses ORB's `e_above_hull` rather
than DFT's, and it applies no uniqueness filter (which does not bind here —
every arm has exactly one fewer unique structure than valid one). The true
MetaSUN ceiling for this gene sample is unknown, and the rising curve says it is
higher than 0.381.

### Two relaxations wedged

856 of the 858 template starts relaxed. The other two — gene 951 (`Li12N8P4`,
I-42d, 24 atoms) and gene 952 (`Cu8Ho4Ru4`, Cmcm, 16 atoms) — stopped making
progress inside a single force evaluation, at 99% CPU with the GPU idle, and
wrote no further BFGS line for over an hour. The stage's `--relax-timeout` could
not stop them: `time_limit` raises from a `SIGALRM` handler, and a handler only
runs at a Python bytecode boundary, so a call that stays inside C indefinitely is
not interruptible by it. Both genes fall back to their random trials in every
arm, so nothing here depends on them.

This is a pre-existing hazard of the relaxation stage rather than of template
starts — but template starts are what reached it, and a wall-clock watchdog that
does not rely on `SIGALRM`, or a step cap on the optimiser, would close it.

## The two limits

**It can hand over a cell of the wrong size.** The template's lattice is used
unscaled, and a template whose atoms are smaller gives a cell the gene's atoms
do not fit in. The dof study's version of this proposal says to "rescale to the
predicted (or composition-estimated) volume" for exactly this reason. It is not
implemented, and on the oracle cohort it is not needed — the relaxation changes
the template cell's volume by a median factor of 1.002 — but the 2.8% of starts
that do break the distance floor recover 27% against 86% for the rest, so
repairing those is the one clear increment left.

**It copies the training set — on the genes that were already copies.** This is
measured, not a risk: `novel_structure` falls from 0.704 to 0.661 at one start
per gene. But the per-trial split above localises it entirely to genes whose own
fingerprint LeMat-Bulk has, where the retrieved template *is* the structure that
makes the gene known. On gene-novel genes the template start adds 179 novel
metastable trials against 13 known ones. Excluding candidates that share the
gene's augmented fingerprint would close it; that is the one change to the
retrieval this study leaves undone.

## Reproduce

```bash
# experiment 1: the 400-structure, DoF >= 6 oracle cohort, 5-trial budget.
# `match` selects and rebuilds on the CPU, `relax` is the only GPU stage, and
# it relaxes 386 starts and no random draws at all.
scripts/platforms/iapetus/run.sh python scripts/run_template_reconstruction_study.py \
    --stage match
scripts/platforms/iapetus/run.sh python scripts/run_template_reconstruction_study.py \
    --stage relax --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2
scripts/platforms/iapetus/run.sh python scripts/run_template_reconstruction_study.py \
    --stage score

# experiment 2: protocol run e9ywwsie, extended rather than repeated.
cp generated/e9ywwsie/protocol/{screen.json,wyckoff_genes.json.gz,manifest.json,\
pyxtal.csv,pyxtal.extxyz,relaxations.csv} generated/e9ywwsie/protocol_template/
scripts/platforms/iapetus/run.sh wyformer-protocol \
    generated/e9ywwsie/protocol_template/wyckoff_genes.json.gz \
    --output-dir generated/e9ywwsie/protocol_template --stage template
scripts/platforms/iapetus/run.sh wyformer-protocol \
    generated/e9ywwsie/protocol_template/wyckoff_genes.json.gz \
    --output-dir generated/e9ywwsie/protocol_template --stage relax \
    --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 --fmax 0.05 --resume
# `arms` splits the extended run into random1 / random / template_first without
# relaxing anything; `score` runs the protocol's own score stage on all four, so
# they come from one revision of it; `table` assembles the comparison.
scripts/platforms/iapetus/run.sh python scripts/analyse_template_protocol.py arms \
    --run-dir generated/e9ywwsie/protocol_template
scripts/platforms/iapetus/run.sh python scripts/analyse_template_protocol.py score \
    --run-dir generated/e9ywwsie/protocol_template
scripts/platforms/iapetus/run.sh python scripts/analyse_template_protocol.py table \
    --run-dir generated/e9ywwsie/protocol_template \
    --baseline generated/e9ywwsie/protocol   # optional: logs the published funnel

# the ceiling: scores each of the 3258 trials rather than each gene (~15 min),
# then re-reads per_trial_scores.csv on any later call
scripts/platforms/iapetus/run.sh python scripts/analyse_template_protocol.py ceiling \
    --run-dir generated/e9ywwsie/protocol_template
```

The relaxation stages are the only expensive ones: 386 starts (21 min) for
experiment 1, 858 (35 min) for experiment 2, on two K20c cards and a GTX 750 Ti.
No random draw is recomputed in either.

## See also

- [Proposed fixes to the gene → structure step](proposed_pyxtal_generation_fixes.md)
  — the other things wrong with the sampler, all of which make the *guessing*
  cheaper rather than removing it
- [What else should WyFormer learn to take the guesswork out of PyXtal?](archive/pyxtal_dof_reduction_study.md)
  — where this method comes from, and the oracle bounds it is measured against
- [The de novo ranking protocol](de_novo_ranking_protocol.md) — the stage this
  plugs into
