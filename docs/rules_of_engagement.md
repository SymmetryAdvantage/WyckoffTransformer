# Rules of engagement: WyFormer's inference modes

> **STATUS (2026-09-22, commit `9145354`).** The composition layer, the four
> modes and the per-sampled-gene accounting are in `wyckoff_transformer.roe`.
> The tensor-space novelty and uniqueness checkers are in
> `evaluation/gene_hash.py`, are the default, and are measured below.
> **broadside, fire-discipline and fire-control have been run** against one
> another at a fixed budget of 1000 reconstructions; **torpedo-run has not**, so
> what it buys is still a hypothesis.

Generating a (M)SUN structure is an attack on the convex hull: a candidate that
lands below it does not merely pass a threshold, it redraws the hull beneath
itself. The four inference modes differ in how much is checked before the
expensive shot is fired, so they are named for rules of engagement.

**Naming rule.** Where a quantity already has a name -- `e_hull`, formation
energy, the convex hull, novelty, uniqueness, validity, metastable, stable, SUN,
MetaSUN, relaxation, gene -- it keeps it, in the docs, the code and the columns.
The naval vocabulary names the one thing that had no name: the modes. A reader
should never have to decode a metaphor to find out what was measured.

## The four modes

| # | Mode | Pipeline | What it adds |
|---|---|---|---|
| 1 | **broadside** | generate → reconstruct | nothing; the baseline |
| 2 | **fire-discipline** | generate → screen → reconstruct | uniqueness and novelty, by fingerprint lookup |
| 3 | **fire-control** | generate → screen → predicted `e_hull` → reconstruct | a predicted formation energy compared with the hull |
| 4 | **torpedo-run** | sample chemical system → generate conditioned on it → predicted `e_hull` → screen → reconstruct | a chosen target |

Ordering is part of a mode, not an implementation detail. `fire-control`
screens before it ranks; `torpedo-run` ranks before it screens. Two modes with
the same components in the other order are different experiments, so the order
is in the mode's definition (`roe/plan.py`) and in its manifest.

`wyformer-roe list -v` prints the same table with each mode's rationale.

**Formal validity is not one of the filters.** A gene naming a Wyckoff letter
its space group does not have is not a cheap reconstruction that a careless mode
would attempt and a careful one would skip -- PyXtal has nothing to place. Every
mode drops it, none is credited for it, and it is reported separately.

### Why each filter is where it is

**`fire-discipline`: the free lookup, first.** The fingerprint lookup against
the reference archive is exact, needs no model, and removes two kinds of wasted
reconstruction: the same gene twice, and a gene LeMat-Bulk already holds. On the
`e9ywwsie` pool it was worth 2.78x MetaSUN at a budget of 250
([generative novelty screen](generative_novelty_screen.md)). It is the first
thing any mode past the baseline should do.

**`fire-control`: screen before you rank.** An energy ranker left to itself
finds low-lying genes partly by finding compositions the archive already holds:
its best-scoring decile was 90% metastable and 83% already-known formulas, and
MetaSUN peaked a decile *lower*
([the DFT screen uplift](archive/e9ywwsie_dft_screen_uplift_report.md)). Removing
the overlap first and ranking the remainder is what recovered the gain, so that
is the order the mode fixes.

**`torpedo-run`: rank before you screen.** Inside a named chemical system most
genes are novel, so the screen has less to remove and less reason to run first;
and the predicted-hull filter is the cheaper of the two whenever the 4.8M
reference fingerprints are not already resident. This ordering is a judgement,
not a measurement -- it is the first thing the mode comparison should check.

## The accounting

A mode filters before the reconstruction, so the protocol's rates -- which are
per gene the protocol was handed -- answer "how good were the genes that got
through". That is not the question a mode is chosen on. `roe/report.py`
restates every rate against the genes the mode *drew*, by feeding the protocol's
own `funnel_structure_metrics` a `GeneScreen` built from the cohort's weights.
MetaSUN must mean exactly what it means in
[the ranking protocol](de_novo_ranking_protocol.md); a second implementation of
those masks would eventually mean something slightly else.

Three quantities, in `engagement.json`:

| | |
|---|---|
| `per_sampled_gene` | the funnel, restated against the drawn cohort |
| `cost.trials_charged` | reconstruction trials the mode is charged for |
| `trials_per_hit` | the first divided by the second: **the number the escalation exists to move** |

**A cohort marks, it never drops.** Filters write verdict columns and a `kept`
mask; no gene leaves the table and no index is reused. That is what keeps the
denominator fixed and the avoided cost recoverable, and a filter that tried to
re-admit a gene an earlier one removed raises rather than being absorbed.

**Duplicates are charged to `broadside` without being relaxed.** It runs no
uniqueness screen, so a faithful campaign at those rules would have reconstructed
every duplicate as though it were new. The protocol nonetheless deduplicates,
because that is what the protocol does, so the cost is recovered from the
cohort's weights instead of spent. This makes `broadside`'s **cost exact and its
yield a lower bound**: the duplicate draws would have been extra trials of the
same gene, and extra trials sometimes find a lower minimum. The bias runs
against the conclusion the other modes want, which is the right direction for it
to run.

## The first comparison: broadside, fire-discipline, fire-control

Run on iapetus on 2026-09-21/22 at commit `9145354`, by
`scripts/platforms/iapetus/run_rules_of_engagement.sh`. Backbone
`unconditional_5x_ehull01-20260915-151250` (unconditional, so no conditioning
target to choose); energy predictor `min_energy_adamw_wsd-20260912-115957`
(scalar, MSE, fitted to `gene_min_formation_energy_per_atom`, trained on
`lemat_bulk_fmax1_stress`). Both read from disk; nothing was written to W&B.

**One pool, three selections, one budget.** A pool of 10,000 genes is drawn once
and all three modes select from it, so the arms differ in what they select and in
nothing else -- no sampling noise between them. Each reconstructs exactly **1000
genes**, so the expensive half is held fixed and what separates the modes is the
selection.

Each mode consumes only as much of the pool as its budget needs, which is what
makes the generation cost visible:

| mode | selection | pool consumed |
|---|---|---|
| broadside | the first 1000 formally valid genes, duplicates included | ~1100 |
| fire-discipline | the first 1000 unique *and* novel genes | ~1700 |
| fire-control | of every unique, novel gene in the pool, the 1000 with the lowest predicted `e_hull` | 10,000 |

**Fire-control selects a budget, not a threshold, and that is a real choice.** On
a 300-gene pilot only **10 of 187** novel genes (5.3%) were predicted at or below
the hull -- the median predicted `e_hull` was 0.074 eV/atom and the 5th percentile
-0.001. Filling a budget of 1000 from a threshold at zero would take roughly
30,000 draws. Taking the 1000 lowest instead spends the fixed budget on the best
the pool holds, which is how the same lever was used in
[the generative novelty screen](generative_novelty_screen.md) ("at B=250"), and
the manifest records what predicted `e_hull` the cut actually landed at so the
selection strength is never implicit. `--energy-select threshold` is still
available and is the right setting when the question is how many genes a
generator puts below the hull rather than how to spend a budget.

Reconstruction is the de novo ranking protocol unchanged: ORB `orb_conserv_inf`,
the default trial schedule, both the fixed-symmetry and free readouts, scored
against `lemat_bulk_fmax1_stress`. Devices are the two K20c cards at two workers
each plus the GTX 750 Ti, as [the host's usage notes](platforms/iapetus/usage.md)
prescribe. The three arms run strictly in sequence: the protocol's score stage
holds the reference at ~25 GB and this host has 30 GB.

**MetaSUN is the readout, SUN is reported.** At a budget of 1000 reconstructions
per arm, SUN (`e_hull <= 0`) is not resolvable: the protocol's own power analysis
puts ~10,000 genes behind a SUN comparison and ~1,900 behind one at 0.05
([the ranking protocol](de_novo_ranking_protocol.md#how-many-genes)), and the
pool here is 1000 per arm. Differences between the modes are therefore read off
`metasun_per_sampled_gene` (`e_hull <= 0.1`), with SUN counted and reported but
not argued from.

### Results

> To be filled in when the run completes. Every number will be per sampled gene
> *and* per reconstruction trial, because the modes deliberately differ in how
> much generation they spent to fill the same budget.

## The architecture

A mode names **slots**; a run fills them. Swapping an implementation into a slot
does not change what the mode means or what it can be compared against.

| slot | what belongs in it | filled today by |
|---|---|---|
| sampler | which chemical system and space group each structure is for | `SystemPriorSampler`, `PlanFileSampler` |
| source | the genes | `WyFormerGeneSource`; `GeneFileSource` for a shared pool, handed out in order |
| `screen` | uniqueness and novelty | `NoveltyUniquenessScreen`, `python` or `tensor` backend |
| `energy` | predicted formation energy against the hull | `PredictedHullFilter`, by threshold or by budget |
| `surprisal` | the generator's own log-density | `SurprisalBandFilter` |
| reconstructor | genes → relaxed, scored structures | `CrySPRReconstructor` |

Every component in `roe/builtin.py` is an adapter. Nothing there implements a
screen, a generator, an energy model or a relaxation: the screen is
`evaluation.protocol.screen_genes`, the energy filter is
`cli.gene_screen.score_genes`, the reconstructor runs `cli.protocol`'s own
stages, and the sampler is `system_prior.SystemSpaceGroupPrior`. When a
component looks like it is doing real work, that work belongs in the module it
wraps.

Adding one means writing a class with `name`, `slot`, `provides`, `requires`,
`apply(cohort)` and `describe()`, and nothing else: the protocols in
`roe/components.py` are structural, so a component is a component by having the
methods.

An assembly is validated **before anything expensive runs** -- a missing slot, a
filter in the wrong slot, a sampler in a mode that does not aim, a chain whose
input nothing upstream produces, a chemical-system-conditioned checkpoint with
no plan to condition on. All of those raise before the first checkpoint loads.

### Running it

```bash
wyformer-roe list -v

# The baseline.
wyformer-roe run broadside --model-path runs/<run> \
    --output-dir generated/<run>/broadside --n-genes 1000 \
    --condition energy_above_hull=0.05 -- --devices cuda:0,cuda:1

# Screen, then rank on the predicted hull.
wyformer-roe run fire-control --model-path runs/<run> \
    --output-dir generated/<run>/fire-control \
    --regressor-path runs/<gene-energy-run> \
    -- --devices cuda:0,cuda:1

# Choose the targets first. Needs a chemical-system-conditioned checkpoint.
wyformer-roe run torpedo-run --model-path runs/<chemsys-run> \
    --output-dir generated/<run>/torpedo-run \
    --system-prior cache/lemat_bulk_fmax1_stress/system_prior.npz \
    --required Li --max-arity 3 --regressor-path runs/<gene-energy-run> \
    -- --devices cuda:0,cuda:1
```

Everything after a bare `--` goes to `wyformer-protocol` untouched, so the
reconstruction keeps every flag it has -- the MLIP, the devices, the trial
schedule, the timeouts -- without this CLI mirroring any of them.

`--no-reconstruct` stops at the filtered gene file, which is what a run whose
reconstruction happens on another machine wants. `wyformer-roe report <dir>`
rebuilds `engagement.json` from a directory afterwards.

To compare modes rather than run one, draw a pool once and point every arm at it
with `--genes`: each consumes only as much of it as its budget needs, so the arms
are paired and their generation costs stay separately visible.
`scripts/platforms/iapetus/run_rules_of_engagement.sh` is exactly that, for this
host.

### Two component decisions worth knowing about

**A gene whose composition the reference hull does not cover is kept, not
dropped** (`--on-missing-hull keep`, the default). There is no hull to compare
it with, and a novel composition is exactly what a discovery campaign is looking
for; dropping it would make the energy filter select against novel chemistry by
construction, which is the failure
[the generative novelty screen](generative_novelty_screen.md) measured for an
energy ranker left to itself. `drop` is available and says so in the manifest.

**`SurprisalBandFilter` keeps a band, not a tail.** Ranking on surprisal in
either direction was worse than random on `e9ywwsie`; keeping a band and ranking
the survivors on energy was the best lookup-free arm there (2.48x). It is
refused outright for a chemical-system-conditioned generator, by
`gene_likelihood.score_gene_likelihood` itself: each gene would be scored under
its own conditioning and the numbers would not be comparable across genes. So
`torpedo-run` cannot use it.

## The tensor-space screen

Built, measured and default (`--screen-backend tensor`). Gene novelty and
uniqueness are decided on a pair of 64-bit integers per gene instead of a nested
`frozenset`, in `wyckoff_transformer.evaluation.gene_hash`.

### It is exactly equivalent, not an approximation

The fingerprint is a *set of multisets*:

    (space group, { { (element, site symmetry, enumeration) x count } for each
                    equivalent enumeration })

so the key is built the same way round -- hash each variant's **sorted** multiset,
then hash the **sorted, deduplicated** list of those variant digests together
with the space group. Sorting canonicalises a multiset, sorting-and-deduplicating
canonicalises a set, and neither step assumes anything about the augmentations.

That last point is the design decision, and checking it turned up something
else. The obvious canonical form is the *minimum over the augmentation orbit*.
It needs the relabellings to form a group -- and **they do**, in all 230 space
groups; see [the augmentation audit](wyckoff_augmentation_audit.md). But the
group property turns out not to be sufficient, because the representation the
fingerprint is built from is not equivariant: it pairs each site's *original*
site-symmetry symbol with its *relabelled* enumeration index, and in 26 space
groups a relabelling changes that symbol. Min-over-orbit is therefore exactly
equivalent to the fingerprint on the other 204 groups and not on those 26.

Hashing the whole variant set needs neither property, so this key is exactly as
correct as `record_to_augmented_fingerprint` -- which, as the audit shows, is
itself not as correct as it looks.

Collision probability over the 4.8M distinct genes of `lemat_bulk_fmax1_stress`
is about 4e-26 at 128 bits. The encoding is dataset-independent by construction:
an element enters as its atomic number and a site symmetry as its own UTF-8
bytes, never as a tokeniser id, so a table built from one model's cache answers
correctly for a model trained on another.

### What it cost and what it bought

Measured on iapetus, 2026-09-21/22, commit `9145354`, against
`cache/lemat_bulk_fmax1_stress` (5,327,342 rows, all three splits):

| | fingerprint set | key table |
|---|---|---|
| distinct genes found | 4,826,004 | **4,826,004** |
| on disk | 246 MB (`gene_fingerprints.pkl.gz`) | **74 MB** (`gene_keys.npz`) |
| resident | ~17 GB | **77 MB** |
| build | ~11 min | **~7 min** (one-off, cached beside the reference) |
| load, then screen 300 genes | ~2 min | **1.2 s** |

The first row is the strongest evidence available that the two encodings agree:
they find the same number of equivalence classes in the same 5.3M rows, and a key
that merged two fingerprints would show up as a smaller table, one that split a
fingerprint as a larger one.

The win is not only the 200x on memory. At 77 MB the screen can run *inside* a
generation loop rather than after it -- which is what makes the top-up to a fixed
reconstruction budget affordable, since every round re-screens the whole
accumulated cohort.

### How it is tested

`src/wyckoff_transformer/tests/test_gene_hash.py` (24 tests, no cache needed):

- **the partition test**, which is the one that matters: over 3000 random legal
  genes, two genes share a key exactly when they share a fingerprint -- an
  identity of the induced partitions, not a sample of spot checks. A companion
  test asserts the sample actually *contains* repeats, so the merge half of the
  claim is exercised rather than assumed;
- the invariances: site order, and every spelling of an element (`Element Fe`,
  `"Fe"`, `Element("Fe")`) that the caches and a restored processor disagree on;
- the distinctions: space group, elements, which element sits on which orbit
  (checked *against whatever the fingerprint answers*, since the augmentation
  makes that case subtle), and a repeated orbit, which a set would collapse and
  a multiset must not;
- **stability across processes** under three values of `PYTHONHASHSEED`, since
  Python randomises `hash()` per process and a cached table must not move;
- the table: membership against a Python set, an empty table, queries below and
  above every key, a matching low word with a different high word, the refusal
  of a duplicated low word, deduplication, and a save/load round trip including
  the refusal of a table from another encoding version;
- uniqueness: representatives and counts identical to `screen_genes` over a
  cohort seeded with duplicates, first-occurrence ordering, and the empty cohort.

`test_gene_hash_reference.py` (marked `needs_cache`) repeats the decisive checks
against the real archive: the class-count identity above, and that a 20,000-row
sample of the reference is found in the table built from it. Both passed on
2026-09-21.

`roe/tests/test_roe.py` pins the two backends *interchangeable* at the component
level -- same kept set, same `duplicates`, same `gene_novel`, same weights, over
four different archives including the empty one -- so a mode's result cannot
depend on which one ran.

### What is still Python, and why that is fine

| stage | tensor? | |
|---|---|---|
| uniqueness within a cohort | yes | `torch.unique` on the keys |
| gene novelty (candidate set) | yes | `searchsorted` against the sorted table |
| the per-gene key | no | `hashlib` over a canonical byte encoding |
| formal validity | no | still the mappings lookup that raises |
| energy prediction | already was | `build_tokenised_prediction_tensors` |
| hull lookup, formula in the table | partly | cached per chemical system |
| **hull for a novel formula** | no | a pymatgen phase diagram per system |
| **PyXtal draw** | no | rejection sampling in Python |
| **MLIP relaxation** | no | |
| **novelty verdict** | no | `StructureMatcher` on geometry |

The per-gene key is `hashlib` rather than a vectorised 64-bit mix because a
thousand-gene cohort takes milliseconds and the reference build is a one-off: at
this scale a hand-written mix would only be one more thing that has to be right.

The last four rows are where the wall-clock is, and none of them moved. That is
the honest limit of the exercise: it makes the *filters* nearly free, which is
what lets them run in a loop, but it does not make a mode cheaper to reconstruct,
and the fingerprint remains a candidate generator for the matcher, never a
verdict ([the ranking protocol](de_novo_ranking_protocol.md#novelty-and-uniqueness-are-two-stage)).

The tokeniser turned out not to be the constraint at all, which was the open
question. Nothing in the key refers to a tokeniser, so the "all models must share
one" precondition never arose. A model-specific gather table is still the right
answer for the *other* gene stages -- `prediction.filter_supported_tokens` is
still a Python loop over rows -- and is not built here.

## Known limitations

- **Torpedo-run has not been run.** What it buys, and whether its
  rank-before-screen ordering beats fire-control's screen-before-rank, is still
  a hypothesis.
- **One backbone, one pool, one budget.** The comparison below is a single
  10,000-gene pool from one unconditional checkpoint at one reconstruction
  budget. It says what these selections do to this generator's output, not what
  they do in general, and the modes' relative order could differ for a
  conditioned backbone whose pool is already enriched.
- **`broadside`'s yield is a lower bound**, for the reason in the accounting
  section. Its cost is exact.
- **DiffCSP++ is not wired up.** The repository reads DiffCSP++ *output*
  (`evaluation/DiffCSP_to_sites.py`) but has no path that hands it genes and
  gets structures back. `DiffCSPReconstructor` exists so that adding one is a
  component rather than a change to every mode, and refuses loudly meanwhile;
  `--no-reconstruct` writes the gene file for an external run.
- **The protocol re-screens the filtered gene file.** Not wasted -- it is the
  audit, and `score` refuses to mix a screen's verdict with a novelty reference
  other than its own -- but a mode with a screen pays the reference load twice
  unless the fingerprint cache is warm.
- **The `surprisal` slot is API-only.** No built-in mode uses it and the CLI
  does not expose it; a mode that does should be defined rather than patched in.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md) -- the
  reconstruction and scoring every mode ends in, and where MetaSUN is defined
- [Generative novelty screening](generative_novelty_screen.md) -- the surprisal
  lever, and the measurement that fixes `fire-control`'s ordering
- [Composition screening](composition_screening.md) -- the energy floor the
  `energy` slot estimates, and the hull it is compared with
- [The chemical system sampler](chemical_system_sampler.md) -- what
  `torpedo-run` aims with
- [Chemical-system mode](chemical_system_mode.md) -- what a chemical-system
  conditioned checkpoint was told, and what it was not
- [Every `e_hull` in this repository](e_hull_definitions.md) -- the predicted
  `e_hull` of the `energy` slot is not the MLIP `e_above_hull` the protocol
  scores with
