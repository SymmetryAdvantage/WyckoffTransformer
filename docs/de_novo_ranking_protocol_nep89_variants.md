# Two NEP89 variants of the de novo ranking protocol

[The protocol](de_novo_ranking_protocol.md) spends almost all of its wall time
in one place: the ORB relaxation of one to three PyXtal draws per gene, four
stages each. Everything before it is cheap and everything after it is cheap.
Both variants here attack that stage with the same lever — a potential that is
two to three orders of magnitude cheaper per force call — and differ in what
they spend the savings on.

**Neither is a default.** `--stage all` still runs screen → generate → relax →
score with a single ORB relaxation per trial and no pre-relaxation, and every
flag below is off unless it is passed. That is deliberate: the trial schedule,
the stage design and the reported numbers all rest on measurements taken with
the published arm, and a variant that changed the default would invalidate them
silently.

## NEP89, and what happens to the elements it does not cover

[NEP89](https://arxiv.org/abs/2504.21286) is a neuroevolution potential fitted
across 89 elements, distributed as a plain-text NEP file in the GPUMD 4.0
release and run here through `calorine`'s CPU implementation. Measured on this
host, one force call on a 16-atom cell costs **0.8 ms** on a single thread,
against tens to hundreds of ms for ORB. That ratio is the whole argument for
both variants.

The 89 elements are H through Bi plus Ac, Th, Pa, U, Np and Pu. Within that
range it omits **Po, At, Rn, Fr and Ra**, and it has nothing above Pu. The
element list is read from the model file's own header
(`wyckoff_transformer.cryspr.nep89.nep89_elements`) rather than hard-coded, so a
fine-tuned model with a different set reports its own.

A structure containing an omitted element gets a **species-aware Lennard-Jones
potential** for the whole cell. Two things about that:

- *For the whole cell, not for the missing atoms.* NEP is a many-body potential
  over a neighbourhood; there is no defined way to evaluate it on part of a
  structure. The fallback is per structure.
- *It is a geometry regulariser, not a model.* Each pair's minimum sits at the
  sum of the two covalent radii — the same contact criterion PyXtal's
  `Tol_matrix(prototype="atomic")` uses for the generation floor — so it pushes
  overlapping atoms apart and pulls a loose PyXtal cell towards contact. Its
  energies mean nothing. ASE's own `LennardJones` takes a single sigma for every
  pair, which on a multi-element crystal either lets the small species overlap
  or holds the large one apart; this one mixes per pair.

Forces and stress are analytic and verified against finite differences in
`cryspr/tests/test_nep89.py` — the fallback relaxes a cell, so a wrong sign
there would not raise, it would just walk structures somewhere.

### How well NEP89's energies track ORB's

NEP89 computes total energy, forces and stress like any ASE calculator, and the
pre-screen's ranking is nothing but its `energy_per_atom`.  So the question of
how far those energies can be trusted is the question of whether variant 2's
selection means anything.  Measured with ORB single points on 2500 NEP89-relaxed
oracle structures over 705 genes -- *same geometry*, so the comparison is the
potentials and not the relaxation:

| measure | value |
|---|---|
| pooled E/atom | Pearson 0.963, Spearman 0.980 |
| offset NEP89 - ORB | -0.470 eV/atom, sd **0.497** |
| composition-controlled (per-element reference removed from each, 77 elements) | Pearson **0.758**, Spearman 0.870 |
| within-gene Spearman | mean 0.508, **median 0.771**, positive in **80.7%** of genes |
| NEP89's argmin = ORB's argmin | **50.3%** against a 24.6% chance rate |
| regret from taking NEP89's pick | median **0.0**, p90 42.2, mean 79.6 meV/atom |
| within-gene ORB spread, for scale | median 232 meV/atom |

Three things follow, and they are the whole design.

**The pooled r = 0.96 is a mirage.** Pooling structures of different chemistry
measures whether both potentials know that Ta is more bound than Na. Remove a
per-element reference energy from each and agreement drops to r = 0.76. Any
claim that NEP89 tracks ORB has to be composition-controlled or it is measuring
the periodic table.

**Nothing absolute can be built on these numbers.** Half an eV/atom of scatter
around a -0.47 eV/atom offset, against the ~0.023 eV/atom effect sizes this
protocol is built to resolve. Hence the hull restriction on `--mlip`, and hence
NEP89's energies reaching nothing but a within-gene ordering.

**Within one gene the signal is real, and that is the only comparison the
pre-screen makes.** Median rho = 0.771, positive in four genes out of five, and
twice the chance rate at identifying the start ORB will like best -- with a
median regret of exactly zero, meaning NEP89 usually picks a start as good as
the best on offer. The tail is heavy (mean regret 79.6 meV/atom against a p90 of
42.2), so it does sometimes pick badly. The arm does not need it to be accurate:
it needs the ordering to beat a coin, because selecting 3 of 27 candidates on a
noisy signal beats drawing 3 blind. That, plus the pre-relaxed geometry ORB
starts from, is where the +8.7 points come from -- not from NEP89 being right.

Note that the same measurement made against ORB's *fully relaxed* energy rather
than a single point looks worse (within-gene rho median 0.50, argmin hit 55.4%
over 1759 trials). That comparison charges NEP89 for ORB's own basin-hopping
during relaxation, so it is the wrong one for judging the potential -- but it is
the right one for judging the *arm*, and it is the reason the arm's gain is
smaller than the within-gene correlation alone would suggest.

### NEP89 energies never reach the funnel

`--mlip` stays restricted to potentials LeMat-Bulk publishes a convex hull for,
and NEP89 is not one. `e_above_hull` is only meaningful when the structure's
energy and the hull come from the same model
([`hull_mlips.py`](../src/wyckoff_transformer/evaluation/hull_mlips.py)), and
NEP89's training data mixes functionals with per-dataset energy shifts fitted
during training. So NEP89 is resolved through a *separate* registry
(`cryspr.mlips.build_prerelax_calculator`), which is not hull-restricted
precisely because nothing it computes is reported. Its only outputs that survive
are a geometry and, in variant 2, an ordering within one gene.

## Variant 1 — two-stage NEP89 → ORB

Everything is the same; each trial is relaxed twice.

```bash
uv run wyformer-protocol genes.json.gz --output-dir run/ \
    --stage relax --prerelax-mlip nep89 --devices cuda:0
```

The cheap potential runs the **symmetry-constrained** schedule first (fix-cell
warm-up, then symmetric cell + positions) into the trial's `prerelax/`
sub-directory, and the usual four-stage ORB relaxation starts from its output
rather than from the raw draw.

Why the pre-relaxation stops at the symmetric stages: the unconstrained stage
and the rattle are the two that may *leave* the space group PyXtal imposed, and
letting a potential that is not the one being scored make that decision would
mean ORB arrives at a structure whose symmetric stages are already spent. The
pre-relaxation moves the draw to a symmetric minimum; every question about
whether to stay there is answered on the scoring potential.
`--prerelax-fmax` (0.1 eV/Å, looser than the protocol's 0.05) is set for the
same reason: converging tightly on the wrong potential reaches the wrong
stationary point more precisely.

What it can buy, and what it can cost:

- **Buy.** The expensive stages start closer to a minimum, so ORB takes fewer
  BFGS steps. If the two potentials agree about where the minimum is, the answer
  is unchanged and the cost falls.
- **Cost.** If they disagree, ORB starts in the *wrong* basin — one NEP89
  prefers — and the rattle is the only stage that can leave it.

### NEP89 inflates a loose cell in about a quarter of draws

That cost is not hypothetical, and it has a specific shape. A PyXtal draw is
deliberately loose — tolerance factor 1.3, median 1.68× the target volume — and
the protocol relies on the variable-cell relaxation *contracting* it. NEP89
mostly does; sometimes it does the opposite. Measured over 300 randomly chosen
draws of the oracle cohort, pre-relaxed with the symmetric schedule at
`fmax = 0.1`:

| volume ratio | |
|---|---|
| median | 0.856 (contraction, as intended) |
| 25th / 75th percentile | 0.747 / 1.032 |
| 95th / 99th percentile | 1.492 / 2.183 |
| expanded at all | **28.0%** |
| expanded by >20% | 13.3% |
| expanded by >50% | 5.0% |
| space group preserved | 99.0% |

The fixed-symmetry claim holds — 99% of pre-relaxations keep spglib's verdict —
but the volume does not. And an inflated cell is the worst kind of bad start for
this protocol specifically: ORB's first two stages are symmetry-constrained, and
gradient descent under a symmetry constraint cannot leave a symmetric stationary
point, so the inflation survives to the rattle. Seen on one Ga₄Ta₆ draw during
development: 28.5 → 42.3 Å³/atom under NEP89, ORB then settled at 44.6, and only
the rattle recovered 17.2 — against the single-stage arm's 16.8 from the same
gene.

Every trial therefore writes a `prerelax.json` next to its stage CIFs, recording
the volume before and after, the ratio, and spglib's space group on both sides.
That is the same shape of evidence as `rattle.json` and for the same reason: a
pool worker configures no logging, and what the cheap potential did to a cell is
not recoverable from the scoring relaxation's output.

`--prerelax-max-expansion 1.0` discards a pre-relaxation that grew the cell and
hands ORB the raw draw instead. It is **off by default**, because "everything is
the same but the relaxations are two-stage" is the variant as specified and a
guard is a change to it — so the study runs `two_stage` and
`two_stage_guarded` as separate arms and reports both, rather than deciding
by argument.

## Variant 2 — wide-then-narrow

Draw ten times as many starts, relax them all on NEP89 under fixed symmetry,
throw away the ones that landed on the same structure, and give ORB the same
number of starts it always had — the best survivors rather than a blind draw.

```bash
wyformer-protocol genes.json.gz --output-dir run/ --stage screen
wyformer-protocol genes.json.gz --output-dir run/ --stage generate \
    --trial-multiplier 10 --pyxtal-cores 16
wyformer-protocol genes.json.gz --output-dir run/ --stage prescreen --cores 16
wyformer-protocol genes.json.gz --output-dir run/ --stage relax \
    --relax-from prescreen --devices cuda:0
wyformer-protocol genes.json.gz --output-dir run/ --stage score
```

**The ORB budget is unchanged.** `--trial-multiplier` scales the *draw* budget;
the pre-screen selects the schedule's unmultiplied number back down. A gene with
0 positional DoF still gets one ORB relaxation and one with 6 still gets three —
the difference is that the three are chosen from thirty rather than drawn blind.
The multiplier multiplies the schedule rather than replacing it so that the
extra starts stay proportional to the free coordinates that make a start worth
repeating in the first place.

**Fixed symmetry, again for the selection's sake.** The pre-screen relaxes with
`fix_symmetry=True, release_symmetry=False, rattle=False`. What it is choosing
between is *which draw of this gene* to spend ORB on, and a draw that broke its
own space group on the cheap potential is no longer a draw of that gene.
`prescreen.csv` records spglib's verdict on each pre-relaxed cell, so that is a
measured claim rather than an assumption.

**Deduplication.** Ten draws of one gene frequently relax into one structure,
and three ORB relaxations of one structure is three times the cost for one
answer. Duplicates are found with `StructureMatcher` at pymatgen's defaults —
the same matcher at the same tolerances the protocol's uniqueness and novelty
filters use, so a pair the pre-screen calls distinct is one the funnel would
also call distinct. Candidates are visited in ascending energy, so a duplicate
group's representative is its lowest-energy member. A `--prescreen-energy-tol`
gate (5 meV/atom) skips the matcher for pairs whose energies are too far apart
to be the same minimum on the same potential; the error direction is the safe
one, since too tight a gate keeps a duplicate — one extra relaxation — where too
loose a one would merge two genuinely different structures.

**What it risks** is choosing on the wrong potential. Every claim variant 2
makes rests on NEP89 ordering a gene's basins the way ORB would. Where it does
not, the arm spends its ORB budget on confidently chosen bad starts, which is
strictly worse than choosing at random. That is the question the oracle cohort
is there to answer.

### Files

| file | contents |
|---|---|
| `prescreen.csv` | per draw: status, NEP89 energy, spglib space group, `backend` (`nep89` or `lj`), seconds |
| `prescreen_all.extxyz` | every pre-relaxed draw, tagged with gene and trial |
| `prescreen_selection.csv` | per candidate: `selected` / `duplicate` / `rejected`, `duplicate_of`, `n_distinct`, `budget` |
| `prescreen.extxyz` | the selected draws only — what `--relax-from prescreen` reads |

The first two are per trial and resumable; the last two are a deterministic
function of them and are rewritten in full whenever the stage runs.
`structures.csv` gains `n_prescreened` and `prescreen_seconds` per gene, without
which a wide run's cost is indistinguishable from a baseline one — `relax` sees
the same number of trials either way.

`backend` is the only per-trial record that a gene's chemistry fell outside
NEP89. A gene pre-screened entirely through the Lennard-Jones fallback was
effectively selected at random, and `manifest.json`'s `prescreen_by_backend`
says how many those were.

## Variant 3 — NEP89 first, ORB only to refine

Run the protocol's *own* four-stage schedule on NEP89 — symmetric stages,
unconstrained stage, rattle — for every trial the schedule allots, then hand the
scoring potential the single lowest-energy winner.

```bash
wyformer-protocol genes.json.gz --output-dir run/ --stage generate --pyxtal-cores 16
wyformer-protocol genes.json.gz --output-dir run/ --stage prescreen --cores 16     --prescreen-release-symmetry --prescreen-rattle --prescreen-select 1
wyformer-protocol genes.json.gz --output-dir run/ --stage relax     --relax-from prescreen --devices cuda:0
```

It is the same `prescreen` stage as variant 2 with its schedule opened up, which
is the honest way to build it: the two arms differ in *settings*, not in code,
so a difference between them cannot be an implementation difference.

**One ORB relaxation per gene against the baseline's 2.4.** That is the whole
appeal, and it is also the whole risk, because the arm bets everything on
NEP89's ordering of a gene's own trials. The measurement above puts that
ordering at median within-gene Spearman 0.77 and twice the chance rate at
identifying ORB's best, with a heavy tail — so the bet is favourable on average
and occasionally lost badly. Note also that this arm lets NEP89 answer the
*symmetry* question: with `--prescreen-rattle` the winner may already be off the
gene's orbits before ORB sees it, which is exactly what variants 1 and 2 refuse
to allow and what this arm exists to price.

## Variant 4 — symmetry-constrained basin hopping

Variant 2 searches by drawing more. This searches by *walking*.

```bash
wyformer-protocol genes.json.gz --output-dir run/ --stage basinhop --cores 16
wyformer-protocol genes.json.gz --output-dir run/ --stage relax     --relax-from basinhop --devices cuda:0
```

From each of the schedule's usual draws, `basinhop` alternates a finite
perturbation with a local NEP89 relaxation and accepts on a Metropolis test, so
the search moves between *adjacent* minima instead of restarting from scratch
every time. Every distinct minimum it visits — accepted or rejected, because the
acceptance test governs where the walk goes and not what is worth relaxing — is
pooled across the gene's walks, deduplicated with the same `StructureMatcher`,
and narrowed to the schedule's usual count. So the ORB budget is again
unchanged, and the arm differs from variant 2 only in how the candidate pool was
built. That makes the two directly comparable: independent draws sample the
whole space badly, a walk samples a neighbourhood well.

### The perturbation stays inside the space group

`symmetric_perturb` works by exploiting the trap that
[`perturb`](../src/wyckoff_transformer/cryspr/relaxer.py) documents.
`Atoms.set_positions` enforces the attached constraints, so a `FixSymmetry`
built from the *unperturbed* structure projects any displacement onto the
symmetric subspace — it "symmetrises the rattle away", which is a bug for stage
4 and is precisely the move a symmetry-constrained search needs. A random
Cartesian rattle therefore becomes a random step along the gene's free Wyckoff
coordinates, and a random strain a step in the lattice parameters the group
allows. Nothing has to enumerate either set.

Two consequences, both measured:

- **The step has to be much larger than the rattle's.** The projection keeps
  about 68% of the drawn magnitude, and the rattle's 0.05 Å was calibrated to
  *break* symmetry rather than to cross a barrier. Over five high-DoF oracle
  draws at 15 hops: 0.05 Å finds 2.0 "minima" spanning 1 meV/atom — one basin,
  found repeatedly — while 0.15 Å and above find 2.4–2.6 spanning 57 meV/atom,
  and the count saturates. The default is 0.3 Å.
- **A gene with zero positional DoF has nothing to walk through.** The projected
  displacement vanishes and only the cell can move, which stages 1 and 2 already
  relax. `basinhop.csv` records spglib's verdict and an `n_symmetry_lost` count
  that is zero by construction, so the claim is checked every run rather than
  asserted once.

## Reporting before and after the rattle

The rattle stage is on by default because it lowers ORB's energy — it did so in
33.1% of trials in the reconstruction study, by a median of 186 meV/atom. It
also does two things the energy cannot show, and the protocol now reports both.

**It discards the prediction.** A rattle is a finite symmetry-breaking
perturbation, so the structure it leaves need not sit on the Wyckoff orbits
WyFormer predicted — and for a Wyckoff generative model those orbits *are* the
prediction. A gene whose pre-rattle structure still re-fingerprints to its own
gene and whose kept structure does not has had its prior traded for
millielectronvolts.

**It can turn a novel structure into a known one.** Relaxing away from a
symmetric stationary point can land on a LeMat-Bulk entry the unrattled
structure was distinct from, which converts a MetaSUN hit into nothing at all.

So every trial now writes **two** structures — `*_kept.cif`, which is what the
protocol scores, and `*_prerattle.cif`, the structure the rattle stage was
handed — and the score stage computes validity, uniqueness, novelty and
`e_above_hull` on each. `funnel.json` carries the kept readout under its
existing keys and the second under a `prerattle_` prefix, plus an explicit
account of the crossings:

| key | meaning |
|---|---|
| `prerattle_metasun_per_sampled_gene` | MetaSUN if the rattle had not run |
| `rattle_moved_off_gene` | genes whose pre-rattle structure re-fingerprints to its own gene and whose kept structure does not |
| `rattle_novel_became_known` / `rattle_known_became_novel` | novelty crossings under the rattle |
| `rattle_metasun_lost` / `rattle_metasun_gained` | MetaSUN hits the rattle destroyed / created |
| `rattle_metastable_lost` / `rattle_metastable_gained` | the same at the metastability threshold |
| `rattle_lowered_energy` | genes where it won at all |

Both directions are counted, not just the loss: the effect runs both ways and
reporting only the damage would overstate it. Crossings are counted over genes
that produced a *unique* structure both ways, so a gene the rattle made invalid
is not charged to novelty.

**This changes no default.** The kept structure is still the rattled one when it
wins, `metasun_per_sampled_gene` still means what it always meant, and the
pre-rattle readout costs a second pass of the matcher and the hull rather than a
second relaxation. `--no-prerattle-metrics` turns it off. The two readouts share
one novelty reference, built over the union of all four fingerprint sets, so the
streaming pass over the 1 GB CIF export — the score stage's dominant cost —
still happens once.

A run relaxed before the pre-rattle CIFs existed has none, and every
`prerattle_*` and `rattle_*` key is reported as `null` rather than guessed.

### What the rattle actually costs and buys

Measured on the published protocol over 1000 `upi73i4k` genes (2411 trials):

| | pre-rattle | kept |
|---|---|---|
| novel structures | 633 | 652 |
| metastable | 359 | **476** |
| **MetaSUN per sampled gene** | **0.152** | **0.264** |
| SUN | 0.004 | 0.010 |
| relaxed fingerprint != the gene's | **115 (11.5%)** | **506 (50.6%)** |

| what the rattle changed | genes |
|---|---|
| lowered the energy at all | 458 |
| **moved the structure off its gene's orbits** | **392** |
| novel -> known | 8 |
| known -> novel | 27 |
| **MetaSUN lost** | **3** |
| **MetaSUN gained** | **115** |
| metastable lost | **0** |

**The novelty-collapse worry is real but small.** The rattle turned 8 novel
structures known and destroyed 3 MetaSUN hits -- while creating 115. It never
loses metastability, which is structural rather than lucky: the stage is
accepted only if it lowers the energy by more than 1 meV/atom, so it is a
ratchet.

**The prior-destruction worry is large.** 392 of 1000 genes had a structure that
re-fingerprinted to its own gene before the rattle and did not after, and the
cohort-wide share of kept structures whose fingerprint differs from the
prediction goes from 11.5% to **50.6%**. So the rattle buys +11.2 points of
MetaSUN (+74% relative) and pays with the Wyckoff prior on about 40% of the
cohort. Half of the protocol's headline MetaSUN successes are structures that no
longer sit on the orbits WyFormer predicted.

### The physics: the rattle is a cheap phonon-stability check

The two readouts are *not* symmetric alternatives, because what the rattle
removes is mostly not a valid structure. Over the same 2411 trials:

| | |
|---|---|
| rattle accepted | 1139 (47.2%) |
| energy drop when accepted | median **190.6** meV/atom; 66.9% above 100; 17.3% above 500 |
| space group **lowered** when accepted | **96.3%** |
| space group *raised* | **0.0%** |
| space group lowered when rejected | 0.9% |
| \|dE\| when rejected | median **0.169** meV/atom |

That is the signature of a soft-mode instability, not of fine-tuning. The
pre-rattle structure sat at a symmetry-constrained *stationary point* that is a
saddle, and an arbitrary symmetry-breaking kick decays it down a
symmetry-lowering distortion by ~100-200 meV/atom -- about 2200 K in thermal
units. A structure unstable to an infinitesimal symmetry-breaking displacement
has imaginary phonon modes and is not a metastable phase at 0 K. The rejected
trials confirm the mechanism from the other side: they move by 0.17 meV/atom,
i.e. they were already at genuine minima.

So **the kept readout is the better "is this a real material" number** and the
pre-rattle one over-counts by crediting structures that would spontaneously
distort; while the pre-rattle readout is the right one for "is WyFormer's
Wyckoff prediction correct", because for those 392 genes the credit belongs to
the rattle and ORB rather than to the model.

Two exceptions matter, and they are not technicalities:

- **Entropic and anharmonic stabilisation.** Cubic perovskites, bcc Ti/Zr/Hf,
  cubic ZrO2 and δ-Pu are all dynamically unstable at 0 K in the harmonic sense
  and experimentally real at temperature. For those the high-symmetry structure
  is the one in the database, and the rattle destroys the right answer. Our
  relaxation is 0 K with no entropy and no zero-point motion.
- **Dynamic disorder.** Diffraction reports a time- and space-averaged
  structure; where local distortions average to higher symmetry the correct
  crystallographic answer *is* the high-symmetry cell, which is exactly what a
  Wyckoff gene describes.

Both biases concentrate in **high-symmetry, low-DoF** structures -- precisely
where a Wyckoff-by-design model is strongest -- so the exception is not random
with respect to what WyFormer produces.

A cheap test would separate the two cases without phonons: rattle each
pre-rattle structure ~10 times with different seeds. A structure with an
imaginary mode decays under *every* perturbation; a true minimum with a deeper
basin nearby escapes only along some directions. At NEP89 speed that is a few
CPU-hours for the cohort, and it would let the protocol report "dynamically
unstable" separately from "metastable but not lowest".

## Evaluating the variants

`scripts/run_nep89_protocol_variants.py` runs all three arms over two cohorts.

**The oracle cohort** is the 750 DoF-stratified LeMat-Bulk structures prepared
by `scripts/oracle_reconstruction.py --stage prepare`, each with an ORB-relaxed
reference. Every gene has a known answer, so the readout is reconstruction —
does the arm's kept structure match its reference under `StructureMatcher` —
rather than a rate over an unknown population. That matters for power: the arms
see the same genes, so the comparison is **paired**, and only the discordant
pairs carry information. An arm that recovers 40 genes the baseline missed while
losing 10 it found is a real improvement at a cohort size where the two overall
rates would not separate.

```bash
uv run python scripts/run_nep89_protocol_variants.py --stage genes
uv run python scripts/run_nep89_protocol_variants.py --stage run --cores 16
uv run python scripts/run_nep89_protocol_variants.py --stage score --cores 16
uv run python scripts/run_nep89_protocol_variants.py --stage report
```

**The W&B cohort** is a WyFormer run's own genes, where there is no known answer
and the readout is the funnel itself — MetaSUN per sampled gene, against the
CPU-seconds each arm cost. `wyformer-protocol-wandb` accepts the same flags and
`--stages` now takes the optional stages too:

```bash
# baseline
uv run wyformer-protocol-wandb <run-id> --output-dir generated/<run-id>/nep89/baseline \
    --devices cuda:0 --workers-per-device 2 --no-upload
# variant 1
uv run wyformer-protocol-wandb <run-id> --output-dir generated/<run-id>/nep89/two_stage \
    --skip-generate --prerelax-mlip nep89 --devices cuda:0 --workers-per-device 2 --no-upload
# variant 2
uv run wyformer-protocol-wandb <run-id> --output-dir generated/<run-id>/nep89/wide \
    --skip-generate --trial-multiplier 10 --relax-from prescreen \
    --stages screen,generate,prescreen,relax,score --cores 16 --no-upload
```

Copy the baseline's `wyckoff_genes.json.gz` into the other two output
directories before `--skip-generate`: the arms must score the *same* cohort, and
a freshly generated one would confound the comparison with sampling noise.

## What the study found

Full tables in [RESULTS.md](../generated/nep89_protocol_variants/RESULTS.md) (not
in git; regenerate with `--stage report`). The oracle cohort, 750 DoF-stratified
LeMat-Bulk genes, every arm relaxing the *same* PyXtal draws except `wide`:

| arm | reconstructed | vs baseline (won/lost) | McNemar *p* | ORB s/gene | pre-screen CPU-s/gene |
|---|---|---|---|---|---|
| baseline | 55.2% (414/750) | — | — | 10.3 | — |
| two_stage | 57.3% (430) | 36 / 20 | 0.044 | **9.4** | — |
| two_stage_guarded | 57.1% (428) | 28 / 14 | 0.044 | 10.4 | — |
| `wide` | **63.9% (479)** | **97 / 32** | **8.5e-9** | **6.8** | 26.2 |

**Variant 2 is the clear winner, and it is cheaper on the GPU.** +8.7 points of
reconstruction at *p* = 8.5e-9, and it spends 6.8 ORB-seconds per gene against
the baseline's 10.3 — a 34% saving — because the starts it hands ORB are already
pre-relaxed. What it costs is 26.2 CPU-seconds per gene of pre-screening plus the
extra draws. If the GPU is the scarce resource, which on a contended box it is,
the arm is a saving twice over; if total CPU is what is counted, it is roughly
three times the compute for the +8.7 points.

The gain lands exactly where the mechanism says it must:

| positional DoF | baseline | `wide` | distinct minima per gene (of 10–30 draws) |
|---|---|---|---|
| 0 | 90.7% | 92.7% | 1.25 |
| 1–2 | 81.3% | 89.3% | 3.06 |
| 3–5 | 58.0% | **72.0%** | 8.33 |
| 6–10 | 33.3% | 41.3% | 18.67 |
| >10 | 12.7% | **24.0%** | 27.17 |

At zero positional DoF ten draws collapse to 1.25 distinct structures, so the
widening buys 2 points and cannot buy more — the same argument that gives that
bin one trial in the default schedule. Above 3 DoF most draws are distinct
minima and the selection has something to choose between, which is where the
double-digit gains are. The distinct-minima column replicates to within a few
percent on the W&B cohort (1.09 / 2.82 / 8.34 / 18.30 / 27.44 over 24,114
draws), so it is a property of the gene's degrees of freedom rather than of
either cohort. **The obvious follow-up is a DoF-dependent multiplier**: the
budget spent below 2 DoF is provably wasted.

### The W&B cohort agrees, and the pairing is what makes it visible

1000 genes from `upi73i4k` at `energy_above_hull=0`, every arm scoring the same
cohort on the same draws:

| arm | novel | metastable | **MetaSUN** | McNemar *p* | ORB s/gene | pre-screen CPU-s/gene |
|---|---|---|---|---|---|---|
| baseline | 0.653 | 0.476 | **0.266** | — | 13.2 | — |
| two_stage | — | — | 0.271 | 0.56 | 12.3 | — |
| `wide` | 0.649 | **0.512** | **0.298** | **0.0016** | **8.7** | 27.4 |

`wide` gains 3.2 points of MetaSUN, **+12% relative**, at *p* = 0.0016 over 98
discordant genes — and it gains them at 66% of the baseline's ORB time. The
`metastable` rate carries all of it (0.476 -> 0.512, *p* = 1.3e-4) while novelty
is flat (0.653 -> 0.649), so what the arm buys is *better structures*, not more
novel ones, which is what a better choice of starting geometry should buy.

**+12% relative is below what this cohort size can resolve unpaired.** The
[power analysis](archive/de_novo_ranking_protocol_rationale.md) sizes 1000 genes
for +20% relative at alpha = 0.05 and 80% power, so an unpaired comparison of two
independently drawn cohorts would have missed this effect. Sharing the draws
between arms is what recovers it, and it is the reason `SHARED_DRAWS` exists.

MetaSUN per DoF bin tells the same story as the oracle:

| positional DoF | baseline | `wide` | n |
|---|---|---|---|
| 0 | 0.104 | 0.098 | 183 |
| 1-2 | 0.295 | 0.309 | 217 |
| 3-5 | 0.309 | 0.336 | 301 |
| 6-10 | 0.320 | 0.366 | 172 |
| >10 | 0.276 | **0.386** | 127 |

Flat to slightly negative where there is nothing to search, +11 points (+40%
relative) where there is.

**Variant 1 is real but small, and its value is the cost saving rather than the
quality.** On the oracle's reconstruction readout it wins: +2.1 points, *p* =
0.044 on 56 discordant pairs. On MetaSUN it does not: 0.266 -> 0.271, 26 won
against 21 lost, *p* = 0.56. Both readouts agree on the cost, 7-9% less ORB time
for free. So it is worth switching on for the GPU saving, and not worth
attributing a quality gain to.

**The expansion guard is a net loss — do not turn it on.** NEP89 inflates a
loose PyXtal cell in 28.8% of draws (median ratio 0.860, 14.1% by >20%, 5.6% by
>50%; space group preserved in 99.7%), and the guard fires correctly on all of
them. It does cut the regression tail — 98 genes worse than baseline by >1
meV/atom becomes 74 — but it cuts the improvement tail harder, 152 to 115, and
gives back the whole cost saving (10.4 ORB-s/gene against 9.4). Inflating the
cell is therefore *not* purely harmful: for a good number of genes ORB's rattle
escapes the inflated basin and lands lower than the baseline reached. The
mechanism argued the other way and the measurement overruled it, which is why
the guard shipped as an arm rather than as a default.

**Cost scales linearly in the draw count.** 10x the draws costs 8x the PyXtal
CPU-hours on oracle genes (1.03 vs 0.13) and 10.6x on generated ones (8.25 vs
0.78). The absolute per-draw cost differs by 5x between the two cohorts — 1.23 s
for a WyFormer gene against 0.21 s for a LeMat-Bulk one, because generated genes
include some PyXtal rejection-samples to the 300 s timeout — but nothing about
the widening is superlinear.

## Installing NEP89

`calorine` builds a pybind11 extension from an sdist — there are no wheels — so
it needs a C++ compiler, and it declares `numpy<=2.3` while this project runs on
2.5. That pin is spurious: `_nepy` touches no numpy C API, and CPUNEP is
verified working here at numpy 2.5.3 (energies, forces and stress all check
out). But `uv sync --extra nep` would honour it and **downgrade numpy**, so on a
host where that matters install it additively instead:

```bash
uv pip install --no-deps calorine
```

Every runtime dependency it then needs — ase, numpy, pandas, scikit-learn,
matplotlib — is already in the project. See
[the zeus notes](platforms/zeus/environment.md#calorine-and-nep89).

## Where to go next

What limits reconstruction, what the oracle arms say each missing piece is worth,
and four proposals with their costs — including why the basin-hopping null result
constrains two of them — are in
[Where the reconstruction ceiling comes from](de_novo_search_ideas.md).

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md) — the arm these vary
- [The rationale notes](archive/de_novo_ranking_protocol_rationale.md) — why every default is what it is
- [Template-matched starts](cryspr_template_starts.md) — the other optional source of starting structures
- [Search ideas and the reconstruction ceiling](de_novo_search_ideas.md) — what to try next, and what the oracle bounds allow
- [The PyXtal tolerance sweep](pyxtal_tolerance_sweep.md) — whether a permissive distance floor reconstructs more (it does not; it is cheaper)
- [CrySPR trial and stage spread](archive/cryspr_trial_and_stage_spread.md) — where the trial and stage numbers come from
