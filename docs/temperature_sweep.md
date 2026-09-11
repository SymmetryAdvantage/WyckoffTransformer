# Sampling temperature and the de novo ranking protocol

What the [de novo ranking protocol](de_novo_ranking_protocol.md) reads off
`e9ywwsie` when the sampler is sharpened or flattened, and nothing else about
the model is changed.

> **Complete.** Nine temperatures screened at the gene level, five of them
> relaxed and scored end to end. MetaSUN is flat from T=0.7 to T=1.2 -- the
> novelty a hotter sampler buys is paid back almost exactly in lost stability --
> and then collapses above T≈1.2, where novelty saturates and there is nothing
> left to pay with.

## What the temperature does, and what it does not

`--temperature T` divides the logits of every *generated cascade field*
(`elements`, `site_symmetries`, `sites_enumeration`) by `T` before the
multinomial draw. `T < 1` sharpens the sampler towards its modes, `T > 1`
flattens it.

**It does not touch the space group.** The start token is drawn from the run's
saved empirical `spacegroup_distribution.json`, not from the model, so the
space-group marginal is held fixed across the sweep by construction. That makes
the sweep a clean test of the *content* of a gene -- which elements land on
which orbits, and when the sequence stops -- against a fixed backbone. Anything
that would flatten the space-group prior as well is a different knob
(`--sg-temperature` in `wyformer-system-prior`, over a different sampler).

**Before this study, `--temperature` did not exist and the argument that did
was ignored.** `WyckoffTrainer.generate_structures` accepted `temperature`,
documented it, and forwarded it only on the element-constrained branch. Plain de
novo generation -- `wyformer-generate` with no element constraints, and every
protocol cohort -- fell through to a `generate_tensors` call that took the
default of 1.0. Nothing raised; a sweep over the argument would simply have
returned the same distribution nine times.
`tests/test_generation_temperature.py` pins both halves: that the generator
applies the temperature, and that the trainer forwards it on the unconstrained
path.

## The cohort moves before any of the metrics do

One thousand genes per temperature, all from `runs/e9ywwsie` at
`energy_above_hull=0`, 1400 attempted per arm.

| | T=0.5 | T=0.6 | T=0.7 | T=0.8 | T=0.9 | **T=1.0** | T=1.1 | T=1.2 | T=1.5 |
|---|---|---|---|---|---|---|---|---|---|
| formal validity of the draw | 0.961 | 0.980 | 0.985 | 0.981 | 0.979 | **0.969** | 0.961 | 0.952 | 0.921 |
| mean orbits per gene | 9.86 | 7.96 | 5.92 | 5.42 | 5.35 | **5.21** | 5.39 | 5.19 | 5.49 |
| mean atoms per gene | 41.8 | 33.2 | 24.4 | 22.1 | 21.5 | **21.2** | 21.5 | 21.8 | 24.9 |
| mean distinct elements | 2.95 | 3.06 | 3.09 | 3.12 | 3.18 | **3.25** | 3.28 | 3.32 | 3.56 |

Two effects, and only one of them is the one people expect.

**Flattening costs formal validity, monotonically.** 0.985 at T=0.7 down to
0.921 at T=1.5: a hotter sampler puts elements on orbit sets that are not
self-consistent. This is invisible in every rate below, because the cohort is
truncated to the first 1000 *valid* genes -- which is why
`formal_gene_validity` is now recorded in `manifest.json`.

**Sharpening does not make genes shorter -- it makes a tail of them
enormous.** The medians barely move (4-5 orbits, 16 atoms at every temperature
from 0.5 to 1.5). The means at T ≤ 0.6 are carried entirely by a runaway tail:

| | orbits p50 / p90 / p99 / max | atoms p50 / p90 / p99 / max | genes ≥ 100 atoms |
|---|---|---|---|
| T=1.0 | 4 / 8 / 14 / 47 | 16 / 40 / 112 / 160 | 1.2% |
| T=0.5 | 5 / 16 / 62 / **62** | 16 / 60 / 496 / **2178** | 7.3% |

62 orbits is `max_sequence_length`: at T=0.5 the sharpened sampler stops
emitting STOP at all for part of the cohort and is cut off by the length cap.
This is the ordinary low-temperature degeneracy of an autoregressive model, and
it is what makes the cold end of the sweep expensive rather than merely worse --
a 2178-atom cell is a PyXtal timeout and an ORB relaxation that will not fit on
a 4.7 GiB K20c. It also explains the non-monotone validity column: below T=0.7
the runaway genes start failing formal validity too.

## Gene novelty is monotone in the temperature

Novelty against LeMat-Bulk, by the sampled gene's augmented Wyckoff
fingerprint. Wilson 95% intervals on 1000 genes.

| | T=0.5 | T=0.6 | T=0.7 | T=0.8 | T=0.9 | **T=1.0** | T=1.1 | T=1.2 | T=1.5 |
|---|---|---|---|---|---|---|---|---|---|
| unique gene | 0.992 | 0.999 | 1.000 | 1.000 | 1.000 | **0.998** | 1.000 | 1.000 | 1.000 |
| novel gene | 0.541 | 0.577 | 0.576 | 0.635 | 0.659 | **0.679** | 0.735 | 0.770 | 0.809 |
| ±95% | ±0.031 | ±0.030 | ±0.030 | ±0.030 | ±0.030 | **±0.029** | ±0.027 | ±0.026 | ±0.025 |

Gene novelty rises monotonically with temperature over the whole range, and the
span is large against the noise: 0.541 to 0.809, against a per-arm half-width of
about 0.03. Sharpening the sampler moves it towards what it was trained on, and
what it was trained on is in LeMat-Bulk.

Uniqueness is not the mechanism. It is ≈1.0 at every temperature and only dips
to 0.992 at T=0.5 -- the sampler collapses onto *training* genes long before it
collapses onto repeats of itself.

**This is the half of MetaSUN that a cold sampler loses.** Whether it is paid
for by the other half -- how often a relaxed structure lands under the hull --
is what the relaxed table below answers.

## The relaxed funnel: a plateau, then a cliff

Every arm is a fresh 1000-gene cohort from the same checkpoint, relaxed with
ORB-v3 conservative-inf at `fmax = 0.05` under the default `0:1,2:2,*:3` trial
schedule, and scored by one revision of the score stage. The run's *published*
funnel (MetaSUN 0.303) is not among them: it was written before commit
`9bf7073` changed what counts as novel, so it is not comparable.

Every rate is per *sampled* gene -- the denominator the protocol holds
constant -- so a duplicate counts once in the numerator and once per sample in
the denominator. (This is why `novel gene` reads 0.679 here against the
funnel's `gene_novelty_rate` of 0.680: that key is per *unique* gene, and
T=1.0 is the only arm with a duplicate.)

| per sampled gene | T=0.7 | T=0.9 | **T=1.0** | T=1.2 | T=1.5 |
|---|---|---|---|---|---|
| novel gene | 0.576 | 0.659 | **0.679** | 0.770 | 0.809 |
| valid = unique structure | 0.905 | 0.930 | **0.891** | 0.907 | 0.899 |
| novel structure | 0.577 | 0.646 | **0.640** | 0.732 | 0.750 |
| metastable (`e_hull ≤ 0.1`) | 0.565 | 0.522 | **0.463** | 0.394 | 0.289 |
| stable (`e_hull ≤ 0`) | 0.068 | 0.044 | **0.040** | 0.026 | 0.012 |
| **MetaSUN** | 0.274 | 0.284 | **0.261** | 0.257 | **0.187** |
| **SUN** | 0.012 | 0.008 | **0.011** | 0.012 | **0.002** |
| `gene_known_became_novel` | 85 | 61 | **61** | 52 | 44 |

**MetaSUN has a broad plateau from 0.7 to 1.2 and falls off a cliff above it.**
Tested rather than eyeballed, with a Pearson homogeneity test across the arms
and a Cochran-Armitage trend test scored by the temperature itself:

| MetaSUN over | p homogeneity | p trend | slope per unit T |
|---|---|---|---|
| T = 0.7 … 1.2 | 0.507 | 0.286 | −0.042 |
| T = 0.7 … 1.5 | **<0.0001** | **<0.0001** | **−0.112** |

On the plateau the arms are statistically indistinguishable -- the homogeneity
test cannot even reject that 0.274, 0.284, 0.261 and 0.257 are the same number.
Add T=1.5 and the dependence becomes overwhelming. The plateau is not a null
result about the sampler; it is a *range* over which two strong, significant
effects cancel.

### Why they cancel, and why the cancellation breaks

Both halves of MetaSUN depend on the temperature at p < 0.0001, in opposite
directions and at nearly equal magnitude:

| slope per unit T | over 0.7 … 1.2 | over 0.7 … 1.5 |
|---|---|---|
| novel structure | **+0.295** | +0.223 |
| metastable | **−0.352** | −0.356 |

Metastability falls at a constant rate across the whole range. Novelty does
not: its slope *drops* once T=1.5 joins, because novelty is saturating. The
1.2 → 1.5 step buys only **+0.018** novel structure while metastability pays
the usual **−0.105**. Past T≈1.2 the sampler has nothing left to buy novelty
with, and MetaSUN collapses -- 0.257 to 0.187, with SUN going 0.012 to 0.002.

That is the shape to remember: **the cancellation is a coincidence of slopes
that holds only while novelty still has room to grow.**

### The practical consequences

**The protocol's arm-to-arm comparison is safe at the default.** A variant
cannot be made to look better on MetaSUN by nudging its sampling temperature,
as long as it stays on the plateau -- which the default T=1.0 sits comfortably
inside, with room on both sides. That is what the plateau buys: the protocol
measures the model, not the sampler's setting.

**Do not sample hot.** Going from T=1.2 to T=1.5 costs 27% of MetaSUN
(0.257 → 0.187) and five sixths of SUN (0.012 → 0.002) to buy 0.018 of novel
structure. Nothing about the flattened sampler is worth that.

**`stable` is the strongest temperature effect in the sweep, and MetaSUN hides
it.** `e_hull ≤ 0` runs 0.068 → 0.012 from T=0.7 to T=1.5, a factor of 5.7,
monotone and significant at p < 0.0001. SUN stays near 0.012 across the plateau
only because novelty collapses in step. Anyone who wants stable crystals rather
than *novel* stable ones should sample cold; the protocol's readout is
constructed not to reward that.

**Relaxation rescues cold genes more often than hot ones.**
`gene_known_became_novel` -- a gene whose sampled fingerprint is in LeMat-Bulk
but whose relaxed structure is not -- runs 85 / 61 / 61 / 52 / 44 from T=0.7 to
T=1.5. A cold draw lands nearer a real structure and the four-stage relaxation
then walks it somewhere new, which is why T=0.7's gene novelty of 0.576 becomes
a *structure* novelty of 0.577 while T=1.2's 0.770 falls to 0.732.

**One arm-level caveat.** `valid_structure` fails homogeneity (p = 0.030) with
no trend at all (p = 0.34): T=1.0 sits at 0.891 against 0.905/0.930/0.907/0.899
elsewhere. That is arm-to-arm scatter, not temperature, and it is part of why
the anchor's MetaSUN reads slightly below its neighbours'.

## What the OOM costs, and what that does to the numbers

A CUDA OOM is not symmetric noise. It can only ever *remove* a relaxation, so
every rate it touches is biased downwards -- and it lands preferentially on the
cold arms, whose cohorts carry the large-cell tail. It therefore has to be
counted separately from the sampler's own behaviour, or it reads as the cold
sampler being worse than it is.

| | trials | lost to OOM | rate | genes touched | lost every trial | one-sided MetaSUN bound |
|---|---|---|---|---|---|---|
| T=0.7 | 2400 | 5 | 0.21% | 3 | 1 | **+0.003** |
| T=0.9 | 2405 | 3 | 0.12% | 1 | 1 | **+0.001** |
| T=1.0 | 2369 | 1 | 0.04% | 1 | 0 | **+0.001** |
| T=1.2 | 2445 | 0 | 0.00% | 0 | 0 | **0** |
| T=1.5 | 2426 | 1 | 0.04% | 1 | 0 | **+0.001** |

The bound is built from two levels, both counted at their maximum, so the true
distortion lies in `[0, bound]`:

- a gene that lost *every* trial has no structure at all -- a zero in the
  numerator while still counting in the 1000, and at most it would have been
  MetaSUN;
- a gene that lost *some* trials still has a structure, but its best energy was
  chosen from a smaller pool, which can only make it higher.

The cold arm's handicap is real and about five times the anchor's, in
proportion to its cohort's large cells (3.0% of genes at 100+ atoms against
1.1% at T=1.0), and it is still an order of magnitude below the statistical
noise. **No arm needs correcting, and the OOM bias runs the wrong way to
explain the cliff** -- it penalises the cold arms, while the effect found here
is the hot arm collapsing.

Note which card binds: both genes that lost every trial did so on the 2 GiB
GTX 750 Ti, not on a K20c. `analyse_temperature_sweep.py` computes this from
`relaxations.csv` and `structures.csv`; `tests/test_temperature_sweep_oom.py`
pins the arithmetic, including that a PyXtal timeout must not be counted as an
OOM.

### Why the cold end would be expensive to measure at all

The K20c cards have 4.63 GiB and the protocol puts two workers on each, so a
worker has about 2.3 GiB. At T=0.5, where 7.3% of genes ask for 100 atoms or
more and the largest asks for 2178, that binds constantly -- and the arm is
also about twice as slow per trial as the anchor, since the same tail sets the
wall clock. A T ≤ 0.6 arm is therefore not just worse but barely measurable on
this host; the gene-level screen above is what stands for the cold extreme, and
the relaxed sweep starts at T=0.7.

## Reproduce

```bash
# gene half: nine cohorts, screen only (~3 min per arm)
for T in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.5; do
    scripts/platforms/iapetus/run.sh wyformer-protocol-wandb e9ywwsie \
        --output-dir generated/e9ywwsie/temperature/T${T} \
        --temperature $T --condition energy_above_hull=0 \
        --n-genes 1000 --oversample 1.4 \
        --stages screen --no-upload
done

# relaxed half: the same cohorts, generate -> relax -> score (hours per arm)
for T in 0.7 0.9 1.0 1.2 1.5; do
    scripts/platforms/iapetus/run.sh wyformer-protocol-wandb e9ywwsie \
        --output-dir generated/e9ywwsie/temperature/T${T} \
        --temperature $T --condition energy_above_hull=0 \
        --skip-generate --stages generate,relax,score \
        --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:2 \
        --no-upload
done

scripts/platforms/iapetus/run.sh python scripts/analyse_temperature_sweep.py table \
    --sweep-dir generated/e9ywwsie/temperature
```

`cuda:1` is missing from `--devices` on purpose: that card's CUDA context
creation hangs on this host, which cost the first attempt at the T=1.0 arm
eight hours and its score stage. See
[the iapetus troubleshooting notes](platforms/iapetus/troubleshooting.md), and
check each card before a long run.

Per arm: ~2400 trials, 7-15 s per trial on three workers over two cards, so
five to nine hours each. The cold arms are the slow ones -- their cohorts carry
the large cells.

`--no-upload` throughout, deliberately: `wyformer-protocol-wandb` writes the
funnel into `run.summary` and one artifact named `protocol_<run-id>`, so an
uploading sweep would overwrite the run's headline numbers nine times with
cohorts drawn at temperatures nobody asked it about.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md) — the instrument this sweep drives
- [The rationale notes](archive/de_novo_ranking_protocol_rationale.md) — why the cohort is 1000 genes and the readout is `e_hull ≤ 0.1`
- [Improving de novo quality](archive/de_novo_quality_plan.md) — the other levers
