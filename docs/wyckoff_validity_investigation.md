# Can the backbone learn the space group rules?

Investigation prompted by two symptoms: weak S.U.N. numbers, and space group / site symmetry
inconsistency in generated Wyckoff genes. The question behind them is whether a ~140k-parameter
backbone that cannot reproduce the International Tables can be trusted on energetics.

**Answer: it can reproduce them, and the good checkpoints do.** The legality map costs about
2% of the parameter budget, and the best runs put 99.8% of their probability mass on legal
site symmetries. What is left is a long tail over rare space groups and one constraint the
architecture gives the model no feature for. Neither is a capacity ceiling.

**The evidence that said otherwise was a broken metric.** The W&B `ss_validity` and
`enumeration_validity` curves, which collapse past three sites on every run, were measuring
the STOP rate. Real per-site validity is 99.6% or better at every position out to 11. Fixed
in `generator.py` and `trainer.py`; section 0 has the mechanism and the before/after.

Measured 2026-09-03 on the 30 local checkpoints under `runs/`.

## 0. The logged validity curves measure the STOP rate

This is the finding that reframes everything else. The `ss_validity` curve logged for every
run drops from 0.998 at the first two sites to roughly 0.5 beyond the third, which reads as
"the model loses the space group rules once sequences get long". It is not measuring that.

`WyckoffGenerator.generate_tensors` scores validity at each position with
`(start, ss_token) in token_engineers["multiplicity"].db` (`generator.py:350-363`). That
table is built in `preprocess_wychoffs.enumerate_wychoffs_by_ss` from real
(space group, site symmetry, enumeration) triples only. **The STOP token is not a key in it.**
The engineer carries a `stop_token` as an output *value*, but a membership test on the index
finds nothing, so every sequence that legitimately ends at position k is scored as an invalid
site symmetry at k.

The guard that was supposed to prevent this never fires. `generator.py:348` skips structures
flagged in `stop_generated`, but that flag is only ever set inside
`if self.stops is not None` (`generator.py:313`), and `WyckoffTrainer.generate_structures`
constructs the generator without a `stops` argument (`trainer.py:1085`). No caller in the
package passes one. So `stop_generated` stays all-False and nothing is skipped, meaning
structures that stopped *earlier* than k are also scored, on tokens the model produced while
being asked to continue a sequence that had already ended.

Reproducing the metric on `4lolpokk`, alongside the STOP rate and the score over real sites
only, with 1100 genes:

| known_seq_len | live | stopped at k | as logged | 1 − P(stop \| live) | real sites only |
|---|---|---|---|---|---|
| 0 | 1100 | 0 | 0.9982 | 1.0000 | 0.9982 |
| 2 | 1098 | 32 | 0.9700 | 0.9709 | 0.9991 |
| 3 | 1066 | 238 | 0.7800 | 0.7767 | 0.9988 |
| 4 | 828 | 295 | 0.6209 | 0.6437 | 0.9962 |
| 5 | 533 | 191 | 0.5764 | 0.6417 | 1.0000 |
| 6 | 342 | 204 | 0.4945 | 0.4035 | 1.0000 |
| 8 | 66 | 30 | 0.5582 | 0.5455 | 1.0000 |
| 11 | 10 | 6 | 0.4900 | 0.4000 | 1.0000 |

The logged column tracks the STOP column, not the validity column. The same holds on
`upi73i4k`, where the real-site score stays between 0.978 and 1.000 out to position 15 while
the logged curve sits near 0.55.

**"Most sequences are legal because they are short" does not survive this either.** The length
distribution is right: `4lolpokk` generates a mean of 4.76 sites against 4.689 in the
`alex_mp_20` validation split, and the site-count histogram peaks in the same place. Validity
does not decay with position, so shortness is not carrying the aggregate.

### Fixed

`WyckoffTrainer.generate_structures` now passes `stops=self.stops_dict`, which is what the
`stop_generated` guard was always waiting for. `WyckoffGenerator.generate_tensors` raises if
`compute_validity` is requested without `stops`, so the metric cannot silently degenerate
again, and a field whose tokeniser has no STOP token is tolerated rather than raising.

What the trainer logs for `4lolpokk` after the fix, against the same run's W&B curve:

| known_seq_len | as logged before | after |
|---|---|---|
| 0 | 0.9977 | 0.9982 |
| 1 | 0.9978 | 1.0000 |
| 2 | 0.9688 | 0.9991 |
| 3 | 0.7607 | 0.9988 |
| 4 | 0.5942 | 0.9962 |
| 5 | 0.5481 | 1.0000 |
| 6 | 0.4925 | 1.0000 |

Regression coverage is in `src/wyckoff_transformer/tests/test_validity_stop_handling.py`,
including a test that reproduces the old curve from the same generated tensors, so the
failure mode itself is pinned rather than only its absence.

**Known limitation.** The average at deep positions is taken over very few live structures,
and the count is not logged, so a 1.0000 past position 10 can rest on a handful of genes.
Adding a live-count column to the W&B table would fix that, at the cost of changing the
table schema and the `generate_tensors` return arity, so it is left out here.

`scripts/diagnostics/logged_validity_curve.py` still prints both versions from a checkpoint,
for reading historical runs whose logged curves predate the fix.

## Correction to an earlier draft of this document

An earlier version reported 45% gene acceptance, a 0.29 total variation distortion of the
space group distribution, and space group conditioning that decays along the sequence.
**All three were artefacts and are withdrawn.** Two mistakes produced them:

- **The checkpoint was unrepresentative.** `885f127a` was picked as "a real alex_mp_20 run",
  and it is, but it is an older one superseded a week later. The sweep in section 4 shows
  properly trained runs on the same data and architecture reaching 100x lower legality error.
- **The decay measurement averaged over dead sequences.** Probability mass on illegal tokens
  was averaged over all samples at each position, including those that had already emitted
  STOP, where the model is being asked to continue a sequence that ended and its output is
  off-distribution. Conditioned on live sequences the decay disappears entirely. The
  production metric has the same defect plus a worse one, which is section 0.

`scripts/diagnostics/illegal_probability_mass.py` now conditions on live sequences and
separates the two failure kinds.

## 1. The legality map is cheap

`SpaceGroupEncoder` hands the model the space group as the matrix representation of its
symmetry operations plus a lattice-centering one-hot, 99 dimensions of rank 58 over 230
groups. Fitting the multi-hot indicator of which site symmetries exist in each group, from
exactly that vector:

| readout | parameters | space groups with the legal set recovered exactly |
|---|---|---|
| linear least squares | 7 800 | 0.513 |
| MLP, 8 hidden | 1 502 | 0.670 |
| **MLP, 16 hidden** | **2 926** | **1.000** |
| MLP, 32 hidden | 5 774 | 1.000 |

Sixteen hidden units and 2 926 parameters reproduce the entire 230 by 78 map with zero bit
errors. The production models have 137k to 144k parameters. Whatever is going wrong when a
model emits an impossible site symmetry, it is not that the map will not fit.

## 2. The good checkpoints have learned it

Probability mass on site symmetries that do not exist in the sampled group, at the first
site, weighted by the space group distribution the model generates from:

| checkpoint | dataset | weighted p(illegal) | gene acceptance |
|---|---|---|---|
| `upi73i4k` | LeMat-Bulk | 0.0014 | 0.964 |
| `agbtx22t` | `alex_mp_20` | 0.0019 | 0.975 |
| `5azgyig0` | `alex_mp_20` | 0.0021 | 0.970 |
| `4lolpokk` | `alex_mp_20` | 0.0021 | 0.967 |
| `t1c9ehzp` | LeMat-Bulk | 0.0018 | not measured |

Conditioned on live sequences, the error is flat in position rather than decaying.
`upi73i4k`, mean over sequences still alive at each index:

| site index | live | p(site symmetry not in group) | p(site symmetry exhausted) | p(enumeration consumed) |
|---|---|---|---|---|
| 0 | 1500 | 0.0017 | 0.0000 | 0.0000 |
| 2 | 1495 | 0.0014 | 0.0007 | 0.0011 |
| 4 | 1083 | 0.0021 | 0.0045 | 0.0069 |
| 6 | 547 | 0.0027 | 0.0121 | 0.0259 |
| 8 | 142 | 0.0011 | 0.0211 | 0.0319 |

The first column does not move. The two that do are the occupancy columns, and they are a
different problem. See section 5.

Rejection is correspondingly almost harmless on a good checkpoint. On `agbtx22t`,
per-space-group acceptance ranges from 0.733 to 1.000 with standard deviation 0.050, and the
total variation distance between the requested and delivered space group distributions is
**0.017**.

## 3. The long tail is the real weakness, and it is coverage, not capacity

Splitting the same measurement by how many training structures each space group has, on the
best `alex_mp_20` checkpoint:

| training structures in the group | groups | mean p(illegal) | median |
|---|---|---|---|
| 3 001 and above | 44 | 0.0014 | 0.0008 |
| 301 to 3 000 | 43 | 0.0104 | 0.0050 |
| 31 to 300 | 42 | 0.0886 | 0.0414 |
| 1 to 30 | 52 | 0.2682 | 0.1658 |

Correlation with the log training count is -0.50, and the same -0.50 holds for `upi73i4k` on
LeMat-Bulk at a lower overall level. Unweighted over groups the best checkpoint sits at 0.100
against 0.0019 weighted: the model is essentially exact where it has data and roughly a coin
flip where it has thirty examples.

This is the expected behaviour of a model that is working. It is also the honest boundary of
what it can be trusted on, and it is the same boundary that applies to stability.

## 4. The spread across checkpoints is larger than any architectural effect

Frequency-weighted p(illegal site symmetry) across every local checkpoint with a
`sites_enumeration` head:

| band | runs | weighted p(illegal) |
|---|---|---|
| best trained | `agbtx22t` `upi73i4k` `t1c9ehzp` `5azgyig0` `4lolpokk` `e9ywwsie` `g6mng6ie` | 0.0014 to 0.0028 |
| trained | `lu4xqw0w` `mawdayg4` `9p1osc13` `mhnnuaqg` `9g489lsv` | 0.0050 to 0.019 |
| superseded | `885f127a` | 0.243 |
| partial or aborted | `8so7mhir` `ffyyrvlm` `s8bgnzq8` `5dcvhzo8` `12dbgi2p` `oc5zc1x4` | 0.57 to 0.77 |
| `epochs: 3` pilots | `gd0yuyb7` `l9yi37fb` `4qllmdxh` `5muzp012` `aw2k0i3x` `dfm10f4q` `o1ui6bv5` `o4qtz7rz` | 0.88 to 0.91 |

A uniform predictor scores 0.936, since the mean legal set is 5 of 78 site symmetries, so the
pilot band is untrained as expected. The point of the table is the range among things that
are not pilots: two orders of magnitude, at fixed architecture and fixed data. Architecture
is not what is separating these runs.

`885f127a` is worth a look. It ran the full 1.5M epochs and still sits at 0.243, against
0.0019 for runs on the same config a week later. If any reported result rests on it, that is
worth knowing.

## 5. What actually remains: the one constraint with no feature

On every good checkpoint the dominant rejection cause is now `repeated_0dof_wp`, reusing a
zero-degree-of-freedom Wyckoff position:

| checkpoint | accepted | `repeated_0dof_wp` | `ss_not_in_sg` | `enum_out_of_range` |
|---|---|---|---|---|
| `agbtx22t` | 0.975 | 0.018 | 0.003 | 0.003 |
| `5azgyig0` | 0.970 | 0.022 | 0.005 | 0.003 |
| `upi73i4k` | 0.964 | 0.026 | 0.006 | 0.004 |

This is the only one of the three rules that depends on the generated prefix rather than on
the start token, and it is the only one the architecture gives the head no feature for.
`concat_token_presence: true` bincounts **the single field currently being predicted**
(`cascade/model.py:489-496`), so the site-symmetry head sees which site symmetries have been
used and the enumeration head sees which enumeration indices have been used, and **neither
sees which site symmetry and enumeration pairs have been used**. The constraint is joint over
that pair. Attention has to recover it by comparing concatenated per-field embeddings.

A space group has at most 27 Wyckoff letters, so the missing feature is a 27-bit occupancy
bitmap. `docs/de_novo_quality_plan.md` item 5 already proposes adding degrees of freedom and
orbit-type descriptors as inputs. Degrees of freedom is exactly the field this rule turns on,
and the model currently never sees it.

## 6. How much does validity tell us about stability?

Less than it looks, in both directions, and this matters for how the proxy is read.

**A high accept rate is weak evidence of a good model.** Legality is a deterministic function
of the start token, memorisable in 2 926 parameters. A model can saturate it and know nothing
about energetics. It is a floor, not a signal.

**A low accept rate is not evidence of a bad backbone either.** It separated a superseded run
from a current one, and pilots from trained runs, which is real diagnostic value, but it did
so by measuring how far training got, not what the architecture can hold.

**What the metric is genuinely good for is coverage.** Read per space group rather than as a
single number, it is a direct readout of where the model has enough data to be exact and
where it does not, with a clean -0.50 dependence on log training count. That boundary is not
specific to symmetry. The rare-space-group region where legality degrades to 27% is the same
region where a stability prediction should be distrusted, and validity gives you that map for
free, before any relaxation. That is a better use of the proxy than the aggregate.

## 7. Keeping the diagnostic and fixing the sampler are not in conflict

Unconstrained sampling is deliberate, to keep validity as a proxy. That trade is not
necessary: **the diagnostic does not need invalid samples, it needs the illegal probability
mass**, which is available at every step before any mask is applied. Logging it is strictly
better than the accept rate:

- **Continuous.** It moves long before genes start dying, so it separates the 0.0014 and
  0.0019 checkpoints, which the accept rate cannot.
- **Decomposable.** Legality, exhaustion and consumed-enumeration are three separate numbers,
  and section 2 shows they behave completely differently along the sequence.
- **Resolvable per group and per position**, which is what makes section 3 possible at all.
- **Low variance.** The tables above come from 1 500 genes, where accept rates at this level
  need tens of thousands to distinguish anything.
- **Not confounded by length.** The accept rate compounds per-site errors over the sequence,
  so it conflates the error rate with the length distribution.

Masking the sample removes none of that. `scripts/diagnostics/constrained_sampling_prototype.py`
records the pre-mask mass and then masks, and on `agbtx22t` takes acceptance from 0.970 to
1.000 while leaving the diagnostic intact.

## Recommendations

1. **Fix the logged validity metric first. Done.** It reported the STOP rate, so every
   judgement drawn from it about long sequences was unsupported. Section 0. Runs logged
   before this change should be re-read with `scripts/diagnostics/logged_validity_curve.py`
   rather than trusted.
2. **Then replace accept rate with illegal probability mass as the tracked metric**, split
   into legality and occupancy, reported per position and per space group density band. This
   is the change that makes the proxy do the job it was kept for, and it needs no sampler
   change.
3. **Find out what happened to `885f127a`**, and whether any reported result depends on it.
   Two orders of magnitude at fixed config is not noise.
4. **Add the joint Wyckoff-letter occupancy bitmap to the head input**, plus degrees of
   freedom as an input field. This targets the only remaining failure mode on good
   checkpoints, and it is the one thing here that is genuinely architectural.
5. **Constrained decoding, once items 1 and 2 are in place.** Worth about 3% of genes on a good
   checkpoint, which is minor, but it is free and it removes the residual space group
   distortion of 0.017. Not before item 1, or the proxy is lost for real.
6. **Treat the rare-space-group tail as a data problem.** The map is deterministic and known,
   so it can be injected rather than learned: oversample rare groups, or supervise the
   legal-set indicator directly as an auxiliary head. 2 926 parameters buys the whole tail.
7. **Do not conclude anything about capacity from these numbers.** The capacity question in
   `docs/de_novo_quality_plan.md` item 1 is still open and still worth running. This
   investigation says only that symmetry validity is not the evidence for it.

## Artefacts

- `scripts/diagnostics/illegal_probability_mass.py` — live-conditioned illegal mass, split by
  failure kind, per position
- `scripts/diagnostics/gene_rejection_reasons.py` — per-site first-failure classification
- `scripts/diagnostics/constrained_sampling_prototype.py` — masked sampler and A/B comparison
- `scripts/diagnostics/stop_head_disagreement.py` — sequence-end disagreement across the three
  heads, a minor issue at 0.8% on `885f127a` and 0.05% on `upi73i4k`
- `scripts/diagnostics/logged_validity_curve.py` — the section 0 reproduction and correction
- `scripts/diagnostics/legality_map_capacity.py` — the section 1 fit
- `scripts/diagnostics/legality_by_space_group.py` — the section 3 breakdown
- `scripts/diagnostics/legality_checkpoint_sweep.py` — the section 4 sweep

All take `<run-dir> <n-samples> [condition-value]` unless noted, and need only a checkpoint.
The per-space-group and sweep scripts additionally read `cache/<dataset>/data.pkl.gz` for the
training frequencies.
