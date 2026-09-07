# Conditioning on how dirty each training row is

Run [`19qbxo6l`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/19qbxo6l),
config `yamls/models/lemat_bulk_ehull/e_all_adamw_wsd.yaml`, dataset `lemat_bulk_fmax1`.
The first attempt, [`wjwmgjag`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/wjwmgjag),
diverged and was killed — see [What went wrong the first time](#what-went-wrong-the-first-time).

## The problem with filtering

`lemat_bulk_ehull`, the dataset behind every conditional backbone so far, was cut at
`max_force <= 0.02` eV/Å by `scripts/pipeline_lemat_20wyckoffs.py`. That reads as a
convergence filter. It is closer to a provenance filter.

`wyckoff_transformer.formula_energy.dataset` already measured what it removes: the cut
keeps 95.6% of Alexandria rows and 35.5% of the ICSD-backed Materials Project ones,
because MP reports forces from a different protocol whose median is 0.028 — above the
cut. `scripts/pull_mp_provenance.py` records the same thing from the other end: 138,931
MP rows in the energy CSV against 31,794 in the cache. The rows a de-novo generator would
most like to imitate are the ones the filter is most likely to drop.

The alternative to filtering a dirty dataset is to keep it and *say* how dirty each row
is. That is what this run does: three conditioning channels, one of which is the
convergence of the relaxation that produced the row.

## The dataset

`scripts/build_lemat_bulk_fmax.py`, cutting at 1 eV/Å instead — enough to exclude only
pathological relaxations (the archive's `max_force` runs to 12,804 eV/Å).

| step | rows |
| --- | ---: |
| LeMat-Bulk `compatible_pbe` | 5,335,299 |
| `max_force <= 1` | 5,303,342 (99.40%) |
| … `e_hull` is not NaN | 4,714,843 |
| … `\|e_form\| <= 5` | 4,714,007 |
| … has an `immutable_id` | 4,714,006 |
| … at most 61 Wyckoff sites | 4,709,425 |
| *the same cascade at `max_force <= 0.02`* | *4,207,250* |

4,509,453 train / 99,985 val / 99,987 test, against 4,007,729 / 99,998 / 99,996 before.
Twelve per cent more rows, and the half million extra are disproportionately the
experimentally grounded ones.

Three filters beyond the force cut, each earning its place:

**`e_hull` is not NaN** (−588,499 rows). `scripts/compute_e_hull.py` returned nothing for
any system containing Yb, anything past Po, or a chemsys of ten or more elements —
of which the Po-and-beyond clause was 98% of the loss and the ten-element one could never
fire. **Those exclusions are gone as of 2026-09-07**: the archive was relabelled with
`formula_energy/hull_table.py`, which applies none of them, so this filter now drops one
row rather than 588,499, and the dataset gains half a million rows of Yb and actinide
chemistry that no cache had ever contained ([what every `e_hull` in this repo
means](e_hull_definitions.md)). The numbers in this document describe the dataset as
built for `19qbxo6l`, before that. The
model is conditioned on `energy_above_hull`, and nothing downstream masks a missing
conditioning value, so these rows cannot be carried.

**`|e_form| <= 5`** (−836 rows). The archive's formation energies run to −37.9 and
+650.7 eV/atom, which are corrupt rather than exotic; `formula_energy` applies the same
window. It matters more here than in a mean-fitting model, twice over: `delta_e_polymorph`
takes a minimum, so one corrupt row poisons the label of every other polymorph of its
composition, and a spuriously low energy also *defines* the hull, so the row arrives
labelled perfectly stable — exactly the thing a stability-conditioned generator would
learn to imitate.

**At most 61 Wyckoff sites** (−4,581 rows, 0.101%). Not a truncation: cutting sites off a
structure silently changes its composition, so its energy labels would then describe a
compound that is not in the row. Every sequence tensor is padded to the longest structure
in the dataset, so 0.1% of rows running past 61 sites would take the padded width from 62
to 361 and roughly sextuple both resident memory and per-step cost. 61 is not an arbitrary
choice: it is what the previous dataset used. That was not documented anywhere, but it is
recoverable — the old cache holds 224 fewer rows than the CSV it was built from, and
exactly 224 of its rows exceed 61 sites.

## The three labels

All per structure, all in physical units, all non-negative.

**`energy_above_hull`** — eV/atom, from `scripts/compute_e_hull.py` against LeMat-Bulk's
own phase diagram. Clipped at zero: `get_e_above_hull` is non-negative by construction and
one row lands at −2.7e-15, which `log1p` would otherwise reject.

**`delta_e_polymorph`** — eV/atom above the best polymorph of the same *reduced* formula
present in this dataset. This is what absolute `e_hull` cannot express. From one row at
0.15 eV/atom the model cannot tell the best structure anyone has computed for that
composition from a polymorph 0.1 above it, and those two deserve different generations.
Computed from `energy_corrected / n_atoms` rather than from `e_form`: within one
composition the elemental reference cancels, so the differences are identical, and the
per-atom energy is defined for every row including the 11% where the hull construction
returned nothing. Median 6.3e-5 eV/atom; 49.9% of training rows are exactly zero, over
2,325,328 distinct reduced formulas.

**`max_force`** — eV/Å, in [0, 1] by construction. Median 0.0035, p99 0.098.

### Which population the minimum is taken over

The whole filtered dataset, not the train split alone.

A train-only minimum makes the label mean something different on either side of the
split. 62.9% of held-out rows share a reduced formula with train, and a quarter of those
sit *below* the train minimum — so a quarter of the covered validation rows would get a
negative offset, which `log1p` cannot represent and which does not mean anything anyway.

The usual objection to computing a label across the split is leakage, and it does not bite
here: `delta_e_polymorph` is a model *input*, not a target. Knowing that some other
structure of the same composition is 0.1 eV/atom lower says nothing about the Wyckoff gene
this row is asking the model to predict. The cache's split has been row-random all along,
so compositions already straddle it.

The honest caveat is a different one, and it is the repo's own: "lowest in this dataset" is
not "the ground state". Seven in ten LeMat-Bulk formulas are singletons with median
`e_hull` 0.235, mostly substitution hypotheticals, so for those the label is zero by
construction rather than by merit (`docs/gene_energy_critic_study.md`). Half the training
rows carry a zero for that reason. What the channel buys is the ability to *say* zero at
generation time; whether the model can act on it is what this run tests.

### Per structure, not per gene

`wyckoff_transformer.censored.gene_level_polymorph_delta` computes a related label and
deliberately assigns it per *gene* — `min(E | gene)` minus the composition's best
`gene_min` — on the grounds that a gene-reading model handed per-structure energies sees
one input with several targets. `delta_e_polymorph` here is per structure, which is what
was asked for and what the conditioning interpretation wants: the label is an input, so two
rows sharing a gene at different offsets are two well-posed `(input, target)` pairs rather
than a contradiction, exactly as they already are under `energy_above_hull`.

It is still worth knowing where that leaves the channel. 3.9% of LeMat-Bulk rows share
both formula *and* gene with another row, and for those the offset varies while the
model's entire input is identical — so on that slice the channel is noise the model can
only average over. The gene-level label remains the sharper instrument if this run shows
the structure-level one is too blunt.

### The gene-energy screening target

The initial gene critic uses the same `lemat_bulk_fmax1` rows, but it has a
different target: `gene_min_formation_energy_per_atom`, the lowest PBE
formation energy observed among every structure sharing an
augmentation-invariant Wyckoff gene. It deliberately uses ordinary MSE rather
than the censored likelihood: for this first version, the archive's lowest
observation is treated as that gene's attainable energy.

The raw `formation_energy_per_atom` travels through the split CSVs only so
cache construction can derive that minimum across **all** train, validation,
and test rows. A split-local minimum would change the target's meaning on the
held-out rows. The critic reads `max_force` as its only condition during
training; `wyformer-gene-screen` sets `max_force=0` at inference, asking for
the clean-relaxation limit of a generated gene rather than its noisy source-row
realisation.

## Held-out sets are inherited, not resampled

`data/lemat_bulk_fmax1` is a superset of `lemat_bulk_ehull`. A fresh random split would
scatter the old held-out rows into the new training set and pull old *training* rows into
the new validation set — either direction quietly invalidates any comparison against the
already-trained e_hull-only baseline. So val and test are the previous dataset's val and
test ids verbatim, minus the 13 and 9 respectively that the new filters drop.

One consequence to keep in mind when reading the loss curve: since the old dataset was cut
at 0.02, **`max_force` on val and test never exceeds 0.02**. Validation measures the clean
subpopulation — which is the mode generation targets, so this is the right thing to
optimise — but it says nothing about how the model handles the dirty tail it trains on.
Measuring that needs a held-out set drawn from the new rows, which this run does not have.

## Conditioning three channels instead of one

The AdaLN machinery was always width-agnostic: the modulation is
`nn.Linear(condition_dim, 2 * d_model)` per encoder layer, zero-initialised, and a
`condition_dim=2` layer test has passed for as long as it has existed. Everything *above*
it assumed a single channel — one feature name, one transform applied to the whole tensor,
a literal `1` in the width, and CLIs that filled every column of the conditioning vector
with the same number.

`condition_feature` now takes a list, `condition_transform` takes one name for all of them
or a list or a per-name mapping, and `WyckoffTrainer.build_cond` remains the single place
the vector is assembled. **The order of `condition_feature` is load-bearing**: every
modulation weight is tied to it, so reordering the list repoints all of them, silently.
`src/wyckoff_transformer/tests/test_multi_condition.py` pins it.

Three smaller things came with it, each fixing something that could not previously fail
loudly:

- The width check — model's `condition_dim` against what `build_cond` produces — used to
  run only when `composition_conditioning` was on. A config declaring `condition_dim: 3`
  with one scalar feature passed every check and failed later inside `nn.Linear` with a
  bare shape mismatch. It now always runs, and `from_config` derives the width rather than
  trusting the number in the yaml.
- A NaN in a conditioning column used to pass through tokenisation untouched (the
  `no_processing` path is one `torch.Tensor(...)` call with no validation) and poison every
  modulation it touched. It is now rejected at trainer construction, naming the column.
- `wyformer-generate` and `wyformer-csp` took a single `--condition-value` float and
  broadcast it across every column. They now take `--condition NAME=VALUE`, once per
  feature; the single-value form survives for single-feature models and is refused rather
  than broadcast otherwise. The seven diagnostics scripts that reimplemented the same
  broadcast go through `WyckoffTrainer.build_condition_from_values`.

### Scales

`condition_scale` divides a channel before its transform. It exists because the three
channels are not commensurable: unscaled, `max_force` reaches `log1p` with a standard
deviation of 0.020 against 0.317 for `e_hull`, and the AdaLN weight would have to make up
two orders of magnitude.

The value is `max_force: 0.05`, chosen by matching spread and tail rather than centre:

| channel, post-transform | sd | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: |
| `energy_above_hull` | 0.317 | 1.317 | 1.622 | 3.642 |
| `delta_e_polymorph` | 0.207 | 0.939 | 1.450 | 1.896 |
| `max_force` @ 0.05 | 0.216 | 1.074 | 1.702 | 3.043 |
| *`max_force` @ 0.01 (diverged)* | *0.525* | *2.365* | *3.153* | *4.613* |

This is a units choice recorded in the config, not a statistic fitted to the data. That
distinction matters: nothing in this repo persists conditioning statistics alongside the
weights, so a fitted scaler would have to be reconstructed at generation time to make the
model mean what it meant in training.

Matching medians instead — which is how 0.01 was picked, and what killed the first run — is
the wrong summary for a variable that is exactly zero on a quarter of rows and has a long
right tail. It equalises the middle and widens the end.

All three channels use `log1p`, so **(0, 0, 0) maps to exactly (0, 0, 0)**: "on the hull,
the ground state of its formula, from a converged calculation" is the origin of the
conditioning space rather than an extrapolation past the edge of the training
distribution.

## Generating from it

```
wyformer-generate ... \
  --condition energy_above_hull=0 \
  --condition delta_e_polymorph=0 \
  --condition max_force=0
```

Omitting them samples whole conditioning rows from the training distribution, which keeps
the three paired as they actually occur; that is what the automatic post-training
evaluation does, so its numbers say nothing about conditioned behaviour.

To audit one channel while holding the others fixed:

```
python scripts/audit_ehull_conditioning.py --run-path runs/wjwmgjag \
  --sweep-feature delta_e_polymorph --baseline max_force=0
```

That writes to `generated/wjwmgjag/delta_e_polymorph_conditioning_audit/`. The output
directory now follows the run and the swept channel, and the script refuses to write over a
directory that already holds a `manifest.json`: the existing
`generated/upi73i4k/ehull_conditioning_audit/` has relaxations and stability scores
downstream of its genes, and regenerating those genes in place would leave every number in
`docs/upi73i4k_ehull_conditioning_audit.md` describing samples that no longer exist. Rerunning
the original audit still lands on its historical path, so the five shell scripts that read
`wyckoff_genes_ehull_*.json.gz` are unaffected.

## Building it again

```
python -m wyckoff_transformer.formula_energy.hull_table --workers 16 \
  --input-file data/lemat-bulk/lemat_pbe.csv.gz \
  --output-file data/lemat-bulk/lemat_pbe_ehull.csv.gz
python scripts/build_lemat_bulk_fmax.py --name lemat_bulk_fmax1 --max-force 1.0 --rebuild-labels
python scripts/cache_a_dataset_reusing.py lemat_bulk_fmax1 \
  --reuse cache/lemat_bulk_fmax1/data.pkl.gz \
  --scalar-columns energy_above_hull delta_e_polymorph max_force max_force_missing \
      formation_energy_per_atom \
  --observed-gene-minimum-target \
  --max-sites 61 --n-jobs 16
python scripts/tokenise_a_dataset.py lemat_bulk_fmax1 \
  yamls/tokenisers/lemat_bulk_fmax1_sg_multiplicity.yaml --new-tokenizer
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  yamls/models/lemat_bulk_ehull/e_all_adamw_wsd.yaml lemat_bulk_fmax1 cuda
```

The first line is new and takes 1 h 23 min on 16 workers: the hull table is no longer a
given, because the labels of `19qbxo6l` were built before the element exclusions came out of
it. Reuse the *fmax1* cache rather than the `lemat_bulk_ehull` one now — it overlaps the new
dataset by 88% against the older cache's 78%, and `--reuse` takes several caches if you want
both. `max_force_missing` has to be in `--scalar-columns` or the indicator is built and never
reaches a tensor.

The dataset and the tensor cache are unchanged between `wjwmgjag` and `19qbxo6l`; only the
config differs, so a rerun needs neither rebuilt.

`cache_a_dataset_reusing.py` exists because the new dataset overlaps the old one by 4.0M
of 4.5M rows, and `structure_to_sites` is a pure function of the structure and the
tolerances, so those symmetry determinations can be copied instead of repeated: 4,007,053
reused against 506,981 computed, 25 minutes rather than about six hours. It recomputes 200
reusable records per split and aborts if any differs, which is how the 61-site cap and the
sort-by-Wyckoff-letter convention of the existing caches were found in the first place —
`structure_to_sites` only sorted when `max_wp` was set, so a naive reuse would have mixed
two site orderings.

`scripts/cache_a_dataset.py` still does the whole thing from scratch — about six hours —
and needs two flags to land on the same cache:

```
python scripts/cache_a_dataset.py lemat_bulk_fmax1 --n-jobs 16 --sort-by-letter \
  --scalar-columns energy_above_hull delta_e_polymorph max_force formation_energy_per_atom \
  --observed-gene-minimum-target
```

`--sort-by-letter` is new and is what makes the site order match; previously that ordering
was only reachable as a side effect of `--max-wp`, which also truncates. That script has no
site-count cap, so it keeps the 4,581 structures with more than 61 Wyckoff sites and the
padded sequence width goes to 361 — filter them out before caching, or use the reusing
script's `--max-sites`.

## What went wrong the first time

`wjwmgjag` reached its best at epoch 3000 and then climbed, on train and validation
together:

| epoch | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 18.34 | 18.18 | **17.72** | 17.84 | 18.59 | 20.13 |
| val | 17.99 | 17.79 | **17.39** | 17.48 | 18.19 | 19.71 |

Train rising with validation rules out overfitting. The comparison that makes it a defect
rather than noise is the two single-channel runs at the *same* learning rate: `upi73i4k` and
`e9ywwsie` oscillate between 17.0 and 18.9 for 85,000 epochs and finish at their best
(17.01). They never leave that band; 20.13 is well outside it, and four consecutive rises
is not an excursion.

Three candidate causes were checked and ruled out:

- **Weight-norm blow-up**, the failure mode `t1c9ehzp` had. Total parameter norm grew 5%
  between epoch 3000 and 6000, AdaLN's 8%. Not it.
- **Gradient clipping becoming the step size**, the failure mode `lu4xqw0w` had. `grad_norm`
  never came near `clip_grad_norm: 10`. (It is logged once per validation, so its apparent
  10× rise across seven points is sampling noise, not a trend.)
- **Misaligned conditioning** — a real possibility given the cache is assembled by
  concatenating reused and freshly computed blocks. Checked exhaustively rather than by
  sampling: for all 4,509,453 train and 99,985 val rows, `pure_sequence_length` equals that
  row's site count and each conditioning tensor equals that row's dataframe column. Exact.

What was left is the conditioning itself, and there the scale is measurably wrong. AdaLN is
`Linear(cond)` feeding a `(1 + gamma)` scaling with no bound on `gamma`, so the tail of the
conditioning input is the tail of the modulation:

| | AdaLN `|W|` | `|gamma|` max |
| --- | ---: | ---: |
| `upi73i4k`, 1 channel, stable at this lr | 1.9 – 4.0 | 6.6 |
| `wjwmgjag`, 3 channels @ `max_force: 0.01` | 2.7 – 12.9 | 23.4 |

Averaged over the training set the three channels contributed comparably to the modulation
(0.10 / 0.05 / 0.11), so the scale was not dominating the *centre* — it was widening the
*tail*, which is what an unbounded modulation is sensitive to.

**This is a diagnosis, not a proof.** The scale is the one demonstrable error, and it is
fixed. The learning rate remains the other candidate: 3e-3 is three times the usual value
for a small transformer, the config has always flagged it as its least-established number,
and three channels make the conditioning input norm about 1.4× the baseline's. It is kept
at 3e-3 for now because the baselines are stable there and the run is now short enough that
the rate matters. If `19qbxo6l` turns upward too, 1e-3 is the next change, and it will then
be the only one — which is why both were not changed at once.

`validation_period` is now 250 rather than 1000. The turn happened somewhere between epochs
3000 and 6000 and, at one point per 1000 epochs, that was four measurements spread over six
hours before it was legible. Validation is two batches against ninety for training, so the
resolution is nearly free.

## Run

4,509,453 train rows at batch 50,000 is 90 steps per epoch, so `epochs: 20000` is
1,800,000 optimiser steps — measured at 3.8 s/epoch, roughly 21 hours, of which the last
20% (360,000 steps, ~4 h) is the WSD decay. **Most of the improvement in WSD appears in
the decay; do not read the flat stable phase as a failed run, and do not kill it early.**

The comparison this is set up for is against `lemat_bulk_ehull`'s e_hull-only model on the
same held-out structures. What it cannot answer on its own is whether the gain (if any)
comes from the extra rows or from the extra channels; separating those needs an
f_max ≤ 1 run conditioned on `energy_above_hull` alone.

The cache and tensor artifacts currently on disk predate the gene-energy
target. Re-run the cache and tokenisation commands above before training
`yamls/models/lemat_bulk_fmax1/gene_min_energy_adamw_wsd.yaml`.
