# Predicting the space group: `predict_start`

In every other model in this repository the space group is the **start token**:
the model is given it and scores the sites, `p(sites | space group, cond)`.
Generation draws the space group beforehand from the training split's
distribution (`spacegroup_distribution.json`), a draw that sees no conditioning at
all. For a model conditioned on a chemical system and an energy that is a real
gap. Li-Mn-Co-O is mostly triclinic and monoclinic, and an unconditional draw
starts a third of its batch in 123 and 216 (see
[chemical_system_sampler.md](chemical_system_sampler.md)).

`CascadeTransformer_args.predict_start: true` closes the gap inside the model:

    p(space group, sites | cond) = p(space group | cond) · p(sites | space group, cond)

The second factor is the existing model, unchanged. The first comes from the same
encoder.

## The architecture

- **A learned query**, `start_query`, one `d_model` vector. It stands in for the
  start token while the start token is still unknown. The encoder reads a
  one-token sequence made of that query. AdaLN delivers the conditioning to every
  layer, so the conditioning is all it has to go on.
- **A head**, `start_prediction_head`, built like the site heads (same
  `perceptron_shape` and `num_fully_connected_layers`), from `d_model` to one
  logit per space group the start tokeniser knows. For a one-hot start the
  classes are the space groups in ascending order, and
  `WyckoffTrainer.start_class_vectors` maps each class back to the encoding the
  model reads.

Both are created after every existing module. With the same seed, a model with
`predict_start` gets exactly the initial site-model weights of one without it
(`test_start_prediction.py` checks this). The site factor then starts from the
same point as its parent's.

It does not combine with `relational_attention_bias`, which is a function of the
start token and so cannot be computed before the start token exists.

## The loss

The space-group term does not take optimiser steps of its own. It is added to
every step:

    loss = site_step_loss + w · CE(start) / batch

`train_epoch` draws `known_seq_len` in proportion to the examples viable there,
and the cascade field uniformly. So the expected site step loss is the
per-structure site NLL times `N / (targets · slots)`, where `slots` is the sum of
the viable counts over `known_seq_len`. The space group adds one term to that NLL
per structure. Weighted by the same factor,

    w = N / (targets · slots)

it joins the objective in its true proportion. Every site step stays exactly the
step a model without `predict_start` takes. The trainer logs `w` at startup.

Why not give the space group sampled steps of its own? That would have spent a few
per cent of the parent's site steps on it. A comparison between the two models
would then also be a comparison of training budgets.

## Reading the losses

`evaluate` puts the start token first, and train, val and test each log:

| key | meaning | comparable to the parent's |
| --- | --- | --- |
| `loss.epoch.<split>.spacegroup_number` | NLL of the space group given `cond`, nats per structure | nothing logged; see below |
| `loss.epoch.<split>.{elements,site_symmetries,sites_enumeration}` | as before | same key |
| `loss.epoch.<split>.total_given_start` | sum of the three above | `total` |
| `loss.epoch.<split>.total` | joint NLL including the space group; selects `best_model_params.pt` | `total` + the parent's space-group NLL |

The parent's generative density has a space-group term too: the cross-entropy of
the empirical training distribution on the split, or `p(G | S)` from
`wyformer-system-prior` if that is what the parent is sampled with. That number
has to be added to the parent's `total` before it can be set against this
model's `total`.

A consequence worth knowing: the checkpoint is chosen by the joint NLL. The
parent chooses by the site NLL alone, so the two can pick different epochs.

## Generating

`WyckoffTrainer.generate_structures` settles the conditioning first, whether
passed or drawn from the training rows. It then draws the space group from
`forward_start` for that row, and then the sites. `--temperature` applies to the
space-group draw as well, as does `--guidance-scale` when the model was trained
with classifier-free guidance (`condition_dropout`). `--calibrate` fits a
temperature to the start head on the validation split, as it does for each site field.
An explicit start tensor (`--space-group`, `--sg-dist`, a system plan) still overrides
the model.

`gene_likelihood` refuses these models. Its space-group prior is the saved
empirical distribution, which is not what they sample from.

## The experiment

`yamls/models/lemat/chemsys_e_hull_sg_adamw_wsd.yaml` is
`chemsys_e_hull_adamw_wsd.yaml` plus `predict_start: true`, and nothing else.
The reference is W&B run `chemsys_e_hull_adamw_wsd-20260915-004642` on
`lemat_bulk_fmax1_stress` (20000 epochs; at its end
`loss.epoch.test.total` = 7.181: elements 4.153, site_symmetries 1.780,
sites_enumeration 1.247; val_best 6.623).

    bash scripts/platforms/aspire2a/train_in_pbs.sh \
        yamls/models/lemat/chemsys_e_hull_sg_adamw_wsd.yaml lemat_bulk_fmax1_stress

### Pilot, 2026-09-16

W&B run `azbo14vf` ran `scripts/train.py ... --pilot` (3 epochs, 102 batches
each) on an A100. The code was the commit that adds this page, with two
differences that did not touch the run. The trainer's class-level
`predict_start = False` default was missing; only trainers built by tests via
`__new__` read it. The `wyformer-generate` help strings were also older. Training, checkpointing, start
head calibration and generation from predicted space groups all ran to the end.

| epoch | val `spacegroup_number` | val `total_given_start` | val `total` |
| --- | --- | --- | --- |
| 0 | 3.294 | 40.006 | 43.300 |
| 1 | 3.225 | 33.096 | 36.321 |
| 2 | 2.937 | 27.482 | 30.419 |

Test `spacegroup_number` at epoch 2 was 2.937 nats. For comparison, the
entropy of the train+val space-group distribution, which is what the parent
samples from, is 3.688 nats over 228 groups. The temperature-calibrated
generation (1969 valid of 10000) spread over 72 space groups, led by 123, 12
and 216.

Three epochs say the plumbing works. They do not say how the models compare:
a 3-epoch WSD schedule is not the full run's early phase. The full run's epoch-0
val `total` of 60.93 was logged during its 3600-step warmup, so neither it nor
the pilot's numbers compare with anything else.

## What has not been measured

- Whether the model's `p(G | S, e_hull)` beats `wyformer-system-prior`'s
  smoothed `p(G | S)` on held-out rows. The prior scores -3.265 nats per row on
  held-out rows of seen systems and -1.997 on unseen ones, on `lemat_bulk_fmax1`,
  not `_stress`.
- Whether the extra term costs the site factor anything. Compare
  `total_given_start` against the parent's `total` at matched epochs.
