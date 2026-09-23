# Classifier-free guidance for the e_hull-conditioned WyFormer

> **Complete, 2026-09-23.** Guidance works, at an on-support target. At
> `e_hull = 0.05` it raises MetaSUN from the baseline's 0.346 to **0.479** at
> w = 3 (+0.133, p = 2e-9) *and* raises gene novelty at the same time, which no
> other knob here has done. At `e_hull = 0` — the default target — it buys
> nothing (0.265 -> 0.269, n.s.). Runs:
> [`ehull_adamw_wsd_5x_cfg-20260916-055000`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x_cfg-20260916-055000)
> against its baseline
> [`ehull_adamw_wsd_5x-20260912-115321`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x-20260912-115321),
> trained at commit `d3f4a5d`, evaluated at `93fdd3e`.

## Why

WyFormer's e_hull conditioning is soft. The 2026-09-15 comparison of seven
protocol artifacts found that a cohort asked for `energy_above_hull = 0`
reproduces archive genes that are mostly not on the hull, and that raw MetaSUN
sits on a plateau across every e_hull-only variant. The conditioning enters
only through AdaLN, as a linear map from log1p(e_hull) to each layer's scale
and shift, and nothing at sampling time can make the model follow it more
closely than it learned to.

Classifier-free guidance (Ho & Salimans, [arXiv:2207.12598](https://arxiv.org/abs/2207.12598))
is that sampling-time control. One network learns both p(gene | e_hull) and
p(gene), and the sampler extrapolates from the second towards the first.

## What was implemented

### Training: condition dropout

`WyckoffTrainer_args.condition_dropout: p` replaces each training example's
entire conditioning vector with the *null condition* with probability p
(`WyckoffTrainer.build_cond(drop_condition=True)`, called from `train_epoch`
only).

**The null condition needs a column of its own.** AdaLN is affine in its input,
so zeroing log1p(e_hull) does not remove the condition. It sets e_hull = 0,
which is exactly the target the protocol samples at. The conditioning vector
therefore gains a trailing **null indicator**: 1 on a dropped row, whose values
are zeroed, and 0 on every conditioned row (`trainer.guidance_conditioning_width`).
A conditioned row reaches AdaLN exactly as it would without dropout, so the
conditional path keeps the baseline's parameterisation `w·v + b`. The null
condition gets a modulation of its own, `w_null + b`, which is the freedom a
learned null embedding would give, with `CascadeTransformer` unchanged.
`condition_dim` goes from 1 to 2, adding 1,440 parameters to 723,350.

**The indicator's polarity matters; the first attempt had it backwards.**
Run `ehull_adamw_wsd_5x_cfg-20260916-015500` (commit `7cf1895`, stopped at
epoch ~640 on 2026-09-16) used a *presence* flag instead: 1 on conditioned
rows, 0 on dropped ones. On 90% of rows that column is a constant 1, so its
weight is a second copy of the AdaLN bias. By epoch 500 the two pointed the
same way in every layer (cosine 0.87–0.96). AdamW steps each by about the
learning rate, so the conditional modulation offset moved at roughly twice the
baseline's rate (layer-0 norm 6.8 against 5.1), and training went unstable
from about step 5,000:

| steps 45k–52k | baseline | presence flag |
|---|---|---|
| mean train batch loss | 1.27 | 1.51 |
| grad-norm median / max | 0.14 / 6.8 | 0.36 / 518 |
| steps clipped at 10 | 0 | 56 |
| val NLL at epoch 500 | 20.45 | 25.38 |

Its val NLL was 5 nats behind, with site symmetries and enumerations worst
(5.50 / 3.00 against 2.72 / 1.74). The baseline's lr of 3e-3 was already at the
edge of stability, which is presumably why doubling one parameter group's
effective step mattered.

A model without `condition_dropout` trains exactly as before, with no extra
column and no extra RNG draw (`test_a_model_without_dropout_is_unchanged_by_drop_condition`).

**Evaluation always conditions.** `loss.epoch.{train,val,test}` is the
conditional NLL, comparable with a run trained without dropout. `train()` also
logs `loss.epoch.val_unconditional`, the same split under the null condition.
The gap between the two is how many nats per structure the condition is worth
to the model. `best_model_params.pt` is still selected on the conditional `val`.

### Sampling: the guidance scale

`generate_structures(guidance_scale=w)` draws every generated token from

    logits = l_u + w (l_c - l_u),   i.e.   p(t) ∝ p(t | c)^w · p(t)^(1-w)

where `l_c` and `l_u` are the logits under the condition and under the null
condition (`WyckoffGenerator.guided_logits`). Details:

- w = 1 is the plain conditional model and makes a single forward pass.
  w = 0 is the unconditional model. w > 1 moves further from the unconditional
  distribution than the conditional model itself does.
- For w ≠ 1 both branches go through the model as one doubled batch, so a
  generation step costs twice the arithmetic.
- Combining logits rather than log-probabilities changes nothing after the
  softmax. Nor does the order relative to a shared temperature or calibration
  temperature, which divides both terms alike.
- **The space group is guided when the model predicts it.** For a model built
  with `predict_start`, guidance applies to `forward_start` through
  `guided_start_logits` as it does to each cascade step. When the start token is
  not predicted, it is drawn from the run's saved unconditional distribution,
  leaving the training set's space-group marginal whatever the condition or
  guidance scale.
- A model trained without `condition_dropout` refuses any w other than 1.

CLI: `wyformer-generate --guidance-scale W` and
`wyformer-protocol-wandb --guidance-scale W --arm NAME`. The protocol records
`guidance_scale` and `generation_condition` in `manifest.json`, and `--arm`
keeps each arm's funnel and artifact separate from the run's headline
`protocol/` numbers (see [the protocol](de_novo_ranking_protocol.md#evaluating-a-wb-run)).

## The experiment

Two runs, identical except for guidance:

| | baseline | CFG |
|---|---|---|
| W&B run | `ehull_adamw_wsd_5x-20260912-115321` | `ehull_adamw_wsd_5x_cfg-20260916-055000` |
| config | `yamls/models/lemat_bulk_ehull/ehull_adamw_wsd_5x.yaml` | `.../ehull_adamw_wsd_5x_cfg.yaml` |
| `condition_dropout` | — | 0.1 |
| `condition_dim` | 1 | 2 |
| host | aspire2a, A100 40GB, PBS chain | zeus GPU 1, RTX 6000 Ada (shared) |

Everything else is the same: dataset `lemat_bulk_fmax1_stress` (5,127,342
training structures), tokeniser `lemat_bulk_fmax1_sg_multiplicity`, width and
depth, AdamW at 3e-3, WSD schedule with a 20% 1-sqrt decay, batch 50,000, and
40,000 epochs × 102 steps. `TestShippedGuidanceConfig` holds the CFG config to
differ in exactly the two keys above. Verified 2026-09-16:

- the baseline's W&B config matches `ehull_adamw_wsd_5x.yaml` key for key;
- `trainer.py`, `cascade/` and `schedules.py` are unchanged between the
  baseline's start and the CFG run's commit.

p = 0.1 is the value Ho & Salimans and the autoregressive image models
(LlamaGen, [arXiv:2406.06525](https://arxiv.org/abs/2406.06525)) settled on. It
is one run's choice, not a sweep.

The two runs necessarily differ in seed and hardware, so a baseline-vs-CFG
difference at w = 1 carries replicate noise: MetaSUN replicate pairs differed by
~0.02 in the 2026-09-15 comparison. **The guided-vs-unguided comparison has no
such noise.** w = 1, 2, 3 are cohorts from one checkpoint, so that is where the
effect of guidance itself is read.

## Results

Every arm: 1000 genes, ORB-v3 conservative-inf, the `lemat_bulk_fmax1_stress`
novelty reference, relaxed on **zeus GPU 1** with `0:1,2:2,*:3` trials and a
300 s timeout, scored 2026-09-22/23 at commit `93fdd3e`. Intervals are Wilson;
differences are Newcombe with a Fisher exact p. Arms are W&B artifacts
`protocol_<run>.guidance[-c0p05]-w<scale>` on their runs, with
`guidance_sweep_ehull_adamw_wsd_5x_cfg-20260916-055000` carrying the tables.

### Training: dropout cost nothing

| | baseline | CFG |
|---|---|---|
| best val NLL | 16.892 | **16.793** |
| final test NLL | 18.480 | 16.863 |
| val NLL, unconditional | — | 18.091 |

The CFG run ends 0.099 nats *below* its baseline, so training 10% of steps
without the condition cost nothing measurable — within replicate noise either
way. Its unconditional loss sits 1.30 nats above its conditional one, up from
0.77 at epoch 500: the model leans on e_hull more as it trains, which is the
quantity guidance extrapolates along.

### At e_hull = 0.05, guidance buys stability *and* novelty

Per sampled gene, free (post-rattle) track:

| | baseline | w = 1 | w = 2 | w = 3 |
|---|---|---|---|---|
| formal validity of the draw | 0.990 | 0.993 | 0.977 | 0.905 |
| novel gene | 0.614 | 0.607 | 0.598 | **0.662** (+0.048, p=0.029) |
| metastable | 0.607 | 0.558 | 0.660 | **0.711** (+0.104, p=1e-6) |
| **MetaSUN** | 0.346 | 0.336 | 0.394 (+0.048, p=0.030) | **0.479** (+0.133, p=2e-9) |
| SUN | 0.013 | 0.004 | 0.009 | 0.012 (n.s.) |
| P(metastable \| gene known) | 0.775 | 0.723 | 0.838 | **0.926** (+0.151, p=2e-8) |
| P(metastable \| gene novel) | 0.550 | 0.498 | 0.594 | **0.662** (+0.112, p=5e-5) |
| P(e_hull > 0.3 \| gene novel) | 0.057 | 0.086 | 0.030 | 0.032 (−0.025, p=0.029) |
| median ORB e_hull | 0.075 | 0.082 | 0.063 | 0.057 |
| mean atoms per gene | 20.6 | 19.6 | 21.8 | 27.3 |

Four things worth separating:

**Guidance is the whole effect.** w = 1 — the CFG model sampled conventionally —
is at or slightly below the baseline (MetaSUN 0.336 against 0.346, n.s.;
metastable −0.049, p = 0.03). Condition dropout alone buys nothing. Everything
below comes from w > 1, which is a comparison within one checkpoint and so
carries no seed or hardware noise.

**It moves the half that was broken.** The 2026-09-21 retrieval finding
localised the conditioned model's weakness to novel genes. Guidance lifts
P(metastable | gene novel) from 0.550 to 0.662, a 20% relative gain, while
cutting the unstable tail (e_hull > 0.3) almost in half. It also helps on known
genes (0.775 → 0.926), but it is not *only* recall.

**Novelty rises with it.** At w = 3 gene novelty is 0.662 against the
baseline's 0.614 (p = 0.029). This is what distinguishes guidance from every
other knob tried here: [sharpening the sampler](temperature_sweep.md) trades
novelty for stability almost exactly one-for-one, and the 5x capacity increase
bought stability at 0.58 novelty against 0.68. Guidance is not a temperature.
The reason is visible in the cohort: guided genes are *larger* (27.3 atoms
against 20.6), and larger cells are both less likely to be in the archive and,
at this target, more likely to relax into something metastable.

**SUN does not move.** 0.012 against 0.013, and the protocol resolves SUN at
roughly ±0.01 on 1000 genes, so this says little either way; SUN needs ~10,000
genes to develop against.

The price is formal validity, 0.990 → 0.905, and relaxation cost, since the
cells are bigger (7.1 worker-hours against 5.9). Both are affordable at w = 3.

### At e_hull = 0, guidance cannot repair an off-support target

| per sampled gene | baseline | w = 1 | w = 2 |
|---|---|---|---|
| metastable | 0.576 | 0.556 | 0.596 (n.s.) |
| MetaSUN | 0.265 | 0.259 | 0.269 (n.s.) |
| P(metastable \| gene novel) | 0.448 | 0.436 | 0.464 (n.s.) |
| P(e_hull > 0.3 \| gene novel) | 0.127 | 0.108 | 0.138 (n.s.) |
| median archive e_hull of known genes | 0.0141 | 0.0187 | **0.0027** |
| known gene within 0.1 of the hull | 0.374 | 0.359 | 0.427 (+0.053, p=0.018) |

Nothing that needs a relaxation moves. What *does* move is adherence in gene
space: at w = 2 the median archive e_hull of the genes the cohort reproduces
falls from 0.0141 to 0.0027 eV/atom, and more known genes are near-hull ones.
So guidance is doing exactly what it claims — pushing the sample towards the
target — and at this target that buys nothing downstream, because the target
itself is off-support (~3% of rows). The bimodality is not something a sharper
conditional can fix.

Validity also collapses much faster here: 0.897 at w = 2 and 0.628 at w = 3,
against 0.977 and 0.905 at 0.05. Pushing hard towards a target the model has
little data for produces genes that are not self-consistent.

### What to use

`--condition energy_above_hull=0.05 --guidance-scale 3`. On the readout the
protocol ranks on, that is 0.479 MetaSUN against 0.346 for the same recipe
without guidance — the largest single gain any lever in this project has
produced — at no cost in novelty and a modest cost in validity.

### Open

- **The optimum is past the last relaxed arm.** MetaSUN rises monotonically
  through w = 3 and was not relaxed beyond it. w = 5 *was* screened (validity
  0.626, gene novelty 0.769, mean 28.7 atoms), so the turnover is somewhere in
  between; relaxing w = 4 and w = 5 is ~3 GPU-hours and would find it.
- **p = 0.1 dropout is one point**, not a sweep, and the guidance scale
  interacts with it.
- **Cross-study comparisons are not established.** The filtered-model numbers
  this is read against (MetaSUN 0.420, P(metastable | novel) 0.561) come from
  iapetus, which reads 0.025–0.030 eV/atom above aspire2a on the same
  structures; everything in this document is zeus GPU 1 and internally
  consistent.

## Appendix: early readings, epoch 500 of 40,000

Not results -- the model is 1.25% trained and its cohorts are nothing like a finished
model's -- but enough to say the machinery works and what it does.

| at epoch 500 | baseline | CFG | CFG, presence flag (stopped) |
|---|---|---|---|
| val NLL total | 20.45 | 21.24 | 25.38 |
| elements / site symmetries / enumerations | 15.98 / 2.72 / 1.74 | 16.25 / 3.07 / 1.92 | 16.88 / 5.50 / 3.00 |
| val NLL, unconditional | — | 22.01 | — |

The CFG run is 0.8 nats behind the baseline, spread across all three heads. Some of that
is the 10% of steps that train the unconditional branch, some is seed and hardware; whether
it closes by the end of the decay is one of the things the run is for. The conditional and
unconditional losses differ by **0.77 nats per structure**, which is what the model thinks
knowing e_hull is worth.

Guidance moves the cohort, monotonically and in the direction a sharper sampler does
(400 draws per scale from the epoch-500 checkpoint, at `energy_above_hull = 0`):

| w | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| formally valid | 0.78 | 0.67 | 0.50 | 0.37 | 0.18 |
| mean orbits per gene | 4.26 | 5.43 | 7.03 | 9.89 | 17.89 |
| mean atoms per gene | 16.0 | 21.6 | 29.7 | 38.8 | 67.7 |
| mean distinct elements | 3.19 | 3.32 | 3.49 | 3.58 | 4.09 |

Two things follow. Conditioning on `e_hull = 0` lengthens genes (w = 1 against w = 0), and
guidance amplifies that, exactly as sharpening the sampler does in [the temperature
sweep](temperature_sweep.md) -- where the cold arms' runaway tail was what made them
expensive. And formal validity falls with w, so the high-w arms need a larger `--oversample`
to fill a 1000-gene cohort; `run_guidance_sweep.sh` scales it with w for that reason.

## Appendix: the baseline arm at e_hull = 0, gene level (2026-09-19)

The baseline finished its 40,000 epochs on 2026-09-19 (best val NLL 16.892, artifact
`best_model_ehull_adamw_wsd_5x-20260912-115321:v31`, epoch 39,999). Its cohort, 1000 genes
at `energy_above_hull = 0`, screened before any relaxation:

| | baseline arm | archive, row-weighted |
|---|---|---|
| formal validity of the draw | 0.984 | — |
| unique gene | 0.998 | — |
| novel gene (no LeMat-Bulk fingerprint) | 0.584 | — |
| archive-known gene | 0.416 | — |
| known **and** on the archive hull | 0.089 (21.4% of known) | 3.2% |
| known **and** within 0.1 eV/atom | 0.374 (90% of known) | 31.4% |
| median archive e_hull of known genes | 0.014 | 0.205 |

The conditioning is therefore *not* inert at the gene level: among the genes it reproduces
from the archive, on-hull ones are enriched about sevenfold over the archive's own rate, and
the median known gene is at 0.014 eV/atom against the archive's 0.205. That is the bar
guidance has to beat, and it is a higher one than the 2026-09-15 reading of the
three-channel `relational_e_all` cohort (9.2% of known genes on the hull, median 0.080)
suggested.

## Appendix: the evaluation plan, as pre-registered

**Two conditioning targets, 0.05 first.** `energy_above_hull = 0` — what
`DEFAULT_CONDITION_TARGETS` applies to any conditioned model — is off-support:
about 3% of the training rows sit there, and a cohort drawn at it comes out
bimodal, a spike of memorised on-hull genes beside a fatter unstable tail. At
0.05 the same checkpoint is statistically indistinguishable from a model
trained only on the `e_hull ≤ 0.1` slice (MetaSUN 0.395 against 0.420, n.s.;
P(metastable | novel gene) 0.549 against 0.561), where at 0 it reads 0.267 and
0.415. Measured 2026-09-21 on the baseline checkpoint; see
`docs/negative_data_strategy.md`.

So 0.05 is where guidance should be *judged*, and 0 is where it might be
*needed*. Guidance extrapolates away from the unconditional distribution
instead of partitioning it, which is the one lever here that could make an
off-support target usable — and if it cannot, that is worth knowing too, since
0 is the default every conditioned model is sampled at. `run_guidance_study.sh`
runs the 0.05 sweep first and the 0 sweep after it, so they never contend for
the GPU.

All arms are 1000 genes and all are relaxed on the same GPU of the same host.
That is not a convenience: protocol arms are not comparable across machines
(iapetus reads 0.025–0.030 eV/atom above aspire2a on the same structures), and
on zeus CPU relaxation is 11× slower per trial with 1 trial in 80 exceeding the
300 s `--relax-timeout`, a handicap GPU arms would not share.

1. **Training curves.** The two runs' `loss.epoch.val.total` at matched epochs,
   and the CFG run's `val_unconditional - val` gap.
2. **Gene-level sweep**, screen only, which is cheap: baseline at w = 1, and the
   CFG model at w ∈ {0, 1, 1.5, 2, 3, 5}. It reads formal validity, uniqueness,
   gene novelty and cohort shape, plus the **archive e_hull** of the genes a
   cohort reproduces. Every generated gene whose fingerprint is in LeMat-Bulk
   already has a PBE e_hull there (the minimum over the archive's structures on
   that gene), so the archive e_hull of the genes a cohort reproduces measures
   how well it follows its target without relaxing anything
   (`scripts/analyse_guidance_sweep.py index`, then `table`). Both targets are
   screened at every scale; only the relaxed arms are rationed.

   The null to read it against is a model that ignores its condition and
   reproduces training genes in proportion to their frequency. Over
   `lemat_bulk_fmax1_stress` (4,826,004 fingerprints, 5,327,342 rows; computed
   2026-09-16 from `gene_ehull_index.pkl.gz`), row-weighted:

   | archive min e_hull of the row's gene | share |
   |---|---|
   | = 0 (on the hull) | 0.032 |
   | ≤ 0.1 eV/atom | 0.314 |
   | median | 0.205 eV/atom |

   For scale, the known genes of an existing e_hull = 0 cohort
   (`relational_e_all_adamw_wsd-20260909-234259`, three-channel, not part of
   this study) come out at 0.092 on the hull, 0.584 within 0.1, and a median of
   0.080: conditioning moves the distribution, but most known genes stay off
   the hull.
3. **Relaxed arms**, through the full protocol: at 0.05 the baseline and CFG
   w = 1, 2, 3; at 0 the baseline and CFG w = 1, 2. The readouts are MetaSUN,
   SUN, metastable, stable and novel structure per sampled gene — **and, above
   all, metastability split by whether the sampled gene is novel.** The
   2026-09-21 result localises everything interesting to that split: at the
   off-support target the conditioned model matched an unconditional one on
   known genes and lost on novel ones. MetaSUN mixes the two, so a guidance
   effect on the novel half could cancel against the known half and read as
   nothing. `analyse_guidance_sweep.py` reports P(metastable | gene novel) and
   P(metastable | gene known) with their own denominators, plus the share of
   novel-gene structures above 0.3 eV/atom, which is the tail the off-support
   target fattens.

## Reproduce

```bash
# 1. Train (zeus). The supervisor resumes itself after a crash; ~8 days on a shared card.
WANDB_ENTITY=symmetry-advantage nohup scripts/platforms/zeus/train_supervised.sh \
    yamls/models/lemat_bulk_ehull/ehull_adamw_wsd_5x_cfg.yaml lemat_bulk_fmax1_stress 1 \
    ehull_adamw_wsd_5x_cfg-20260916-055000 \
    > "$(.venv/bin/python -c 'from wyckoff_transformer.paths import runs_root; print(runs_root())')/.logs/cfg.log" 2>&1 &

# 2. The archive e_hull index, once per reference (~10 min, ~30 GB RAM).
.venv/bin/python scripts/analyse_guidance_sweep.py index

# 3. Both sweeps, end to end. Waits for both runs to finish training, screens every
#    scale on the CPU, relaxes the pre-registered ones on GPU 1, tabulates, uploads.
WANDB_ENTITY=symmetry-advantage GPU=1 nohup scripts/run_guidance_study.sh \
    ehull_adamw_wsd_5x_cfg-20260916-055000 ehull_adamw_wsd_5x-20260912-115321 \
    /mnt/hdd/kna/wyformer_generated > guidance_sweep.log 2>&1 &
```

One arm by hand, if that is all you want:

```bash
run=ehull_adamw_wsd_5x_cfg-20260916-055000
dir=/mnt/hdd/kna/wyformer_generated/target0p05/$run/w3
.venv/bin/wyformer-protocol-wandb $run --output-dir $dir \
    --condition energy_above_hull=0.05 --guidance-scale 3 --arm guidance-c0p05-w3 \
    --oversample 3.64 --gen-device cpu --stages screen --no-upload
.venv/bin/wyformer-protocol-wandb $run --output-dir $dir \
    --condition energy_above_hull=0.05 --guidance-scale 3 --arm guidance-c0p05-w3 \
    --skip-generate --stages generate,relax,score \
    --pyxtal-cores 16 --devices cuda:1 --workers-per-device 6
```

**`--devices cuda:N` names the card; do not also set `CUDA_VISIBLE_DEVICES`.** Each
relax worker pins itself by writing the index out of its own device string into
`CUDA_VISIBLE_DEVICES` before CUDA initialises, which overwrites what it inherited:
`CUDA_VISIBLE_DEVICES=1 ... --devices cuda:0` puts every worker on physical card 0.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md), the instrument the arms are scored with
- [Sampling temperature](temperature_sweep.md), the other sampling-time knob, and the sweep this one is modelled on
- [Every `e_hull` in this repository](e_hull_definitions.md), for what the archive e_hull and the ORB hull each are
- [The negative-data strategy](negative_data_strategy.md), for why conditioning alone cannot suppress mass, and where the on-support target finding comes from
