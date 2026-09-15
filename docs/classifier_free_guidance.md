# Classifier-free guidance for the e_hull-conditioned WyFormer

> **In progress.** Training started 2026-09-16 01:50 +08 on zeus GPU 1 at commit
> `7cf1895`, as W&B run
> [`ehull_adamw_wsd_5x_cfg-20260916-015500`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x_cfg-20260916-015500).
> Its baseline, [`ehull_adamw_wsd_5x-20260912-115321`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/ehull_adamw_wsd_5x-20260912-115321),
> was itself still training on aspire2a (epoch 22,500 of 40,000). No results yet.

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
therefore gains a trailing presence flag: 1 on every conditioned row, 0 on a
dropped one, with the values zeroed alongside it
(`trainer.guidance_conditioning_width`). The unconditional branch then gets its
own modulation (the AdaLN bias), and the conditional branch gets an independent
offset (the flag's weight). That is the same freedom a learned null embedding
would give, with `CascadeTransformer` unchanged. `condition_dim` goes from 1 to
2, adding 1,440 parameters to 723,350.

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
- **The space group is not guided.** It is the start token, drawn from the
  run's saved unconditional distribution, exactly as for the temperature. A
  guided cohort therefore has the training set's space-group marginal whatever
  it is conditioned on, and so does the baseline's.
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
| W&B run | `ehull_adamw_wsd_5x-20260912-115321` | `ehull_adamw_wsd_5x_cfg-20260916-015500` |
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

## Evaluation plan

All arms are drawn at `--condition energy_above_hull=0` with 1000 genes, and
all are relaxed on the same hardware with the same settings (zeus GPU 1, after
training). Relaxing on CPU is 11× slower per trial here, and 1 trial in 80
exceeded the 300 s `--relax-timeout` there, a handicap GPU arms would not share.

1. **Training curves.** The two runs' `loss.epoch.val.total` at matched epochs,
   and the CFG run's `val_unconditional - val` gap.
2. **Gene-level sweep**, screen only, which is cheap: baseline at w = 1, and the
   CFG model at w ∈ {0, 1, 1.5, 2, 3, 5}. It reads formal validity, uniqueness,
   gene novelty and cohort shape, plus the **archive e_hull** of the genes a
   cohort reproduces. Every generated gene whose fingerprint is in LeMat-Bulk
   already has a PBE e_hull there (the minimum over the archive's structures on
   that gene), so the share of known genes on the hull measures how well the
   cohort follows `e_hull = 0` without relaxing anything
   (`scripts/analyse_guidance_sweep.py index`, then `table`).

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
3. **Relaxed arms**: the baseline, CFG w = 1, and the two most promising guided
   scales from step 2, through the full protocol. The readouts are MetaSUN,
   SUN, metastable, stable and novel structure, each per sampled gene.

## Reproduce

```bash
# train (zeus); resumes itself after a crash
WANDB_ENTITY=symmetry-advantage nohup scripts/platforms/zeus/train_supervised.sh \
    yamls/models/lemat_bulk_ehull/ehull_adamw_wsd_5x_cfg.yaml lemat_bulk_fmax1_stress 1 \
    ehull_adamw_wsd_5x_cfg-20260916-015500 > runs/.logs/cfg.log 2>&1 &

# archive e_hull index, once per reference (~10 min, ~30 GB RAM)
.venv/bin/python scripts/analyse_guidance_sweep.py index

# one arm: cohort + screen on CPU, then relax + score on the GPU
run=ehull_adamw_wsd_5x_cfg-20260916-015500
for w in 0 1 1.5 2 3 5; do
    .venv/bin/wyformer-protocol-wandb $run --output-dir generated/$run/guidance/w$w \
        --condition energy_above_hull=0 --guidance-scale $w --arm cfg-w$w \
        --stages screen --no-upload
done
.venv/bin/wyformer-protocol-wandb $run --output-dir generated/$run/guidance/w2 \
    --condition energy_above_hull=0 --guidance-scale 2 --arm cfg-w2 \
    --skip-generate --stages generate,relax,score --devices cuda:0 --workers-per-device 4

.venv/bin/python scripts/analyse_guidance_sweep.py table --reference base \
    --arm base=generated/ehull_adamw_wsd_5x-20260912-115321/protocol \
    --arm w1=generated/$run/guidance/w1 --arm w2=generated/$run/guidance/w2 \
    --output-dir generated/$run/guidance/tables
```

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md), the instrument the arms are scored with
- [Sampling temperature](temperature_sweep.md), the other sampling-time knob, and the sweep this one is modelled on
- [Every `e_hull` in this repository](e_hull_definitions.md), for what the archive e_hull and the ORB hull each are
