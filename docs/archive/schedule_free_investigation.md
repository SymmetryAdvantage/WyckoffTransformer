# Why the LeMat-Bulk training loss rises

Investigation into runs whose **training** loss climbed steadily after an early minimum —
first seen on `lu4xqw0w`, reproduced on `t1c9ehzp`. Four defects were found: three in our
code, all fixed, and one in how we use `schedulefree`, still open.

Closed 2026-08-31 with the mechanism established and its cost not yet demonstrated.

## The phenomenon

Both production runs on the e_hull-conditioned LeMat-Bulk set reached a minimum and then
degraded by ~0.4 nats, on every split at once:

| run | epochs | steps | wall | lr | best val | final val | rise | train − val |
|---|---|---|---|---|---|---|---|---|
| `lu4xqw0w` | 149 000 | 23 840 162 | 117.3 h | 0.2 | 19.3138 | 19.7040 | **+0.3902** | +0.0247 |
| `t1c9ehzp` |  80 000 | 12 808 827 |  66.1 h | 1.4 | 18.2770 | 18.7049 | **+0.4279** | −0.0293 |

### It is not overfitting

The single most important fact, because it is what the curve looks like at a glance:

- **Train rises too.** `lu4xqw0w` ended at train 19.7288 / val 19.7040 / test 19.7027 —
  all three within 0.03 of each other, correlation 0.9999.
- **The gap is ~0 and sometimes negative.** `t1c9ehzp` finished with train *above* val by
  0.029: the validation split is the easier one.
- **There is no capacity to overfit with.** ~144 000 parameters against 4 007 729 training
  structures is 28 examples per parameter.

A model that is overfitting drives train down while val goes up. Both of these drive
everything up together, which is a different failure entirely.

## Three defects in our code

All fixed in `4d17aea`. Together they moved the optimum from 19.314 to 18.277 in half the
epochs — but did not remove the rise.

### 1. `known_seq_len` was sampled uniformly over the padded width

With a Wyckoff-sequence cap of 62 and a median length of 4, drawing `k` uniformly put **82%
of steps where under 1% of the data is viable**, while **88% of the reported loss comes from
k ≤ 4**, which only 8% of steps reached. The gradient signal and the metric were being
computed over almost disjoint parts of the distribution.

Fixed by `AugmentedCascadeDataset.sample_known_seq_len`, drawing `k` with p ∝
`viable_counts[k]`. 88.4% of steps now land at k ≤ 4, matching the metric's mass. Costs
0.017 ms against a ~17.7 ms step.

`alex_mp_20` has a cap of 21 and a third of the mismatch, which is why 1.5M-epoch runs
there never showed this.

### 2. The correction for that did not survive gradient clipping

A `viable_count / n_samples` rescale was meant to fix the imbalance in the loss instead. It
could not: at k ≤ 4 a per-step loss is ~3250 nats, so those gradients were clipped to norm 2
**exactly like the tail steps**, and the weighting was discarded in the normalisation.

With `lr 0.2` and `clip 2`, clipping bound on **100% of steps** — it was not a safety guard,
it *was* the learning rate, fixing every step at distance 0.4 regardless of the gradient.

Fixed by weighting in the sampling rather than the loss (`rescale_to_viable=False` in
training, still `True` in `evaluate`, so the reported metric stays comparable to earlier
runs), and by taking a per-example mean. Gradient norms moved from [1e3, 1e5] to
[0.05, 5], which is what lets a threshold sit above the working range. `clip_grad_norm` is
now optional and `grad_norm` is always logged.

### 3. `sample_viable_batch` drew with replacement

`torch.randint` where `n_viable <= batch_size` (k ≥ 13 at this cap) drew `n_viable` indices
from `n_viable` — a bootstrap resample **missing ~37% of the set it was meant to cover**, in
`evaluate` as well as in training.

Now: the whole prefix when the batch is the whole viable set, `randperm` above 1/16
coverage, `randint` only for thin slices where the finite-population correction is
negligible and permuting 4M rows would cost 2.4 ms/step.

## The residual: schedule-free's averaging window

`SGDScheduleFree` keeps `x` — the iterate it reports and checkpoints — as a weighted mean of
the raw iterates `z`:

```
x  <-  x + ckp1 * (z - x),    ckp1 = weight / weight_sum,
weight = (k+1)**r * lr_max**weight_lr_power
```

With the **default `r=0` and a constant lr the weight is constant**, so `ckp1 = 1/(k+1)` and
`x` is the *uniform* mean of every `z` since step 1 — step 1 and step ten-million carry equal
weight.

Uniform (Polyak–Ruppert) averaging assumes the iterate settles. **Mini-batch gradients never
vanish**, so `z` never stops moving, and `x` trails it by roughly `v·k/(r+2)` — unbounded in
`k`. The optimiser is working correctly; the number it reports is stale.

### The discriminator

Every mini-batched run here shows the rise. **None of the ~22 full-batch runs does.**
`AugmentedCascadeLoader` sets `batches_per_epoch = 1` when `batch_size is None`, so a
full-batch step sees the whole split, its gradient genuinely vanishes at a stationary point,
`z` settles, and the uniform average is then exactly the right thing. That is why this
appeared only on LeMat-Bulk, which is too large to train full-batch.

### A second, smaller numerical effect

The update is applied in the parameter's dtype. Once `ckp1·‖z−x‖` drops below the float32
spacing of `‖x‖` it rounds away — harmless while `(z−x)` changes sign, since the lost
increments cancel, but a one-way bias as soon as `z` is going somewhere. In a CPU toy at
this geometry, `loss(x)` rises 22× in **float64** (so the effect is algorithmic, not a
rounding artefact) and float32 adds a further 27%.

Reproduced in `src/wyckoff_transformer/tests/test_schedulefree_long_run.py` (marked `slow`,
deselected by default; run with `pytest -m slow`).

## The r=0 vs r=8 experiment

Both branches warm-started from `t1c9ehzp`'s best checkpoint with `warmup_steps=2000`,
identical apart from `r`. Planned for 6000 epochs; **stopped at 500 (8.3%)**.

| arm | epoch | lag ‖x−z‖ | rel | ‖z‖−‖x‖ | loss(x) train | loss(x) val | loss(z) |
|---|---|---|---|---|---|---|---|
| r=0 | 250 |  6.7545 | 0.0398 | 0.3582 | 18.1618 | 18.1717 | 20.7157 |
| r=0 | 500 | 12.1461 | 0.0715 | 0.6405 | 18.1648 | 18.1745 | 23.1652 |
| r=8 | 250 |  3.8990 | 0.0230 | 0.1145 | 18.1680 | 18.1787 | 23.2196 |
| r=8 | 500 |  5.4500 | 0.0321 | 0.2247 | 18.1680 | 18.1797 | 21.0118 |

### Established

**The lag grows and `r` controls it.** r=0 goes 6.75 → 12.15 (×1.80), r=8 goes 3.90 → 5.45
(×1.40); neither plateaus, and the r0/r8 ratio *widens* from 1.73 to 2.23. The branches
differ only in `r`, so this pins the lag to the averaging weight schedule rather than to the
data, the model or the sampler.

The predicted ratio was 5:1 (`1/(r+2)`). The observed 1.7–2.2 is a **lower bound**: `‖x−z‖`
also contains r-independent stochastic scatter of `z` about its own mean, which inflates both
arms and pulls the ratio toward 1.

**The lag is ~95% tangential.** At epoch 500 the displacement is 12.15 while `‖z‖−‖x‖` is
only 0.64 — the radial component is **5%**. The iterate is not inflating outward; it wanders
on a shell of roughly constant radius, and `x` trails that wandering. **Weight decay
constrains the radius and therefore does not address this.**

**Averaging is worth 2.5–5.0 nats.** `loss(z)` sits at 21–23 against `loss(x)` at 18.17, on
identical weights and batches. Averaging is not the mistake — it is carrying enormous weight.
The defect is specifically the *unbounded window*.

### Not established

**That the lag costs loss.** At epoch 500 the arms agree to 0.003 nats and r=0 — the *larger*
lag — is marginally ahead on both splits:

```
epoch 500:  train  r0 18.1648  vs  r8 18.1680   (r0 better by 0.0032)
            val    r0 18.1745  vs  r8 18.1797   (r0 better by 0.0053)
```

Consistent rather than contradictory: the production effect took ~48 000 epochs to appear
and this reached 500. A shorter window also buys less variance reduction, which costs
something immediately against a staleness that costs nothing yet. The crossover is what the
run was meant to find, and it did not get there. `r` is free in compute: 78.9 vs 79.0 minutes
for 500 epochs.

## Hypotheses eliminated

| hypothesis | how it was killed |
|---|---|
| Overfitting | Train rises too; gap ~0 or negative; 28 examples per parameter |
| LR / convergence schedule | A schedule problem gives noise, not a monotone rise — and schedule-free has no schedule to misconfigure |
| Dropout | Control vs treatment: control slope −6.66e-6 against a predicted +7.11e-6, 13 SE apart. Causally excluded (dropout had also been present since long before) |
| Biased loss estimator | Batched vs exhaustive agree to 0.00% on real val data; a 12 500 → 200 000 batch sweep spreads 0.054% (Monte Carlo noise) against a 2.1% effect; bias bounded at ~0.05% |
| Broken weighting in batch mode | The training objective is provably proportional to the metric: `E[∇train_step] = (N / 3Z)·∇metric`, `Z = Σ_k n_viable(k)` |

A W&B entity ambiguity was also found and fixed (`0384399`) — not a cause, but it had split
one comparison across two workspaces.

## Upstream

The [schedule_free README](https://github.com/facebookresearch/schedule_free) recommends
β = 0.95–0.98 for extended runs, warmup, and tuning weight decay. The production config left
**all three at library defaults**. The failure mode itself — that `r=0` with a constant lr
makes the averaging window grow without bound, so the reported iterate goes stale under any
persistent drift — is not documented.

No issue filed yet: the mechanism is solid but the harm is not demonstrated, and the r=0 arm
currently looks *better*.

## Where this leaves us

`yamls/models/lemat_bulk_ehull/ehull_adamw_wsd.yaml` (`984a1eb`, `de06d2d`) replaces
schedule-free with AdamW under a warmup-stable-decay schedule, sized for 72 hours. WSD keeps
the long constant-rate phase but reports the **final** iterate, so nothing depends on an
average keeping up with a moving target.

**The open risk this investigation hands to that config:** WSD does not average at all, so
it must earn back the 2.5–5.0 nats measured above through its lr decay collapsing the noise
ball instead. Theory says it will — that is what the decay phase is for, and the `z` measured
here was at a constant lr of 1.4 with no annealing, so it is not the right analogue for an
annealed final iterate. But it is now a checkable prediction: **if the run's loss does not
drop sharply during the final ~7 hours, this is why**, and a bounded-window weight EMA is the
insurance.

Note also that the `weight_decay: 0.01` in that config is ordinary mild regularisation. An
earlier draft justified it as bounding the weight-norm growth behind the averaging failure;
the tangential-lag measurement above shows that reasoning was wrong.

## Open questions

1. **Where is the loss crossover?** The r experiment reached 8.3% of its planned length.
2. **Is the drift endogenous?** A separable-logistic-regression CPU toy would show whether
   weight decay stops drift that the model generates itself, rather than the tangential
   wandering seen here. Never run.
3. **Does WSD's decay recover the averaging benefit?** Answered by the first 72-hour run.

## Artefacts

- Fixes: `4d17aea`; W&B entity `0384399`; AdamW/WSD config `984a1eb`, `de06d2d`
- `src/wyckoff_transformer/schedules.py` — WSD, with the horizon guard in `WyckoffTrainer.train`
- `src/wyckoff_transformer/trainer.py` — `schedule_free_lag()`, logged each validation
- Tests: `test_schedulefree_long_run.py` (slow), `test_schedule_free_lag.py`,
  `test_loss_estimator.py`, `test_viable_batch_sampling.py`, `test_grad_clipping.py`,
  `test_schedules.py`, `test_wsd_config.py`
