# Training on several GPUs of one node

`scripts/train.py` trains with PyTorch DistributedDataParallel (DDP) when it is
launched by `torchrun`: one process ("rank") per GPU, each holding a full copy
of the model and of the dataset. Run without `torchrun`, nothing changes.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 \
    scripts/train.py <model.yaml> <dataset> cuda
```

- Pass the device *type*, `cuda`. Each rank takes `cuda:LOCAL_RANK` of the cards
  `CUDA_VISIBLE_DEVICES` shows; naming a card (`cuda:1`) is refused.
- The backend is NCCL on CUDA and gloo on the CPU; `--dist-backend` overrides it.
- `torchrun` is also `python -m torch.distributed.run`, which is the spelling to
  use where the platform launcher resolves only `python` against the venv.
- Every other flag works as it does in one process, including `--resume`,
  `--reschedule`, `--pilot` and `--production`.

How to spell this on a given host is in `docs/platforms/<platform>/`. Only
iapetus has been checked so far ([usage](platforms/iapetus/usage.md#multi-gpu-training)).

## What a config means on N GPUs

**`train_batch_size` is the global batch.** Each step, every rank draws the same
global batch and keeps its own `train_batch_size / N` share. The loss each rank
backpropagates is weighted by its share, so the gradient DDP averages is the
gradient of the mean over the whole global batch. The same config therefore
trains the same run on any number of GPUs: the learning rate,
`lr_per_sqrt_n_samples`, `batches_per_epoch` and a step-indexed schedule's
`total_steps` are unchanged. The GPU count is not recorded in `config.yaml`,
which a resume is held to; it goes to the W&B config as `distributed.world_size`,
next to `code`.

To use more GPUs for a *larger* batch, write a new config with a larger
`train_batch_size`, as you would for one GPU. What N GPUs buy is per-card memory
(a card holds `train_batch_size / N` examples) and, where compute dominates the
step, time (see [Performance](#performance)).

`train_batch_size` must be set and divisible by N. Full-batch training is refused
because every rank would compute the same step.

### What the ranks share and what they do not

| Shared by every rank | Per rank |
| --- | --- |
| Weights: DDP broadcasts rank 0's at start and averages gradients every step | Order permutations of the multiclass target |
| The step's `known_seq_len` and head, from a `random.Random` seeded identically | The augmentation draw |
| The global batch's indices, from a CPU `torch.Generator` seeded identically | Evaluation batches of `val` and `test` |
| Validation losses, averaged over ranks, so early stopping and `ReduceLROnPlateau` agree | |

A NextToken step trains one prediction head, and a head no rank uses gets no
gradient. That is why the step's shape is drawn from a stream every rank
shares, and why DDP runs with `find_unused_parameters=True`.

When the global batch is a whole viable set (a `known_seq_len` reached by at most
`train_batch_size` examples), the set is split as evenly as it goes. A rank left
with nothing runs one stand-in example at zero weight so that it still joins the
backward pass. The loss weighting above keeps both cases exact.
`test_distributed_training.py` checks the gradients against a single-process
reference, uneven and empty shares included.

### Evaluation, logging and files

- Every rank evaluates on its own draws, and the losses are averaged. On N GPUs
  a validation estimate therefore averages N times as many sampled batches.
- Rank 0 alone opens the W&B run. The other ranks run a disabled W&B run with the
  same id, which is how they find the run directory. Rank 0 alone writes
  `best_model_params.pt`, the checkpoint and the artifacts, and generates
  structures after training. `loss.batch.train` is averaged over ranks before it
  is logged.
- A crash on any rank ends the job: `torchrun` stops the others.

## Supported targets

| Target | Distributed |
| --- | --- |
| `NextToken` with `multiclass_next_token_with_order_permutation` | yes; gradients and resume tested, trained on two K20c |
| `Scalar` with `scalar_loss: mse` | yes; gradients tested |
| `Scalar` with `scalar_loss: censored` | implemented, untested: the criterion's own parameters are broadcast at start and their gradients averaged |
| `NumUniqueTokens` | implemented, untested |
| `NextToken` without order permutation | refused: it trains on the whole split every step |

`compile_model: true` with DDP has not been tried. DDP wraps the compiled module,
which works but is the reverse of the order PyTorch recommends. Nor has more than one node been
tried: `torchrun --standalone` is single-node by construction.

## Resuming

`--resume` works under `torchrun` as it does in one process. The checkpoint keeps
the shared state once, as before: weights, optimiser, schedule, the step stream
under `rng.python`, and the training loader's generator. It adds `per_rank`, each
rank's torch and CUDA RNG and its `val`/`test` loader positions. Checkpoint
format 1 is unchanged, and a single process ignores the new keys.

- **Same number of GPUs:** the run continues exactly. In
  `TestDistributedResume`, a two-rank gloo run on the CPU that crashed and
  resumed ends with weights bit-identical to one that never stopped.
- **A different number, or a checkpoint from one process:** the shared state
  restores exactly and the per-rank streams are reseeded, with a warning. The
  continuation is statistically the same run, not the same bits.
- A chain may change its GPU count between links; `config.yaml` does not record it.

## Performance

Measured on iapetus, 2026-09-17, two Tesla K20c, NCCL 2.23.4, with the change that
introduced this page (parent commit `becaccc`). The model is
`yamls/models/mp_20/NextToken/distributed/ddp_smoke.yaml` on `mp_20`, and "4× wider"
multiplies its embedding and feed-forward sizes by 4. The figures are seconds per
training epoch, excluding evaluation, averaged over 3 epochs after a warm-up. These
are development runs with W&B disabled; they are not in W&B.

| Model | Global batch | 1 GPU | 2 GPUs | Peak memory per card, 1 → 2 GPUs |
| --- | --- | --- | --- | --- |
| smoke, 90k params | 512 | 1.15 s | 1.50 s | 107 → 82 MiB |
| smoke | 4096 | 0.28 s | 0.19 s | 266 → 155 MiB |
| 4× wider, 735k params | 512 | 1.30 s | 1.42 s | 237 → 145 MiB |
| 4× wider | 4096 | 0.57 s | 0.36 s | 828 → 450 MiB |

DDP adds a fixed cost of about 6 ms per step on these cards. That outweighs the
saving when a step is already ~20 ms of mostly Python and kernel launches. Where
the cost goes has not been profiled. The candidates are the gradient all-reduce,
`find_unused_parameters` walking the autograd graph, the extra all-reduce of the
logged loss, and moving the shard's indices to the card. Expect a speed-up only when
the per-card compute dominates, i.e. larger batches or models. Expect a
memory saving always.

End to end, the smoke config's 16 epochs, evaluation included, reached a best
validation loss of 25.09 on one K20c and 24.79 on two, in 25 s and 26 s of
training (W&B offline runs `2lh51za3` and `sgfm9sxf`, not synced).
