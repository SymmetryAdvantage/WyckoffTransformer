# LeMat-Bulk datasets
`lemat_bulk_fmax1_stress` is the only current variant. Use it for new work, and follow
`docs/lemat_bulk_pipeline.md` to rebuild any of it.

Every other LeMat variant in `data/` and `cache/` — `lemat_bulk_fmax1`,
`lemat_bulk_ehull`, `lemat_bulk_ehull_wp20`, and the `_pilot` subsamples — is superseded.
Use them **only** to analyse runs already trained on them, never for new training,
evaluation or caching. They predate `scripts/recover_mp_forces.py`, so they either drop
the 30,679 Materials Project rows whose archived forces are empty (a `max_force <= X` cut
drops NaN for every X) or carry a fabricated `max_force` of 0.0415 against a true median
of 0.088. `python scripts/audit_lemat_variants.py` reports what any variant on disk was
built with.

# Where data lives
Untracked datasets, `cache/`, `runs/` and W&B's directory live outside the checkout,
at locations each machine records in `~/.config/wyformer/paths.env` -- a file the
code reads itself, not variables from a shell profile. Resolve them through
`wyckoff_transformer.paths`: `data_path()` for datasets, `cache_root()`,
`runs_root()`, `resolve_store_path()`, and `wandb.init(dir=wandb_dir())`. Never use
`Path(__file__)...` or a bare relative literal, and never assume `WANDB_DIR` is in
the environment. Shell launchers use `scripts/wyformer_paths.sh`, which follows the
same rules. See `docs/data_store.md`.

`data_path()` looks a dataset up by its top-level name in two places: the store
first, then `<repo>/data`. Datasets tracked by git, plainly or through LFS, stay
in the checkout; untracked ones live in the store, which is also where a new
dataset is written. There is no `data_root()` -- with two places it would be
ambiguous. Once a machine has a config file it is authoritative: a missing key
is an error, never a fallback to the checkout.

The data store and the cache are replicated across machines by
`scripts/store_sync.sh`; tracked datasets travel by git.

`runs/` is **not**. It is per-machine working output, shared between the worktrees
on one machine and, on a cluster, placed on scratch.

**Results belong in W&B, not in `runs/`.** Anything a run produces that is worth
keeping must be logged as a W&B artifact; new code must not make a run directory
the only place an output exists. A run directory is a local working copy that may
be absent on another machine, so fetch what you need from the run instead --
`wyckoff_transformer.cli.protocol_wandb.ensure_run_files` is the pattern, pulling
each missing file of `REQUIRED_RUN_FILES` down from W&B on demand.

**`last_checkpoint.pt` is the one file that needs care.** It holds mid-training
resume state -- optimiser, scheduler, RNG, loader position -- and
`scripts/train_in_pb.sh` chains PBS jobs by resuming from it across the queue's
24 h ceiling. It used to exist in exactly one place, so a purge of `runs/`
silently restarted multi-day chains from epoch 0. It is now mirrored into the
run's W&B **files** (`WyckoffTrainer.mirror_checkpoint_to_wandb`, rate-limited by
`CHECKPOINT_MIRROR_INTERVAL_S`), and a resume that finds no local copy downloads
it (`trainer.restore_checkpoint_from_wandb`). The launchers ask
`python -m wyckoff_transformer.cli.resume_probe <run-id>` before treating an id
as spent, and refuse to start over when they cannot get an answer.

Two rules follow. **Never delete a run directory to work around a missing
checkpoint** -- move it aside; it can hold the only local `best_model_params.pt`.
And if you add anything else that lives only under `runs/`, either log it to W&B
or expect to lose it: nothing else in a run directory is recoverable.

# Running code
Just running system Python won't work and can destroy the virtual environment! You must follow the machine-specific instructions! WyFormer runs on a range of machines (workstations, clusters, CI). Anything that is true for **one** host only must not leak into the top-level `README.md`, `AGENTS.md`
or `scripts/`:
 - human/agent-readable setup & run instructions -> `docs/platforms/<platform>/`,
   split by use case (e.g. `usage.md`, `environment.md`, `troubleshooting.md`)
 - scripts (container builds, env init, scheduler submission) -> `scripts/platforms/<platform>/`

`<platform>` is the short host or cluster name, e.g. `luna`, `zeus`.

Before setting up an environment, check whether `docs/platforms/<platform>/` exists
and follow it instead of improvising. When you learn something host-specific, record
it there rather than in the generic docs.
