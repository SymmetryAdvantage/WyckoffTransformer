# Datasets and what their energies mean
Whether a dataset may be used, and what each of its fields means, lives in
`yamls/datasets/<name>.yaml` -- see `docs/energy_fields.md`. The current ones are
`lemat_bulk_fmax1_stress` (and its `_ehull01` slice), `mp_20`, `mp_2026_gga_gap` and
`formula_energy`; `lemat-bulk` is a source they are built from. Every other dataset,
and any without a manifest, is obsolete: use it **only** to analyse runs already
trained on it. Training, caching and tokenising refuse it unless
`--allow-obsolete-dataset` is given, and reading one warns. Follow
`docs/lemat_bulk_pipeline.md` to rebuild the LeMat variant.

A column name does not say which energy it holds -- `energy_above_hull` is raw PBE on
LeMat and MP2020-corrected on MP -- so never compare two energies by name. Name
columns canonically (`energy_fields.canonical_id`), never after a model or tokeniser,
label every energy field in its dataset's manifest, and combine energies only through
`energy_fields.check_compatible`. Trained models record their fields' definitions in
`field_provenance.json`.

# Where data lives
Untracked datasets, `cache/`, `runs/` and W&B's directory live outside the checkout,
at locations each machine records in `~/.config/wyformer/paths.env` -- a file the
code reads itself, not variables from a shell profile. Resolve them through
`wyckoff_transformer.paths`: `data_path()` for datasets, `cache_root()`,
`runs_root()`, `resolve_store_path()`, and `wandb.init(dir=wandb_dir())`. Never use
`Path(__file__)...` or a bare relative literal, and never assume `WANDB_DIR` is in
the environment. Shell launchers use `scripts/wyformer_paths.sh`, which follows the
same rules. See `docs/data_store.md`.

A dataset's cache is the **directory** `cache/<dataset>/`, holding one Parquet
file per split. Read and write it through `wyckoff_transformer.dataset_cache`
(`dataset_cache_dir`, `load_split`, `load_cache`, `save_cache`), never by naming
a file inside it and never with `pandas.read_parquet`: the columns hold
`Element`s, a `Counter` and a `frozenset`, and the module's metadata is what
restores them. Pass `columns=` -- most callers want four of twenty. The
superseded `data.pkl.gz` is still read and never written; `docs/data_store.md`
says how to convert one. Anything that writes a cache passes `provenance(...)`
naming the options that change its contents, so `build_info` can say what built
it; a missing record means "not recorded", never "the defaults".

`data_path()` looks a dataset up by its top-level name in two places: the store
first, then `<repo>/data`. Datasets tracked by git, plainly or through LFS, stay
in the checkout; untracked ones live in the store, which is also where a new
dataset is written. There is no `data_root()` -- with two places it would be
ambiguous. Once a machine has a config file it is authoritative: a missing key
is an error, never a fallback to the checkout.

**Results belong in W&B, not in `runs/`.** Anything a run produces that is worth
keeping must be logged as a W&B artifact; new code must not make a run directory
the only place an output exists. A run directory is a local working copy that may
be absent on another machine, so fetch what you need from the run instead --
`wyckoff_transformer.cli.protocol_wandb.ensure_run_files` is the pattern, pulling
each missing file of `REQUIRED_RUN_FILES` down from W&B on demand.

**`last_checkpoint.pt` is the one file that needs care.** It holds mid-training
resume state -- optimiser, scheduler, RNG, loader position -- and
`scripts/platforms/aspire2a/train_in_pbs.sh` chains PBS jobs by resuming from it across the queue's
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

Each platform's `agent_brief.md` is the short version, and its environment
initialisation symlinks it to the gitignored `CLAUDE.local.md` at the checkout root,
which Claude Code and Gemini CLI / Antigravity (via `.gemini/settings.json`) load
into every session. If a session starts without it, run
`scripts/platforms/link_agent_brief.sh <platform>`. See `docs/platforms/README.md`.

## Shell commands in agent worktrees
A Claude Code session isolated in a git worktree refuses any shell command it cannot
statically show stays out of another checkout's git. It does not run the command at
all. What trips it, none of which needs to involve git:
 - `source` or `.` of a file -- so `source .venv/bin/activate`;
 - a program named by a variable -- `PY=.venv/bin/python; $PY -c ...`;
 - a wrapper such as `time` or `env` given variable arguments;
 - `cd` to a computed path, e.g. `cd "$(dirname ...)"`, in a command that also runs git.

Write program names and paths literally, and run from the worktree itself -- never
`cd` to the main checkout, whose venv imports the main checkout's code. Call the
checkout's interpreter by path, or the launcher your platform documents; which one
applies is host-specific and lives in `docs/platforms/<platform>/`.

# Project conventions
This is a research project, which means that ideas will be tried and discarded. As such, it is extremely important to avoid confusion. In particular:
1. When describing the results of an experiment, note the date, git commit hash and applicable WanDB runs
2. Files in docs/archive/ are not maintained - there is no need to update them, but they might contradict the state of the code base and later findings
3. In general, files in docs/*.md are supposed to be current - correct them if you notice a contradiction
4. In general, files in yamls/ are supposed to be immutable to preserve reproducibility and data-cache-config-run correspondence. They can be edited if asked explicitly or to fix a bug
5. By default, run with WanDB online logging enabled - unless there is a specific reason not to, such as avoiding overwriting
