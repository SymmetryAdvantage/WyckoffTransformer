# The data store

WyFormer's bulky inputs and outputs live outside any git checkout, in one place
per machine, so that every worktree on that machine shares them and none of them
has to be hand-linked.

| What | Where | Replicated? |
| --- | --- | --- |
| Datasets tracked by git, plainly or through LFS | `<repo>/data`, via git | by git |
| Untracked datasets | the **data store**, `WYFORMER_DATA` | `scripts/store_sync.sh` |
| Tokenised tensors and Wyckoff records | the **cache**, `WYFORMER_CACHE` | `scripts/store_sync.sh` |
| Run directories | `WYFORMER_RUNS` | no -- per machine |
| W&B's local run directories | `WANDB_DIR` | no -- per machine |

## Configuring a machine

Each machine records where these are in one file, which the code reads itself:

```
# ${XDG_CONFIG_HOME:-~/.config}/wyformer/paths.env
WYFORMER_DATA=/home/kna/.local/share/wyformer/data
WYFORMER_CACHE=/home/kna/.local/share/wyformer/cache
WYFORMER_RUNS=/home/kna/.local/share/wyformer/runs
WANDB_DIR=/home/kna/.local/share/wyformer
```

The format is docker's `--env-file`: `KEY=VALUE` per line, `#` comments, and
**nothing is expanded**. `$HOME`, `~` and relative paths are rejected with an
error rather than half-supported, because docker would pass them through
literally. The same file therefore serves Python, the shell launchers, and
`docker run --env-file`. No `export`, no quotes.

The values for a host live in `docs/platforms/<host>/environment.md`, not here:
one machine's filesystem says nothing about another's.

Check what a machine resolves to, and where each value came from:

```bash
python -m wyckoff_transformer.paths
```

### Why a file, and not variables in a shell profile

A profile only reaches the shells that source it. cron, Jupyter kernels, IDE run
configurations, batch schedulers and containers each miss it in their own way --
and Ubuntu's stock `~/.bashrc` returns before its end for non-interactive shells,
so `bash -lc` never sees an `export` appended to it. A process that missed the
variables used to fall back to the checkout without a word, and write a fresh
`cache/` or `runs/` there beside the real one. A file at a fixed path is found
however the process was started.

### Resolution

For each key:

1. **The environment variable**, if set. For containers, CI, tests and one-off
   overrides.
2. **The config file.** When it exists it is authoritative: a key it does not set
   is an error, **never** a fallback to the checkout.
3. **Only when there is no config file at all**: `<cwd>/<name>` if that directory
   exists, else `<repo>/<name>`, `<repo>` being found by searching upward from the
   working directory for `pyproject.toml`. This is what a fresh clone of the
   public repository, and CI, run on.

With no config file and no `WANDB_DIR`, wandb keeps its own default, the working
directory.

`data/`, `cache/` and `runs/` are not shipped in the wheel, so a non-editable
install has no step 3 and needs the file.

## Three kinds of data, two places

`data/` holds three kinds of file, and they are replicated differently:

| Kind | Examples | Lives in |
| --- | --- | --- |
| Tracked by git | `mp_20`, `perov_5`, `carbon_24`, `mpts_52`, `wbm`, and the `matbench_discovery_*` alias symlinks | `<repo>/data` |
| Tracked by git LFS | `alex_mp_20/{train,val}.csv.gz` | `<repo>/data` -- needs `git lfs` installed, or you get 130-byte pointer files |
| Not in git | `lemat-bulk`, `lemat_bulk_fmax1*`, `formula_energy`, `unique_fingerprints.parquet` | the data store |

The first two are versioned and already on every machine that has the
repository, so copying them into the store would only create a second,
unversioned copy that goes stale the first time one changes. Only the third kind
lives in the store.

Code looks a dataset up **by its top-level name, store first, then the
repository**:

- `data_path("mp_20", "train.csv")` finds `mp_20` in the checkout.
- `data_path("lemat-bulk", "lemat_pbe.csv.gz")` finds `lemat-bulk` in the store.
- A dataset in neither resolves to the store -- where it will be written.

The whole dataset comes from one place. Resolving per file instead would let a
split mix a store copy of `train` with a repository copy of `test`. The cost is
that a directory holding both kinds -- today only `mp_2026_gga_gap`, whose README
is tracked and whose CSVs are not -- resolves to the store as a whole, so the
store's copy must be complete for anything the code reads. `python -m
wyckoff_transformer.paths` lists every name present in both places.

## The dataset cache: one Parquet file per split

`cache/<dataset>/` holds the Wyckoff records as `train.parquet`, `val.parquet`
and `test.parquet`, beside the tokenised `tensors/`, the `tokenisers/` and
whatever else is derived from them. **The cache is the directory**, not a file
inside it.

Read and write it through `wyckoff_transformer.dataset_cache`, never with
`pandas.read_parquet` directly -- a column holds `pymatgen` `Element`s, a
`Counter` and a `frozenset` of augmentation variants, and the module's metadata
is what turns the stored strings, maps and nested lists back into them:

```python
from wyckoff_transformer.dataset_cache import (
    dataset_cache_dir, iter_splits, load_cache, load_split, save_cache)

load_cache(dataset_cache_dir("mp_20"))                        # every split
load_split("cache/lemat_bulk_fmax1_stress", "test")           # one split
load_split(cache, "train", columns=("elements", "spacegroup_number"))
save_cache({"train": frame}, dataset_cache_dir("new_dataset"))

for split, frame in iter_splits(cache, columns=("elements",)):  # split by split
    ...
```

`iter_splits` is what a loop over splits should use when it keeps only a
summary. It holds one frame at a time, and on a cache still in the superseded
format it reads that file once rather than once per split.

### What built it

Each split records what wrote it, in its own Parquet schema metadata:

```json
{"tool": "wyformer-cache-dataset", "built": "2026-09-22T09:48:53+00:00",
 "version": "1.0.7", "commit": "773726d...", "dirty": true,
 "options": {"max_sites": 61, "symmetry_precision": 0.1, "symmetry_a_tol": 5.0,
             "sort_by_letter": true, "scalar_columns": null,
             "observed_gene_minimum_over": ["train", "val", "test"]}}
```

```python
from wyckoff_transformer.dataset_cache import build_info, provenance

build_info("cache/lemat_bulk_fmax1_stress")            # every split
build_info(cache, "train")                              # one
save_cache(frames, cache, provenance("my-tool", cutoff=0.1))   # writing one
```

`options` holds what changes the **contents** — a cap, a tolerance, an ordering —
not what only changes how long the build takes, such as the worker count. Two
caches with the same options hold the same thing.

It lives in the file rather than a `build.json` beside it for three reasons: a
split copied to another machine takes its provenance with it; splits written at
different times each say so, which matters because `slice_dataset_by_ehull.py`
and `migrate_cache_to_parquet.py` write them one at a time; and anything that
opens the Parquet can read it.

A cache built from a dataset with a manifest (`yamls/datasets/`) also records,
as `options.manifest` and `options.fields`, what the manifest said each of its
columns meant; loading it for a model refuses a cache whose manifest has since
changed those definitions. See [energy_fields.md](energy_fields.md).

`build_info` returns `None` where there is no record — a split written before
this existed, one still in the superseded format, or one converted from it,
whose original build options are in the pickle nowhere and are genuinely not
knowable. **`None` means "not recorded", never "built with the defaults".**

Two options record themselves in the data and so are not the point of this:
`--observed-gene-minimum-target` adds the `gene_min_formation_energy_per_atom`
column, present exactly when it was passed; `--max-sites` bounds the longest
row. What the column cannot say is *which splits the minimum was taken over* —
it spans every split present at cache time, so the same flag over two splits and
over three gives a different target under the same name. That is why
`observed_gene_minimum_over` names them rather than being a boolean.

A training run logs the record into its W&B config under `dataset_cache`, beside
`code` and `distributed` and never in `config`, which a resume has to match
exactly (`trainer.log_dataset_cache_provenance`). So "which cache did this run
see" is answerable from the run, not only from the machine that built it. It is
best-effort: a machine that holds `tensors/` but no split files -- a normal way
to train on a cluster -- logs `null` and a warning rather than failing the run.

`dirty` says the checkout had uncommitted changes, so `commit` does not fully
describe the code that ran. It is recorded rather than refused, unlike a
training run: rebuilding a cache is not a result, and a six-hour job should not
die over an unstaged file.

`columns=` is worth passing. Parquet reads only the bytes those columns occupy:
four gene columns of LeMat-Bulk's training split take 0.1 s against 3 s for the
whole 5.1M-row frame.

### The format it replaced

Until 2026-09-22 a cache was one `data.pkl.gz`, a pickle of
`{split: DataFrame}`. Every reader still **falls back** to it, so no cache has
to be converted to keep working, but nothing writes it any more. Three reasons:

* `pickle.load` runs whatever the file says, and caches are copied between
  machines, pulled from the store and shared with collaborators.
* It was slow and all-or-nothing: `lemat_bulk_fmax1_stress` took 83 s to
  unpickle 5.3M rows, and a caller that wanted one split paid all of it --
  the post-training evaluation loaded 399 MB to read `test`. The same data as
  Parquet reads in 3 s, one split in 0.3 s, a few columns in 0.1 s.
* Nothing outside this repository could read it. Parquet opens in pandas,
  polars, DuckDB and Arrow.

It costs about 8% more disk: the energy columns are float64 noise, which
compresses no better here than in gzip.

Convert a cache built before the change:

```bash
python scripts/migrate_cache_to_parquet.py lemat_bulk_fmax1_stress   # or --all
```

It keeps the pickle -- a job on another machine may still be reading it, and
nothing here can rebuild it. Delete it by hand once every machine sharing the
store has been converted.

One consequence is worth knowing before re-tokenising a converted cache. Caches
built before the augmentation audit hold `sites_enumeration_augmented` as a
`frozenset`, which has no order; Parquet writes it sorted, so the variants come
back in a different order than the pickle gave them. Nothing a run measures
changes: training draws a variant uniformly at random, both
`record_to_augmented_fingerprint` and `gene_key` are order-independent by
construction, and the token ids are not affected either, since
`Tokenizer.from_token_set` numbers `sorted(all_tokens)`. But the tokenised
tensors are not bit-identical to ones built from the pickle. An existing
`tensors/` file is untouched.

The derived sets beside the records -- `gene_fingerprints.pkl.gz`,
`gene_ehull_index.pkl.gz` -- are still pickles. They are a set of tuples and a
dict, not a table, and Parquet is the wrong shape for them.

## Finding things from code

Never build one of these paths by hand. Use `wyckoff_transformer.paths`:

```python
from wyckoff_transformer.paths import (
    cache_root, data_glob, data_path, data_store, resolve_store_path, runs_root, wandb_dir)
from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split

data_path("mp_20", "train.csv")          # a dataset, looked up in both places
data_glob("lemat_bulk*")                 # enumerate datasets across both places
data_store()                             # the store itself, for tooling
dataset_cache_dir("mp_20")               # a dataset's cache; never cache_root() / name
load_split("cache/mp_20", "test")        # and never open a split's file yourself
runs_root() / run_id
wandb.init(dir=wandb_dir(), ...)         # never rely on WANDB_DIR being inherited
resolve_store_path(args.reference)       # re-root a `data/`, `cache/` or `runs/` default
```

There is deliberately no `data_root()`: with two places, "the data directory" is
ambiguous, and a single root that silently meant the store would miss every
tracked dataset.

### Defaults that name a store path

Keep the readable literal and resolve it where it is used, not where it is
defined:

```python
DEFAULT_REFERENCE = Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz")

def load_reference(path: Path = DEFAULT_REFERENCE) -> pd.DataFrame:
    path = resolve_store_path(path)     # <- in the body
    ...
```

Resolving in the constant would resolve at *import* time, so importing the
package anywhere without a store would raise -- breaking test collection for the
whole repository. The literal also keeps `--help` readable, and a caller can pass
an absolute path, which `resolve_store_path` returns untouched.

### Why not `__file__`

Paths of the form `Path(__file__).resolve().parents[2] / "cache"` were the
previous convention. Worktrees share one venv whose editable install points at
the main checkout, so such a path resolves into the main checkout whichever
worktree the process runs in. Package resources such as
`wyckoff_transformer/evaluation/data` are unrelated and correctly
`__file__`-relative, as are the caches at `.cache/` (run-local hull caches) and
`~/.cache/wyckoff_transformer/` (downloaded model weights).

## Shell launchers

`scripts/wyformer_paths.sh` is the shell half, with the same file, precedence and
refusals; a test holds the two to agreement. Source it, then:

```bash
. "$REPO/scripts/wyformer_paths.sh"
RUNS_DIR=$(wyformer_path WYFORMER_RUNS "$REPO/runs") || exit 1
```

`scripts/platforms/aspire2a/train_in_pbs.sh` resolves the locations **once, at submission**, writes
them into the job spec, and exports them in every link. A chain therefore uses
the same directories throughout, even if the config is edited while it is
queued. Exporting matters for containers: the environment tier wins, so the Python
inside uses exactly what the launcher resolved.

## Containers

A container usually has its own `$HOME` and cannot see the host's
`~/.config/wyformer/paths.env`. Give it the environment tier instead, and mount
each location **at its host path** so the values mean the same inside:

```bash
docker run --env-file ~/.config/wyformer/paths.env \
    --volume /home/kna/.local/share/wyformer:/home/kna/.local/share/wyformer ...
```

The platform's own launcher is responsible for this -- see
`docs/platforms/<host>/`.

## Worktrees

Nothing to do. The config file is per machine, so every worktree resolves the
same store, cache, runs and W&B directory. Do not hand-make `data/lemat-bulk`
symlinks in a worktree: that recipe has to be repeated for every dataset ever
added, and it drifts.

Nothing outside this repository should rely on the old in-checkout paths either;
there are no compatibility symlinks.

## Syncing between machines

```bash
scripts/store_sync.sh <remote> [pull|push] [--go] [subpath ...]
```

It pulls and only reports unless told otherwise; `--go` transfers. Subpaths are
store-relative (`data/lemat-bulk`, `cache/mp_20`); with none, both stores are
synced. Only the stores move: tracked datasets travel by git, and `runs` stays
per machine. Each side's locations come from that machine's own `paths.env` --
the remote's is read with a plain `ssh <remote> cat`, so nothing depends on which
shell profile the remote sources.

Both sides are reachable, so the dry run **is** the status command: it lists
exactly what differs, with no manifest to maintain.

Addresses belong in `~/.ssh/config`, so a command names a host rather than an
address; rsync cannot take a raw IPv6 address in `host:path` at all. In
`ssh_config`, `%` is an expansion character, so a link-local scope is written
`%%`:

```
Host iapetus
    HostName fe80::ae9e:17ff:fef6:ecbb%%eno1
    User kna
```

### Flags, and why

- `-a` and never `-L`. `-a` keeps symlinks as symlinks; `-L` would dereference
  them and inflate the transfer.
- `-W`, no `-z`. The files are overwhelmingly large, compressed or binary, so a
  rebuilt file differs in every block. Delta transfer and compression cannot win
  and only cost a full read of both sides.
- `--update`. Never overwrites a file that is newer on the receiver. The store is
  not versioned, so a two-sided diff cannot say which side is right; this, plus
  pull-by-default and dry-run-by-default, is the entire safety story. It also
  means a *stale* file with a newer timestamp is skipped -- read the dry run.
- `--backup --backup-dir`. An overwrite stays recoverable. Prune it deliberately.
- `--partial-dir`, to resume a dropped transfer. It is **mutually exclusive**
  with `--inplace` (`rsync: --inplace cannot be used with --partial-dir`), and
  needs headroom for partial plus final copy of the largest file. Swap in
  `--inplace` when the receiving filesystem is tight, and `--append-verify` to
  resume one large file across a dropped link.
- No `--delete`, ever, by default: it turns a wrong-direction sync into data
  loss.

## `runs/` — per machine, not replicated

`runs_root()` resolves like the cache. It is deliberately **not** part of the
replicated store, and `scripts/store_sync.sh` rejects any subpath but `data/...`
and `cache/...`, so it cannot be synced by accident. One `WYFORMER_RUNS` per
machine means a checkpoint written from one worktree is visible from all of
them; on a cluster, point it at scratch.

### W&B holds the results — but not the resume state

Two halves, and conflating them is how you lose a week of training.

**Results should live in W&B.** Anything a run produces that is worth keeping is
logged as an artifact, so a run directory is a local working copy that may simply
be absent on another machine. Fetch what you need from the run:
`wyckoff_transformer.cli.protocol_wandb.ensure_run_files` is the pattern — given a
run id it downloads each missing file of `REQUIRED_RUN_FILES`
(`best_model_params.pt`, `wyckoff_processor.json`, `spacegroup_distribution.json`)
into the run directory. New code must not make a run directory the only place an
output exists.

The same applies to derived data a run needs to be *sampled*, not only to the
weights. A `chemical_system_conditioning` run cannot generate anything without the
(chemical system, space group) prior built from the tensor cache it trained on, so
`WyckoffTrainer.save_system_prior` writes `system_prior.npz` into the run directory,
mirrors it into the run's W&B files and logs it once as an artifact; the optional
counterpart of `ensure_run_files` is `ensure_system_prior`. Before that existed, the
prior was nowhere at all and the run was unsamplable on any machine without the 6.5 GB
cache — see `docs/chemical_system_sampler.md`.

**`last_checkpoint.pt` needs its own mechanism**, and it has one. It holds
mid-training resume state — optimiser, scheduler, RNG, loader position — and
`scripts/platforms/aspire2a/train_in_pbs.sh` chains PBS jobs by resuming from it across the queue's
24 h ceiling. It used to be the one part of a run that existed in exactly one
place, and a purge of `runs/` restarted a multi-day chain from epoch 0 without an
error. Three pieces close that:

1. **Mirror.** `save_training_checkpoint` copies the file into the run's W&B
   **files** — one slot that later uploads overwrite, not an artifact, which
   would mint a new version of a 1–40 MB file on every checkpoint. Throttled in
   time (`CHECKPOINT_MIRROR_INTERVAL_S`, 30 min), which bounds both the upload
   traffic and what a purge can cost; the save that ends the training loop always
   mirrors. Best-effort: a network failure never costs the epoch just trained.
2. **Recover.** A `--resume` that finds no local checkpoint calls
   `restore_checkpoint_from_wandb`. It returns `None` only when the run genuinely
   has no mirror; a listed file that cannot be fetched is an error, never a
   silent restart. Absence is read off the file's size, because W&B reports a
   never-uploaded file as a zero-sized entry rather than an error.
3. **Ask before giving up.** The launchers run
   `python -m wyckoff_transformer.cli.resume_probe <run-id>` before treating an id
   as spent: exit 0 means resume state exists somewhere, 1 that there is none
   anywhere, 2 that the question could not be answered. On 2 they refuse to
   proceed. They move a spent run directory aside instead of deleting it, since it
   can hold the only local `best_model_params.pt`.

Two limits remain: a run whose last checkpoint predates the mirror has none, so a
purge is still unrecoverable for it; and with `--offline` the probe cannot reach
W&B, so the launcher warns and treats the id as spent.

## `wandb/`

The five `wandb.init` calls pass `dir=wandb_dir()` explicitly, so W&B's directory
does not depend on `WANDB_DIR` being inherited. Once a config file exists,
`WANDB_DIR` is required in it like the other keys -- otherwise wandb would quietly
write `wandb/` into whatever directory the process ran in. Note the trailing
`wandb` component is wandb's: `WANDB_DIR=/home/kna/.local/share/wyformer` gives
`/home/kna/.local/share/wyformer/wandb/`. wandb creates it if missing.

Code that calls `wandb.init()` *without* `dir=` only follows `WANDB_DIR` from the
environment -- wandb offers no file-based alternative: `root_dir` and `dir` set in
its global settings file, `~/.config/wandb/settings` (which it does read), are
ignored. Do **not** export `WANDB_DIR` machine-wide from a shell profile to cover
such code: a shared machine runs other projects too, and every one of them would
then write into WyFormer's W&B directory. New WyFormer code passes
`dir=wandb_dir()` instead.

Related variables, which this repository leaves at their defaults:

| Variable | Default | What it holds |
| --- | --- | --- |
| `WANDB_CACHE_DIR` | `~/.cache/wandb` | artifact cache; already XDG-correct |
| `WANDB_DATA_DIR` | `~/.local/share/wandb` | artifact staging before upload |
| `WANDB_ARTIFACT_DIR` | `./artifacts` | downloaded artifacts — relative to the working directory |

Sharing one W&B directory between worktrees is safe: run directories are named
`run-<timestamp>-<id>` and cannot collide, and two concurrent runs writing into
one directory were tested. The only change is that `wandb/latest-run`,
`debug.log` and `debug-internal.log` -- symlinks wandb rewrites on every
`wandb.init` -- point at the last run started anywhere on the machine, not in
this worktree. Don't build anything on them.

## What is not in the store

- `generated/` -- 20 GB across 1.3M files, the opposite transfer profile, and
  largely an archive of old experiments, part of which is private. It has its
  own ad-hoc handling; the new evaluation protocol pushes structures to W&B
  instead.
- `runs/` and W&B's directory -- per-machine working output; see above.
- `.cache/` and `~/.cache/wyckoff_transformer/` -- run-local caches and
  downloaded model weights, refetched per machine.

## Provenance

`data/README.md` records where each dataset came from: the HuggingFace repo and
revision, or the script that builds it. That is documentation for the case where
every copy is lost -- it is not how a machine gets its data. Transferring beats
rebuilding here: the internet is the slow link, and CPU is scarce (iapetus has
little, ASPIRE2a has a slow filesystem and needs PBS for preprocessing).
