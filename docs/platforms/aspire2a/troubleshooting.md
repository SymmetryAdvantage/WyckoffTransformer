# ASPIRE 2A troubleshooting

Ordered roughly by how often each one bites.

---

## `.venv` vanished / is suddenly empty

You ran `uv run` or `uv sync` from a **host** shell. The host has no `python`,
and `.venv/bin/python` points into the container, so uv decides the venv is
"linked to a non-existent Python interpreter", deletes it, and makes an empty
one. Any job running at the time dies with import errors.

Rebuild: [environment.md](environment.md#never-run-uv-from-the-host). Always go
through `scripts/platforms/aspire2a/run_in_singularity.sh`.

---

## `singularity: command not found`

It is not on the default `PATH`: `module load singularity`. It prints nothing,
and puts `singularity` on `PATH` of the shell it runs in — so not inside `( ... )`
or a pipeline such as `module load singularity 2>&1 | tail`, where the change is
lost with the subshell. In a batch shell the module system may not be initialised
either; `scripts/platforms/aspire2a/run_in_singularity.sh` handles both:

```bash
if ! command -v singularity >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load singularity
fi
```

---

## The ORB checkpoint downloads at a crawl

`orb_conserv_inf` pulls `orb-v3-conservative-inf-omat-20250404.ckpt` (102 MB)
from an S3 bucket in `us-west-1` through `cached_path`. Measured from
`asp2a-gpu002`, 2026-09-08:

| Path | Throughput |
| --- | --- |
| S3, direct, one connection | 32 kB/s |
| S3, via `socks5h://asp2a-login-nus02:1081` | 24 kB/s |
| S3, direct, 8 parallel ranges | 258 kB/s aggregate |
| S3, direct, 16 parallel ranges | ~400 kB/s aggregate |
| HuggingFace, direct | 16-24 MB/s |
| HuggingFace, via the same proxy | 4.5 MB/s |

Two things follow. **The throttle is per-connection, not per-host** — streams
scale nearly linearly. And **tunnelling through a login node does not help**: the
bucket is just as slow from there, and the proxy costs ~5x on everything else,
so a login node is not a faster egress, it is the same egress. (The proxy itself
is healthy; the HuggingFace row proves that.)

Single-stream, the file takes ~53 min, and every `--workers-per-device` worker
blocks on the one `cached_path` file lock while the first crawls through it —
which is why a cold `--stage relax` looks like a deadlock rather than a
download.

The fix is parallel ranges, wrapped up in:

```bash
scripts/platforms/aspire2a/prefetch_cached_path.sh <url> [streams]
```

It fetches with N range requests, re-fetches any short stream, verifies the
result against the server's ETag, and writes the `cached_path` cache entry
(`sha256(url).sha256(etag)` plus its JSON meta) so the next `cached_path(url)`
is a hit. It is idempotent and works from a compute node.

**The orb-v3 checkpoint is already installed** (2026-09-08, ETag-verified), so
relax stages should no longer download anything. Re-run the script above for any
other checkpoint.

If an interrupted attempt left debris, it is `~/.cache/cached_path/*.tmp` plus a
stale `.lock`; the cache entry itself is only written on success.

Note this cache is on the 50 GB **home** quota — `cached_path` ignores
`XDG_CACHE_HOME`.

---

## `matminer` is broken against the resolved scipy and pymatgen

```
ImportError: cannot import name 'sph_harm' from 'scipy.special'          # scipy >= 1.17
ImportError: cannot import name '_pt_data' from 'pymatgen.core.periodic_table'
```

`matminer` 0.8.0 (Aug 2023, unmaintained) against scipy 1.18.1 and pymatgen
2026.5.4. The ASPIRE 2A build resolves fresh (no `uv.lock`), so it takes the
newest of everything — and `matminer` used to be unpinned, so the resolver
preferred pandas 3 and backtracked `matminer` to 0.8.0.

What it breaks:

| Broken | Because |
| --- | --- |
| `wyckoff_transformer.evaluation.cdvae_metrics` | imports `matminer.featurizers.site.fingerprint` |
| `formula_energy` Magpie baselines | import `matminer.utils.data.MagpieData` |
| `pytest` collection of `tests/test_trainer_cache.py` | transitively the same |
| 5 tests in `formula_energy/tests/test_experiment.py` | `MagpieData` at setup |

Training, generation, screening and relaxation are unaffected.

**Fixed** by the `matminer >=0.10.1` floor in `pyproject.toml`: 0.10.1 uses
`scipy.special.sph_harm_y` and the public `Element` API, and its own `pandas<3`
pulls pandas back to 2.3.3. The shared venv built by `build_venv.sh` on 2026-09-16 has
matminer 0.10.1 and pandas 2.3.3; the older scratch venv the pre-2026-09-16 chains run on
still has the old pair. For any other venv built before that floor, rebuild it, or move
the packages in individually — including `pytz`,
which pandas 2 needs and pandas 3 dropped, so a `--no-deps` install of just
`matminer` and `pandas` trades this failure for `Unable to import required
dependencies: pytz`. A full run is then 664 passed / 41 skipped, with nothing
deselected:

```bash
bash scripts/platforms/aspire2a/run_in_singularity.sh python -m pytest -q
```

---

## `qsub -q aiq1` is refused

All of `aiq1`–`aiq4`, `aidev` and `ailong` have `from_route_only = True`. Submit
to the router instead — `-q ai` — and let it pick from `(ngpus, walltime)`. The
map is in [usage.md](usage.md#queues).

---

## A job cannot submit to the other PBS server

The AI partition is `pbs102`; `g1`/`q*`/`glong` live on `pbs101`. A job running
in the AI partition was refused when it tried `qsub -q g1@pbs101` (observed
2026-09-05, not re-tested since), and it cannot work around it by hopping:
`ssh` from a compute node to `asp2a-login-nus02` is closed immediately
(`Connection closed by ...`, re-checked 2026-09-08).

Cross-server submission has to come from a login node. This is also why the
self-chaining scripts re-`qsub` to the **same** queue they were submitted to.

---

## `ModuleNotFoundError: No module named 'orb_models'`

The base venv build does not install the `relax` extra. See
[environment.md](environment.md#the-relax-extra-orb-and-mace).
`scripts/platforms/aspire2a/protocol_relax.pbs` stops with a pointer there rather than
installing into the shared, read-only venv. As of 2026-09-16 it is installed.

---

## `wyformer-<something>: command not found`

Only `wyformer-generate`, `wyformer-cryspr` and `wyformer-protocol` are in
`.venv/bin`; the venv predates the other entry points in `[project.scripts]`.
Run them as modules:

```bash
bash scripts/platforms/aspire2a/run_in_singularity.sh python -m wyckoff_transformer.cli.protocol_wandb --help
```

or reinstall the project — [environment.md](environment.md#adding-or-repairing-the-project-install).

---

## A tokenisation pass gets OOM-killed

`pandarallel` reads the **node's** core count, not the cgroup's — 128 on a
`gpu001`-class node against the 16 CPUs the job actually owns — and forks that
many workers into a `mem=` limit sized for 16. Pass `--n-jobs $NCPUS`, as the recipe in
[usage.md](usage.md#adding-a-dataset-or-tokeniser-to-the-cache) does. (Training jobs no
longer tokenise at all.)

---

## `python -c "..."` fails inside the container on parentheses

`singularity run` re-parses the argument vector through a shell.
`scripts/platforms/aspire2a/run_in_singularity.sh` sets `SINGULARITY_NO_EVAL=1` to stop that. If
you are invoking `singularity` directly, set it yourself.

---

## Files under `/raid` are gone

`/raid` is node-local NVMe and PBS hands each job `/raid/pbs.<jobid>`. It is not
shared between nodes and does not survive the job — a chained job's next link
usually lands on a *different* node. Anything that has to outlive one link goes
on `/scratch`.

It is also not mounted into the container automatically;
`run_in_singularity.sh` binds it explicitly.

---

## `nqstat: error while loading shared libraries: libcjson.so.1`

The site's pretty `qstat` wrapper does not work on the compute nodes. Use plain
`qstat` (`/opt/pbs/bin/qstat`), or run `nqstat` from a login node.

---

## Out of home quota

50 GB, and 12 GB of it is the container image. `~/.cache/huggingface` (5.8 GB)
and `~/.cache/cached_path` grow silently. Check with `myquota`; move what you
can under `$HOME/scratch/.cache` (which is where `XDG_CACHE_HOME` already
points for anything that honours it).

---

## Out of SUs

`myprojects` shows the project balance and `myusage` your own burn. As of
2026-09-07, project 11001786 had 1.91 M SU left of 46.3 M, expiring 2027-08-31.
A chained 24 h 1-GPU run is a real charge; check before launching one.

---

## `No such file or directory: /home/project/...` inside the container

Unlike `/scratch` and `$HOME`, `/home/project` (and `/data/projects`) is **not** mounted
automatically into Singularity containers on ASPIRE 2A.
`scripts/platforms/aspire2a/run_in_singularity.sh` binds `/home/project` and `/data/projects`
explicitly. If you invoke `singularity run` or `singularity exec` manually, you must add:

```bash
singularity run --nv --bind /home/project,/data/projects ...
```

---

## Code changes are not taking effect (another checkout's code is imported)

Every checkout shares one venv, and its editable install's `.pth` names the main
checkout's `src`, not your worktree's
([environment.md](environment.md#where-the-one-venv-actually-is)). (The venv used before
2026-09-16, and still used by chains launched from the old scratch checkout, names
`/scratch/users/nus/kna/WyckoffTransformer/src` instead.)

`scripts/platforms/aspire2a/run_in_singularity.sh` sets `PYTHONPATH=$REPO_DIR/src` inside
the container, which Python searches before the `.pth`. If running a custom script or
container invocation, put the checkout's `src` first yourself:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

A branch created before the worktree tooling was committed (2026-09-16) carries the old
`run_in_singularity.sh`, which does not set `PYTHONPATH`: merge `main` into it.
`create_worktree.sh` refuses such a branch.

---

## `train_in_pbs.sh`: `... has uncommitted changes`

At submission, the launcher refuses a checkout with modified, staged, or untracked files
that `.gitignore` does not cover; the message lists them. Commit them, or ignore what
should not be committed. The same check at the start of every link **stops the chain**
if someone left changes in a worktree with a live chain. Nothing was touched, so commit
and resubmit from that worktree to continue the pinned run.

---

## `train_in_pbs.sh`: `... is a Claude Code worktree`

Chains are refused from `.claude/worktrees/`, which Claude Code may delete when the
session ends. Commit, leave the session (keeping the worktree), then check the branch
out in a worktree of your own and submit there:

```bash
git worktree remove .claude/worktrees/<name>      # from the main checkout; the branch stays
bash scripts/platforms/aspire2a/create_worktree.sh worktree-<name>
```

See [usage.md](usage.md#working-in-a-git-worktree).

---

## `train_in_pbs.sh`: `the cached data for <dataset> ... is incomplete`

Training jobs do not build caches. Build the missing tensor cache, tokeniser or
`test.parquet` first -- [usage.md](usage.md#adding-a-dataset-or-tokeniser-to-the-cache) --
or, when only the test split is missing and no post-training evaluation is wanted, submit
with `--train-arg --no-test`.

---

## `train_in_pbs.sh`: `... pins run <id> for this config from before runs were keyed by branch`

A run pinned under the old key (`.<dataset>__<config>.runid`) exists. It was started from
some branch; the launcher does not guess whether it was this one. `--run-id <id>` continues
it from the current branch, `--fresh` starts a new run.

---

## A link stops: `... has '<branch>' checked out, but this chain was submitted from '<other>'`

Someone switched the branch of a worktree with a live chain. Check the chain's branch
out again and resubmit; or leave that worktree to the new branch and continue the chain
from a worktree on the old one with `--run-id`.

---

## `Permission denied` writing into the cache or the venv

Both are read-only on purpose ([usage.md](usage.md#the-shared-cache-and-venv-are-read-only)).
A tokeniser or install that has to write there is a deliberate step: `store_lock.sh open`
(new files into a directory) or `unlock`, then `lock` again. `rm -rf` of anything inside
fails the same way -- that is the point.

