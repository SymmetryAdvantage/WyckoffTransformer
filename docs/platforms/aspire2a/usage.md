# Running WyFormer on ASPIRE 2A

Everything here assumes the environment is already built. If it is not, or you
broke it, read [environment.md](environment.md) first.

```bash
cd /home/users/nus/kna/scratch/WyFormer/worktrees/<name>   # an experiment's worktree
# or the main checkout, which merges experiments rather than hosting them:
# cd /home/project/11001786/WyFormer/WyckoffTransformer
```

Several experiments -- each a branch, each in its own worktree -- are developed and
trained at once. What keeps them apart, and what keeps them from damaging the shared
cache and venv, is in [Working in a git worktree](#working-in-a-git-worktree) and
[The shared cache and venv are read-only](#the-shared-cache-and-venv-are-read-only).

---

## Queues

PBS on ASPIRE 2A is **two servers**, and they do not see each other's queues:

| Server | Queues | What |
| --- | --- | --- |
| `pbs102` | `ai` (router) -> `aiq1`-`aiq4`, `aidev`, `ailong` | the A100 AI partition |
| `pbs101` | `normal` -> `q1..q6`, `qlong`, `g1..g4`, `glong`, `gdev`, ... | the general CPU/GPU partition |

**Every AI queue is `from_route_only`.** `qsub -q aiq1` is rejected. You submit
to the router `-q ai` and it picks the destination from `(ngpus, walltime)`:

| Destination | ngpus | walltime | `max_run` per user | Priority |
| --- | --- | --- | --- | --- |
| `aidev` | 1–4 | <= 02:00:00 | 2 | 100 |
| `aiq1` | 1 | 02:00:01 – 24:00:00 | 5 | 10 |
| `aiq2` | 2–3 | 02:00:01 – 24:00:00 | 5 | 10 |
| `aiq3` | 4 | 02:00:01 – 24:00:00 | 25 | 20 |
| `aiq4` | >= 5 | 02:00:01 – 24:00:00 | 5 | 30 |
| `ailong` | 1–4 | 24:00:01 – 120:00:00 | **1** | 10 |

So a normal WyFormer training job — `-q ai`, `select=1:ngpus=1`,
`walltime=23:59:59` — lands in `aiq1`, where you may have five at once.

(`aiq5` and `aindustry` also exist on `pbs102` but are not router destinations.)

`g1` on `pbs101` is an equivalent <= 24 h single-GPU fallback
(`qsub -q g1@pbs101`), but **only from a login node**: a job running in the AI
partition cannot submit across servers, and cannot `ssh` to a login node to try.
Submitting to `-q ai` from inside an AI-partition job does work -- it is how every
chain re-queues itself, and `train_in_pbs.sh` submitted `22160934.pbs102` from an
interactive `aidev` session on `asp2a-gpu001` on 2026-09-16.

Charging: `-P 11001786`. Check the balance with `myprojects` and your own burn
with `myusage`, both in `/app/apps/local/bin`.

---

## Submitting a training run

`scripts/platforms/aspire2a/train_in_pbs.sh` is the entry point for anything that trains. It takes a
model config and a dataset, and it is both the submitter and the job. Run it from the
worktree whose code should train:

```bash
cd /home/users/nus/kna/scratch/WyFormer/worktrees/<name>
bash scripts/platforms/aspire2a/train_in_pbs.sh yamls/models/lemat/chemsys_e_hull_adamw_wsd.yaml lemat_bulk_fmax1_stress
```

A full config is far more epochs than 24 h, so the job **chains itself**. Each
link pins one W&B run id (in `$WYFORMER_RUNS/.<dataset>__<config>__<branch>.runid`),
resumes from `$WYFORMER_RUNS/<id>/last_checkpoint.pt`, runs `train.py` under `timeout`
so it stops ~30 min before the wall, and re-`qsub`s itself. It stops on `train.py`
exit 0, on the attempt cap, or when a link crashes without getting anywhere.

What it enforces, at submission and again at the start of every link:

- **The checkout is committed.** Modified, staged, or untracked-but-not-ignored files
  are refused, with the list. A link that finds them stops the chain before touching
  anything; commit and resubmit to continue the pinned run. Each link prints the
  commit it trains with, and `scripts/train.py` writes it and the branch into the W&B
  run's config under `code` (the latest link's; earlier ones are in their logs).
- **The branch is part of the run.** The run key -- pinned id, queued-job record,
  duplicate check, job name `wyf_<config>.<branch>` -- includes the branch, so the same
  config trained from two branches is two independent runs. A link refuses to run if
  its checkout has since been switched to another branch.
- **It is not a Claude Code worktree** (`.claude/worktrees/`), which Claude Code can
  delete when its session ends; see [Working in a git worktree](#working-in-a-git-worktree).
  A worktree it launches from is locked with `git worktree lock`.
- **The data is already cached.** The tensor cache and tokeniser JSON for the config's
  tokeniser, and `data.pkl.gz` (read by the evaluation after training; `--train-arg
  --no-test` waives it), must exist. Jobs build nothing: see
  [Adding a dataset or tokeniser to the cache](#adding-a-dataset-or-tokeniser-to-the-cache).

Runs pinned before the branch joined the key (`.<dataset>__<config>.runid`) are not
adopted silently: the submitter names the old pin and asks for `--run-id <id>` to
continue it from this branch, or `--fresh`.

Options worth knowing (`--help` prints them all):

| Flag | Effect |
| --- | --- |
| `--dry-run` | print the qsub command and the job spec, submit nothing |
| `--pilot` | 3-epoch smoke test: own run id, no chaining, 01:59:00 wall -> `aidev` |
| `--fresh` | forget the pinned W&B id and start a new run |
| `--run-id ID` | continue a specific W&B run |
| `--offline` | W&B offline; `wandb sync <run-dir>` afterwards |
| `--ngpus/--ncpus/--mem/--walltime` | resources (defaults `1/16/64gb/23:59:59`) |
| `--train-arg ARG` | pass an extra argument to `train.py`, repeatable |
| `--allow-duplicate` | submit even though a chain for this dataset+config+branch is live |

Re-submitting the same dataset+config from the same branch after a stop continues the
pinned run from its last checkpoint with a fresh attempt budget.

### Fixing the code of a running experiment

Runs of a new experiment are not fire-and-forget. Every link reads the worktree afresh,
so a fix **committed** in the worktree while the chain is queued is what the next link
trains with. To make it take effect now rather than at the next 24 h boundary, `qdel`
the running link and resubmit from the same worktree: the chain resumes from the last
checkpoint, losing the epochs since (it is written every `checkpoint_period` epochs,
which defaults to `validation_period`). Resuming refuses a changed model config
(`check_resume_config`), so a fix that changes the config is a new run (`--fresh`).
Leave nothing uncommitted in a worktree with a live chain: the next link to start
would stop.

`train_in_pbs.sh` launches `scripts/train.py` only. The composition-floor ensemble
has its own trainer and its own one-slot job,
`scripts/platforms/aspire2a/train_formula_energy.pbs`; see its header for the
`-v` overrides.

---

## Submitting a relaxation / scoring pool

`scripts/platforms/aspire2a/protocol_relax.pbs` runs the expensive half of
`docs/de_novo_ranking_protocol.md` — `--stage relax` over every unique gene, then
`--stage score` — for one generated pool, chaining the same way:

```bash
qsub -v REPO=$PWD,POOL=generated/<run> scripts/platforms/aspire2a/protocol_relax.pbs
```

`REPO` defaults to the main checkout: PBS runs a copy of the script, which cannot tell
where it came from, so name the worktree. `POOL` is a directory under `REPO` holding
`wyckoff_genes.json.gz`; output lands in `POOL/protocol`. The link runs
`--stage screen` if `screen.json` is missing, so one `qsub` is enough. It stops if
`orb_models` is not importable rather than installing into the shared, read-only venv
([environment.md](environment.md#the-relax-extra-orb-and-mace)); it is installed today.
The orb-v3 checkpoint is already in `~/.cache/cached_path`, so relax stages
start immediately. For any *other* checkpoint, pre-fetch it with
`scripts/platforms/aspire2a/prefetch_cached_path.sh` rather than letting the job
crawl through it one connection at a time — see
[troubleshooting.md](troubleshooting.md#the-orb-checkpoint-downloads-at-a-crawl).

Useful `-v` variables: `WORKERS` (relaxation processes sharing the one GPU,
default 4), `MAX_ATTEMPTS` (default 6), `RELAX_TIMEOUT`, and `POST` — a command
run inside the container once `--stage score` succeeds, with `POOL` appended, so
whatever the pool was relaxed *for* does not wait for someone to notice the job
finished:

```bash
qsub -v "POOL=generated/<run>,POST=python scripts/analyse_dft_screen_uplift.py" \
     scripts/platforms/aspire2a/protocol_relax.pbs
```

Quote the whole `-v` list when `POST` has spaces in it; PBS splits it on commas
only.

---

## An interactive GPU session

```bash
qsub -I -q ai -P 11001786 \
     -l select=1:ngpus=1:ncpus=16:mem=110gb -l walltime=02:00:00
```

That routes to `aidev` (priority 100, so it starts quickly) and gives you two
hours. Two such jobs at a time. Then, on the node:

```bash
cd /home/project/11001786/WyFormer/WyckoffTransformer
module load singularity                      # or singularity/4.3.1
bash scripts/platforms/aspire2a/run_in_singularity.sh python scripts/train.py \
    yamls/models/lemat_bulk_fmax1/gene_min_energy_adamw_wsd.yaml lemat_bulk_fmax1_stress cuda --pilot
```

`scripts/platforms/aspire2a/run_in_singularity.sh` is the only supported way to run anything:
it puts `.venv/bin` on `PATH` inside the image, binds `/home/project` and `/raid`, sets
`PYTHONPATH=$REPO_DIR/src` so the active checkout's code is executed, and
sets `SINGULARITY_NO_EVAL=1` so `python -c` snippets with parentheses survive.
Override `SIF=` or `REPO_DIR=` if you need a different image or checkout, and
`EXTRA_BIND=` for extra mounts.

**Do not run anything long in an interactive session.** `aidev` is 2 h and the
node is taken back with no checkpointing grace; submit a chained job instead.

### Console scripts

Only `wyformer-generate`, `wyformer-cryspr` and `wyformer-protocol` are in
`.venv/bin` — the venv was installed before the rest were added to
`[project.scripts]`. Everything else runs as a module:

```bash
bash scripts/platforms/aspire2a/run_in_singularity.sh python -m wyckoff_transformer.cli.protocol_wandb --help
```

`screen`, `gene_screen`, `dft_screen`, `csp` and `protocol_wandb` all work this
way. To get the entry points properly, re-run step 4 of the build — see
[environment.md](environment.md#adding-or-repairing-the-project-install).

---

## Working in a git worktree

One experiment, one branch, one worktree. Worktrees share the repository's object store,
the [read-only cache and venv](#the-shared-cache-and-venv-are-read-only), the runs
directory and W&B; what they do not share is code, so experiments cannot see each
other's edits, and a chain trains exactly its own worktree's commits.

### Create the worktree yourself, then start Claude Code in it

```bash
bash scripts/platforms/aspire2a/create_worktree.sh <name> [start-point]   # branch <name>
cd /home/users/nus/kna/scratch/WyFormer/worktrees/<name>
claude            # later: claude --continue, from the same directory
```

`create_worktree.sh` puts the worktree on a branch named after it (an existing branch
of that name is checked out; otherwise it is created at `start-point`, default the
current `HEAD`), links `.venv` to the shared venv and `CLAUDE.local.md` to
[agent_brief.md](agent_brief.md), and checks `paths.env`. It refuses a branch that
predates this tooling, whose launcher would import the main checkout's code.

**Do not use `claude --worktree` (or ask Claude to "work in a worktree") for an
experiment that will train.** Claude Code owns those worktrees: on exiting the session
it removes a clean one automatically and offers to remove one with work in it, deleting
the branch with it; a backgrounded session's worktree is swept after
`cleanupPeriodDays`; and it branches from `origin/HEAD`, not from local commits
(Claude Code 2.1.273, <https://code.claude.com/docs/en/worktrees>). A chain re-reads
its worktree on every link for days, well past any session. A worktree made with
`git worktree add` is never removed by Claude Code, and `train_in_pbs.sh` refuses to
launch from `.claude/worktrees/` at all. Claude Code worktrees remain fine for work
that ends with the session.

Two consequences of starting `claude` in a plain worktree: Claude Code keeps its
project memory per directory, so a new worktree starts without the main checkout's; and
it does not apply the `--worktree` isolation checks, so nothing but the rules in
`CLAUDE.md` stops a session from editing the main checkout.

The worktree may live anywhere -- the scratch location is a convention, and project
storage works too. What matters is that nothing deletes it while a chain runs from it.
`train_in_pbs.sh` locks the worktree it launches from (`git worktree lock`), which makes
`git worktree remove` and `git worktree prune` refuse it. It is not unlocked when the
chain ends, because another chain may run from the same worktree; once none does:

```bash
git worktree list                      # locked worktrees are marked
git worktree unlock <path> && git worktree remove <path>
```

### Inside the worktree

- Run anything with `bash scripts/platforms/aspire2a/run_in_singularity.sh python ...`:
  it puts the worktree's `src` first on `PYTHONPATH`, ahead of the shared venv's
  editable install, which points at another checkout
  ([environment.md](environment.md#git-worktrees-reusing-the-shared-virtual-environment)).
- Pilot uncommitted code on a dev node directly (see
  [An interactive GPU session](#an-interactive-gpu-session)); `train_in_pbs.sh` takes
  committed code only, `--pilot` included.
- Commit, then submit with `bash scripts/platforms/aspire2a/train_in_pbs.sh <config> <dataset>`.
- Data, caches and runs resolve from `~/.config/wyformer/paths.env`; nothing to link.
- Merge the branch into `main` when the experiment is worth keeping. A branch that
  changes dependencies cannot be trained here yet: every worktree shares one venv.

---

## The shared cache and venv are read-only

Every worktree's jobs read the same tensor caches and the same `.venv`, so a stray write
-- a tokenisation pass pointed at the wrong name, a `uv pip install`, an `rm` -- damages
every experiment at once. Both are kept read-only with
`scripts/platforms/aspire2a/store_lock.sh`, which removes the write bits from every file
and directory (it holds on GPFS and Lustre alike; checked 2026-09-16):

```bash
bash scripts/platforms/aspire2a/store_lock.sh status   # writable entries in the cache and the venv
bash scripts/platforms/aspire2a/store_lock.sh lock     # both, by default; or name paths
```

It guards against accidents, not intent: the owner can always `chmod`. Reading is
unaffected, and Python simply skips writing bytecode caches into a read-only venv.
Installing into the venv means `unlock` on it, the install, and `lock` again -- with
the live-risk caveat in [environment.md](environment.md).

### Adding a dataset or tokeniser to the cache

Training jobs never build caches: `train_in_pbs.sh` refuses to submit, and every link
refuses to train, without the tensor cache, the tokeniser JSON and `data.pkl.gz`.
Building one is the job of whoever needs it. Most experiments reuse an existing cache;
one that changes tokenisation, or the tokenisation code, needs a **new tokeniser name**
(or a separate cache root, `--cache-dir`), never a rebuild over a name other runs use.

```bash
CACHE=/home/project/11001786/WyFormer/cache
# a new tokeniser for an existing dataset: let new files into the two directories,
# leaving the caches already there read-only
bash scripts/platforms/aspire2a/store_lock.sh open $CACHE/<dataset>/tensors $CACHE/<dataset>/tokenisers
# a new dataset: mkdir $CACHE/<dataset> after `store_lock.sh open $CACHE`
qsub -I -q ai -P 11001786 -l select=1:ngpus=1:ncpus=16:mem=110gb -l walltime=02:00:00
bash scripts/platforms/aspire2a/run_in_singularity.sh python scripts/tokenise_a_dataset.py \
    <dataset> yamls/tokenisers/<tokeniser>.yaml --new-tokenizer --n-jobs $NCPUS
bash scripts/platforms/aspire2a/store_lock.sh lock $CACHE/<dataset>
```

Pass `--n-jobs $NCPUS`: pandarallel otherwise forks one worker per core of the *node*
and the job is OOM-killed ([troubleshooting.md](troubleshooting.md#a-tokenisation-pass-gets-oom-killed)).
A LeMat-Bulk-sized pass can outlast the 2 h of `aidev` (the in-job build this replaced
allowed itself 4 h); for those, run the same commands in a `-q ai` batch job with
`walltime` to spare.

---

## Monitoring

```bash
qstat -u $USER                 # your jobs on the current server
qstat -u $USER @pbs101         # ...and on the other one
qstat -f <jobid> | grep -E 'queue|Resource_List|comment'
tail /scratch/users/nus/kna/WyFormer/logs/<job id>.OU   # e.g. 22196965.pbs102.OU, once the job has ended
```

PBS stdout/stderr are joined (`-j oe`) into `/scratch/users/nus/kna/WyFormer/logs/<job id>.OU`,
for every launcher and every checkout, so a log outlives the worktree it came from; the
cluster purges old files there itself. PBS copies the file there when the job ends; while
it runs, it grows in `/var/spool/pbs/spool/<job id>.OU` on the execution node (readable
from an interactive job on the same node). (Chains submitted before 2026-09-16 still write
to `logs/` in the checkout they came from.) The job spec a chain reads is in
`$WYFORMER_RUNS/.jobspec/<key>-<timestamp>.sh`. W&B goes live to
`symmetry-advantage/WyckoffTransformer`; auth is `~/.netrc`, which the container sees
through the automatic `$HOME` bind.

`nqstat` (the site's pretty `qstat`) is on `PATH` but broken on the compute
nodes — it cannot find `libcjson.so.1`. Use plain `qstat`.

---

## Node-local scratch

`/raid` is 14 TB of node-local NVMe, and PBS gives each job
`/raid/pbs.<jobid>`. It is **not** mounted into the container automatically —
`run_in_singularity.sh` binds it explicitly — and it is **not** shared between
nodes and **does not survive the job**. Anything you want to keep goes to
`/scratch`.
