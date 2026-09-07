# Running WyFormer on ASPIRE 2A

Everything here assumes the environment is already built. If it is not, or you
broke it, read [environment.md](environment.md) first.

```bash
cd /scratch/users/nus/kna/WyckoffTransformer
```

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

Charging: `-P 11001786`. Check the balance with `myprojects` and your own burn
with `myusage`, both in `/app/apps/local/bin`.

---

## Submitting a training run

`scripts/train_in_pb.sh` is the entry point for anything that trains. It takes a
model config and a dataset, and it is both the submitter and the job:

```bash
bash scripts/train_in_pb.sh yamls/models/lemat_bulk_ehull/ehull_adamw_wsd_5x.yaml lemat_bulk_ehull
```

A full config is far more epochs than 24 h, so the job **chains itself**. Each
link pins one W&B run id (in `runs/.<dataset>__<config>.runid`), resumes from
`runs/<id>/last_checkpoint.pt`, runs `train.py` under `timeout` so it stops ~30
min before the wall, and re-`qsub`s itself. It stops on `train.py` exit 0, on
the attempt cap, or when a link crashes without getting anywhere.

Before training it also, when needed: builds the `site_symmetry_ops` engineers
and their lookup table (~1 min, idempotent), and builds the tensor cache for the
config's tokeniser (one pandarallel pass over the dataset, sized to the job's
CPUs rather than the node's — which is what keeps it from being OOM-killed).
Both come out of the first link's wall clock.

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
| `--allow-duplicate` | submit even though a chain for this config+dataset is live |

Re-submitting the same config+dataset after a stop continues the pinned run from
its last checkpoint with a fresh attempt budget.

`scripts/train_ehull_5x.pbs`, `scripts/train_ehull_ssops.pbs` and
`scripts/train_formula_energy.pbs` are the older single-purpose versions of the
same scheme, kept because chains launched from them are still running.

---

## Submitting a relaxation / scoring pool

`scripts/protocol_relax.pbs` runs the expensive half of
`docs/de_novo_ranking_protocol.md` — `--stage relax` over every unique gene, then
`--stage score` — for one generated pool, chaining the same way:

```bash
qsub -v POOL=generated/<run> scripts/protocol_relax.pbs
```

`POOL` is a directory holding `wyckoff_genes.json.gz`; output lands in
`POOL/protocol`. The link runs `--stage screen` if `screen.json` is missing and
installs the `relax` extra if `orb_models` is absent, so one `qsub` is enough —
but **pre-fetch the ORB checkpoint first**, or the first link will spend hours
downloading at 70 kB/s. See
[troubleshooting.md](troubleshooting.md#the-orb-checkpoint-will-not-download-on-a-compute-node).

Useful `-v` variables: `WORKERS` (relaxation processes sharing the one GPU,
default 4), `MAX_ATTEMPTS` (default 6), `RELAX_TIMEOUT`.

---

## An interactive GPU session

```bash
qsub -I -q ai -P 11001786 \
     -l select=1:ngpus=1:ncpus=16:mem=110gb -l walltime=02:00:00
```

That routes to `aidev` (priority 100, so it starts quickly) and gives you two
hours. Two such jobs at a time. Then, on the node:

```bash
cd /scratch/users/nus/kna/WyckoffTransformer
module load singularity                      # or singularity/4.3.1
bash scripts/run_in_singularity.sh python scripts/train.py \
    yamls/models/lemat_bulk_ehull/ehull_adamw_wsd_5x.yaml lemat_bulk_ehull cuda --pilot
```

`scripts/run_in_singularity.sh` is the only supported way to run anything:
it puts `.venv/bin` on `PATH` inside the image, binds the repo and `/raid`, and
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
bash scripts/run_in_singularity.sh python -m wyckoff_transformer.cli.protocol_wandb --help
```

`screen`, `gene_screen`, `dft_screen`, `csp` and `protocol_wandb` all work this
way. To get the entry points properly, re-run step 4 of the build — see
[environment.md](environment.md#adding-or-repairing-the-project-install).

---

## Monitoring

```bash
qstat -u $USER                 # your jobs on the current server
qstat -u $USER @pbs101         # ...and on the other one
qstat -f <jobid> | grep -E 'queue|Resource_List|comment'
tail -f logs/<jobid>.OU
```

PBS stdout/stderr are joined (`-j oe`) into
`/scratch/users/nus/kna/WyckoffTransformer/logs/<jobid>.OU`. W&B goes live to
`symmetry-advantage/WyckoffTransformer`; auth is `~/.netrc`, which the container
sees through the automatic `$HOME` bind.

`nqstat` (the site's pretty `qstat`) is on `PATH` but broken on the compute
nodes — it cannot find `libcjson.so.1`. Use plain `qstat`.

---

## Node-local scratch

`/raid` is 14 TB of node-local NVMe, and PBS gives each job
`/raid/pbs.<jobid>`. It is **not** mounted into the container automatically —
`run_in_singularity.sh` binds it explicitly — and it is **not** shared between
nodes and **does not survive the job**. Anything you want to keep goes to
`/scratch`.
