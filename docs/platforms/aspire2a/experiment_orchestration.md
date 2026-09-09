# Orchestrating WyFormer training experiments on NSCC ASPIRE 2A

Goal: run many WyFormer experiments that vary **both config and code**, on a PBS
cluster, with the trained model delivered through W&B. This note records the
design and the environment facts it rests on.

## Environment facts (verified 2026-09-04)

### Queues (`qstat -q`, ASPIRE 2A)

| Queue | Walltime | GPUs/job | Notes |
|-------|----------|----------|-------|
| `g1` | ≤ 24 h | 1 | single-GPU, the target for these runs |
| `g2`/`g3`/`g4` | ≤ 24 h | 2–3 / 4 / 5–256 | multi-GPU |
| `glong` | 24–120 h | 1–64 | long walltime, longer queue wait |
| `gdev` | ≤ 2 h | 1–4 | pilots; max 2 running jobs/user |
| `ai`, `aiq1`..`aiq4`, `ailong` | ≤ 24 h (`ailong` ≤ 120 h) | AI partition; `ailong` max 1 running job/user |

- System-wide cap: **100 jobs per user**.
- No mid-run resume today (trainer checkpoints the best `state_dict` only, no
  optimiser/scheduler/step state) — a "clean resume" is being written. Until it
  lands, every run must finish inside its walltime; `g1`'s 24 h is the ceiling.
- We submit to the **≤ 24 h queue** (`g1` / `ai`). `glong` is avoided because of
  queue wait.

### W&B connectivity — tested from a GPU compute node

Ran a probe job on `asp2a-gpu002` (queue `ai`):

| Target | Result |
|--------|--------|
| `GET https://api.wandb.ai/` | 404 in 0.26 s (reachable) |
| `GET https://wandb.ai/` | 301 (reachable) |
| `POST https://api.wandb.ai/graphql` | **200, `{"data":{"viewer":null}}`** — real API, `null` only because no key sent |
| `storage.googleapis.com` (W&B artifact backend) | 400 (reachable) |
| `huggingface.co` | 200 |
| DNS `api.wandb.ai` / `wandb.ai` | resolves |
| proxy env vars | none set, none needed |

**Conclusion: W&B is fully usable from GPU compute nodes.** The NSCC firewall
(blacklist of IPs/URLs) does not block W&B or Google Cloud Storage.

- Live logging works. `wandb.Settings(init_timeout=180)` in `scripts/train.py` is
  just defensive.
- For an unattended batch, `WANDB_MODE=offline` + `wandb sync` from a login node
  is still a reasonable robustness choice (a transient blip otherwise stalls
  against the 180 s timeout, and many concurrent jobs raise that probability). It
  is now a preference, not a requirement.
- HuggingFace: set a shared `HF_HOME`, pre-download on a login node, set
  `HF_HUB_OFFLINE=1` in jobs.

### What already works (no code needed)

- `trainer.py` logs `best_model_params.pt` as a W&B artifact on every validation
  improvement, plus config and processor artifacts.
- `scripts/predict.py:load_wandb_model(run_id)` reconstructs a trained model from
  only a W&B run ID.

So **"trained model shared via W&B" is already solved** — the run ID is the
deliverable. Only the dispatch layer is missing.

## Design

### 1. One immutable shared venv

venv construction under `uv` here is slow, so build it **once** and share it
read-only across all jobs.

- Build on a login node, deps only (no editable project install, so no baked-in
  `src/` path to shadow): `uv sync --frozen --no-install-project`.
- Install path keyed by `sha256(uv.lock)`, e.g. `$SCRATCH/wyformer-venv/<hash>`.
- Rebuilt only when `uv.lock` changes; keep the last couple.
- The dispatcher refuses a spec whose lock hash has no prebuilt venv and prints
  the build command (or fires a `qdev` CPU job to build it). venv builds never
  happen inside a GPU job.

### 2. Per-experiment code snapshot via `git worktree`

Worktrees share the object store, so this costs seconds, not a sync.

```bash
git worktree add --detach $WT $SHA
mkdir -p $WT/cache
ln -s $CANON/data                      $WT/data                      # shared raw data
ln -s $CANON/cache/$DATASET/data.pkl.gz $WT/cache/$DATASET/data.pkl.gz  # shared, code-independent (stage 1)
# stage-2 tensor cache: symlink the shared one when the tokenisation fingerprint
# matches, else point at a snapshot-private @<fp> path (see section 3)
# wyckoffs_enumerated_by_ss.json is build-generated and NOT committed; it is loaded
# relative to the package source dir, so generate it once per worktree (pure Python, no GPU):
cd $WT && PYTHONPATH=src $VENV/bin/python -m wyckoff_transformer.preprocess_wychoffs
```

Jobs then run `PYTHONPATH=$WT/src $VENV/bin/python scripts/train.py ...` from
`$WT`. `PYTHONPATH` shadows the (absent) project install; the repo's
`parents[2]/yamls` and `parents[2]/cache` lookups resolve inside the worktree.

**Config overrides:** `train.py` only takes `--pilot` / `--compile` / etc., not
arbitrary `key=value`. Generate a concrete YAML per variant (declarative,
git-tracked) rather than patching in dotlist support.

### 3. Data preprocessing — a two-stage cache

`scripts/train.py` builds **neither** cache; a missing cache is a hard
`FileNotFoundError` (`tokenization.load_tensors_and_tokenisers`). So the
dispatcher must guarantee both stages exist before the GPU job. Both stages are
CPU work and must never run inside the 24 h GPU job (`use_cached_tensors=False`
tokenises on the fly — acceptable only for pilots).

**Stage 1 — dataset cache (`scripts/cache_a_dataset.py`)**
`data/<dataset>/{train,val,test}.csv` → `cache/<dataset>/data.pkl.gz`
(pyxtal symmetrisation, parallel by structure, slow). Depends on the raw CSVs,
`symmetry_precision` / `symmetry_a_tol` / `max_wp`, and the symmetry code in
`data.py` — **not** on model or tokeniser code. Fully shareable: build once,
symlink `cache/<dataset>/data.pkl.gz` into every worktree. The existing
`data/<name>_tol_*` / `<name>_0.01_1_63` naming convention already folds the
symmetry params into the dataset name, so param variants get separate cache
dirs for free. Rebuild only when raw data or `data.py` symmetry code changes.

**Stage 2 — tensor cache (`scripts/tokenise_a_dataset.py`)**
`data.pkl.gz` + `yamls/tokenisers/<tok>.yaml` →
`cache/<dataset>/tensors/<tok>.safetensors` + `cache/<dataset>/tokenisers/<tok>.json`.
Depends on `data.pkl.gz`, the tokeniser YAML, **and the tokenisation code**
(`tokenization.py`, `wyckoff_processor.py`, `cascade/`, `preprocess_wychoffs.py`).
The current layout keys it only by tokeniser name → a code snapshot that changes
tokenisation would read a stale or schema-mismatched cache. Handle it by
**fingerprinting**:

- `fp = hash(tokenization.py, wyckoff_processor.py, cascade/**, preprocess_wychoffs.py, <tok>.yaml)`
  for the `code_ref`.
- `fp` equals the canonical checkout's → symlink the shared
  `cache/<dataset>/tensors` + `tokenisers` read-only (the common case: a code
  change that doesn't touch tokenisation costs nothing).
- `fp` differs → dispatcher runs a **tokenise pre-job** (CPU, `normal`/`qdev`)
  into a snapshot-private path `cache/<dataset>/tensors/<tok>@<fp>.safetensors`,
  and the training job is `qsub`ed `-W depend=afterok:<pre-job>`.
- Small patch needed: let the tensor-cache loader / `tokenise_a_dataset.py` take
  a cache path or `@<fp>` suffix so snapshot-private caches don't collide with
  the shared name. A fully uniform variant (always `@<fp>`, always
  tokenise-if-missing as a pre-job) drops the shared-vs-private branch at the
  cost of a hash + a usually-skipped no-op job.

**Pilots:** `train.py --pilot` only cuts epochs and reuses the full cache — safe.
`tokenise_a_dataset.py --pilot` writes a 100-sample tensor set to the *same* path
as the full cache — the dispatcher must never point pilot tokenisation at a
shared cache dir.

### 4. FireWorks-style dispatcher (lighter than FireWorks)

The VASP setup (`~/NSCC-VASP-computer`) needs MongoDB + FireWorks because it runs
thousands of multi-step DAG jobs with `auto_continue` and a hull-export step. A
WyFormer experiment is **one self-contained single-GPU job, no DAG, no export**.
That collapses the stack:

| FireWorks piece | WyFormer equivalent |
|-----------------|---------------------|
| MongoDB LaunchPad | a `queue/` dir of YAML specs, tracked on a git branch |
| `submit_fireworks.py` | `wyformer-submit` — commits `{code_ref, config, dataset, seed, overrides}` and pushes |
| `infinite_fireworks.sh` + `qlaunch rapidfire` | `wyformer-dispatcher` — tmux loop on the login node |
| PBS template + `auto_continue` | per-run PBS script; resume-chain with `-W depend=afternotok` once clean resume lands |
| `export_via_pbs.sh` | not needed — W&B already has the model artifact |

No MongoDB, no queue adapter, no serialization layer. The git branch is both the
queue and the state store — auditable, and submittable from a laptop with
`git push` + one SSH command.

**Spec** (`queue/<id>.yaml`, checked in):

```yaml
code_ref: 4e0eea7                                    # git SHA (must be pushed to the NSCC clone)
config: yamls/models/lemat_bulk_ehull/ehull_adamw_wsd.yaml
dataset: mp_20_biternary
seed: 0
status: pending                                      # dispatcher updates: pending -> submitted -> running -> done/failed
```

**`wyformer-dispatcher`** (single instance, `flock`-guarded, in tmux on
`asp2a-login-nus02`): loop every ~60 s —
1. `git fetch` the queue branch.
2. Count in-flight jobs (`qstat`); stop if at the throttle target.
3. For each `pending` spec: verify the venv for its lock hash exists; ensure the
   worktree (create + symlink + preprocess if new); ensure stage-1 cache exists
   (else CPU pre-job); compute the tokenisation fingerprint and either symlink
   the shared stage-2 cache or `qsub` a tokenise pre-job and depend on it;
   render the PBS script; `qsub`; write `status: submitted` + `pbs_job_id` +
   `wandb_run_id` back as a commit.
4. On job exit: if not converged and clean resume is available, resubmit
   `--resume <ckpt>` with `-W depend`; on nonzero exit retry up to N times, then
   `status: failed`.

**PBS job** (`g1`, ≤ 24 h):

```bash
#PBS -q g1
#PBS -P 11001786
#PBS -l select=1:ngpus=1:ncpus=16:mem=64gb
#PBS -l walltime=23:55:00
#PBS -j oe -o logs/

cd $WT
export WANDB_RUN_GROUP=$BATCH HF_HUB_OFFLINE=1   # + WANDB_MODE=offline if chosen
PYTHONPATH=$WT/src $VENV/bin/python scripts/train.py "$CONFIG" "$DATASET" cuda --run-path "$WT/runs"
```

### Submitting from anywhere

1. `git push nscc <branch>` so the SHA exists in the NSCC clone.
2. `ssh asp2a-login-nus02 wyformer-submit --code-ref <sha> --config <path> --dataset <name>`
   (or drop a spec in the Dropbox dir already synced by `scripts/Dropbox_pull.sh`).

The trained model is then the W&B run — consume it with
`predict.py:load_wandb_model(run_id)`.

## Alternative considered: reuse FireWorks

Viable and almost no new code — a training run is a one-`ScriptTask` workflow and
`qlaunch rapidfire` already throttles against the job cap. Rejected as the
default because it means keeping MongoDB up for training too and adding a GPU
queue adapter, to wrap what is fundamentally a single `qsub`.

## Status / next steps

- [ ] Wait for the clean-resume branch so the dispatcher's chaining targets the
      real checkpoint format.
- [ ] Small patch: let the tensor-cache loader / `tokenise_a_dataset.py` accept a
      cache path or `@<fp>` suffix, so snapshot-private stage-2 caches don't
      collide with the shared name.
- [ ] Scaffold `scripts/experiments/`: `wyformer-submit`, `wyformer-dispatcher`,
      PBS templates (train + tokenise pre-job), login-node venv-build +
      (optional) `wandb sync` helpers.
- [ ] One-time: build the shared venv; stage raw `data/` and pre-cache datasets
      under `HF_HOME`; build stage-1 `data.pkl.gz` and the canonical stage-2
      tensor cache.
