# This host is ASPIRE 2A

Linked as `CLAUDE.local.md` by `scripts/platforms/link_agent_brief.sh aspire2a` or
`scripts/platforms/aspire2a/env_init.sh`. A brief; full pages are in
`docs/platforms/aspire2a/`.

- **Environment:** a Singularity image
  (`~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif`) with one shared venv at
  `/home/project/11001786/WyFormer/WyckoffTransformer/.venv`, linked as `.venv` from every
  worktree, rebuilt only with `scripts/platforms/aspire2a/build_venv.sh`.
  The host has no `python` or `uv`. **Never run `python`, `uv` or `pytest`
  from a host shell** -- `uv sync`/`uv run` there deletes `.venv`, which running
  jobs share.
- **Run inside the image:** `bash scripts/platforms/aspire2a/run_in_singularity.sh python ...`,
  after `module load singularity`. It binds `/home/project`, uses the shared venv, and puts
  the invoking checkout's `src` first on `PYTHONPATH` -- the venv's editable install points
  at the main checkout, so in a worktree nothing run outside it imports the worktree's code.
- **Experiments are branches in worktrees** at `/home/users/nus/kna/scratch/WyFormer/worktrees/<name>`,
  made with `bash scripts/platforms/aspire2a/create_worktree.sh <name> [start-point]`, with
  `claude` started inside. Not `claude --worktree`: Claude Code deletes its own worktrees
  when sessions end, and a training chain re-reads its worktree for days. The launcher
  refuses to run from `.claude/worktrees/`.
- **Everything real is a PBS job.** GPU jobs cap at 24 h (2 h in dev), so
  training chains itself: `bash scripts/platforms/aspire2a/train_in_pbs.sh <config.yaml> <dataset>`
  (`--pilot` for a 2 h dev run), run from the experiment's worktree. It takes **committed
  code only** (at submission and at every link), keys the run by dataset, config **and
  branch**, and trains whatever the worktree has committed when each link starts. Submit to
  `-q ai` -- every AI queue is route-only, `qsub -q aiq1` is rejected.
- **The cache store and the venv are read-only** (`scripts/platforms/aspire2a/store_lock.sh`).
  Jobs build no caches and refuse to train without one. A new tokeniser or dataset is built
  deliberately first, under a new name: `docs/platforms/aspire2a/usage.md`.
- **Storage paths:** defined in `~/.config/wyformer/paths.env`:
  `WYFORMER_DATA=/home/project/11001786/WyFormer/data`,
  `WYFORMER_CACHE=/home/project/11001786/WyFormer/cache`,
  `WYFORMER_RUNS=/scratch/users/nus/kna/WyFormer/runs`,
  `WANDB_DIR=/scratch/users/nus/kna/WyFormer`. PBS logs: `/scratch/users/nus/kna/WyFormer/logs`.
- **Don't move or rename** `scripts/platforms/aspire2a/train_in_pbs.sh` or the `*.pbs` scripts,
  and don't remove, switch the branch of, or leave uncommitted changes in a worktree, while
  a `wyf_*` job from it is queued or running: chains re-submit themselves by path.
- **Quotas:** home is the tight one -- 50 GB, 12 GB of it the image, and `cached_path`
  (ORB checkpoints) writes there. The main checkout, its venv and data/cache live on
  `/home/project/11001786/WyFormer`. Runs, logs and worktrees live on
  `/scratch/users/nus/kna`. `/raid` is node-local and gone when the job ends.
  Project `11001786`; check `myprojects` before a long chain.
