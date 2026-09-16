# This host is ASPIRE 2A

Linked as `CLAUDE.local.md` by `scripts/platforms/link_agent_brief.sh aspire2a` or
`scripts/platforms/aspire2a/env_init.sh`. A brief; full pages are in
`docs/platforms/aspire2a/`.

- **Environment:** a Singularity image
  (`~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif`) with `.venv` built against it
  at `/home/project/11001786/WyFormer/WyckoffTransformer/.venv`.
  The host has no `python` or `uv`. **Never run `python`, `uv` or `pytest`
  from a host shell** -- `uv sync`/`uv run` there deletes `.venv`, which running
  jobs share.
- **Git worktrees:** located in `/home/users/nus/kna/scratch/WyFormer/worktrees/<name>`.
  Two differences from other platforms:
  1. Worktrees live on Lustre scratch (`/home/users/nus/kna/scratch/WyFormer/worktrees`).
  2. Worktrees **reuse the main virtual environment** (`/home/project/11001786/WyFormer/WyckoffTransformer/.venv`)
     because ASPIRE 2A has a slow filesystem. Initialise a worktree with
     `bash scripts/platforms/aspire2a/env_init.sh` (or create with
     `bash scripts/platforms/aspire2a/create_worktree.sh <name> [branch]`).
- **Run inside the image:** `bash scripts/platforms/aspire2a/run_in_singularity.sh python ...`,
  after `module load singularity`. `run_in_singularity.sh` automatically binds `/home/project`,
  reuses the shared `.venv`, and sets `PYTHONPATH=$REPO_DIR/src` so the worktree's own
  code is executed.
- **Storage paths:** defined in `~/.config/wyformer/paths.env`:
  `WYFORMER_DATA=/home/project/11001786/WyFormer/data`,
  `WYFORMER_CACHE=/home/project/11001786/WyFormer/cache`,
  `WYFORMER_RUNS=/scratch/users/nus/kna/WyFormer/runs`,
  `WANDB_DIR=/scratch/users/nus/kna/WyFormer`.
- **Everything real is a PBS job.** GPU jobs cap at 24 h (2 h in dev), so
  training chains itself: `bash scripts/platforms/aspire2a/train_in_pbs.sh <config.yaml> <dataset>`
  (`--pilot` for a 2 h dev run). Submit from a **login node**, to `-q ai` --
  every AI queue is route-only, `qsub -q aiq1` is rejected.
- **Don't move or rename** `scripts/platforms/aspire2a/train_in_pbs.sh` or the `*.pbs` scripts while
  a `wyf_*` job is queued or running: chains re-submit themselves by path.
- **Quotas:** home is the tight one -- 50 GB, 12 GB of it the image, and `cached_path`
  (ORB checkpoints) writes there. The main checkout and data/cache live on
  `/home/project/11001786/WyFormer`. Runs and worktrees live on `/scratch/users/nus/kna/WyFormer`.
  `/raid` is node-local and gone when the job ends. Project `11001786`; check `myprojects`
  before a long chain.
