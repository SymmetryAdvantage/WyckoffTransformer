# This host is ASPIRE 2A

Linked as `CLAUDE.local.md` by `scripts/build_singularity_venv.sh` when run with
`WYFORMER_PLATFORM=aspire2a`. A brief; the full pages are in
`docs/platforms/aspire2a/`.

- **Environment:** a Singularity image
  (`~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif`) with `.venv` built against it.
  The host has no `python` or `uv`. **Never run `python`, `uv` or `pytest`
  from a host shell** -- `uv sync`/`uv run` there deletes `.venv`, which running
  jobs share.
- **Run** inside the image: `bash scripts/platforms/aspire2a/run_in_singularity.sh python ...`,
  after `module load singularity`, on a compute node.
- **Everything real is a PBS job.** GPU jobs cap at 24 h (2 h in dev), so
  training chains itself: `bash scripts/platforms/aspire2a/train_in_pbs.sh <config.yaml> <dataset>`
  (`--pilot` for a 2 h dev run). Submit from a **login node**, to `-q ai` --
  every AI queue is route-only, `qsub -q aiq1` is rejected.
- **Don't move or rename** `scripts/platforms/aspire2a/train_in_pbs.sh` or the `*.pbs` scripts while
  a `wyf_*` job is queued or running: chains re-submit themselves by path.
- **Quotas:** home is the tight one -- 50 GB, 34.6 GB used on 2026-09-07,
  12 GB of it the image, and `cached_path` (ORB checkpoints) writes there. The
  checkout is on `/scratch/users/nus/kna`. `/raid` is node-local and gone when
  the job ends. Project `11001786`; check `myprojects` before a long chain.
