# This host is zeus

A brief; the full pages are in `docs/platforms/zeus/`.

- **Environment:** a plain host venv, no container, no scheduler. torch is built
  locally and installed from `/mnt/hdd/torch_wheels/`.
- **Run** from the root of the checkout you are in, main or worktree:
  `.venv/bin/python scripts/...` or `uv run python scripts/...`. Do not
  `source .venv/bin/activate` -- a worktree-isolated session refuses it -- and
  do not `cd /home/kna/WyckoffTransformer` from a worktree: that runs the main
  checkout's code.
- **A worktree needs its own venv:** `scripts/platforms/zeus/env_init.sh` (about
  20 s). Never a bare `uv sync`, which prunes the `dev`/`relax`/`nep` extras,
  and never `uv venv` over an existing `.venv`.
- **GPUs:** two shared RTX 6000 Ada, usually busy. Check `nvidia-smi`, set
  `CUDA_VISIBLE_DEVICES=0` or `=1` explicitly and pass `cuda`.
- **CPU-only tests:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest ...`;
  otherwise pytest can OOM on a full GPU 0.
- **Cores:** 24 physical (`nproc` says 48). CrySPR pools:
  `NP=16 OMP_NUM_THREADS=1`; long relaxations: `wyformer-cryspr --device cpu`.
- **Data store, cache, runs, W&B dir:** `/home/kna/.local/share/wyformer`,
  shared by every checkout; `.venv/bin/python -m wyckoff_transformer.paths`.
- **W&B:** credentials in `~/.netrc`; log real runs with
  `WANDB_ENTITY=symmetry-advantage`.
- **Long jobs:** `nohup ... &` or `tmux`; nothing holds a job for you.
