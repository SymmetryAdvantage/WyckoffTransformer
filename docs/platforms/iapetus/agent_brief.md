# This host is iapetus

Linked as `CLAUDE.local.md` by `scripts/platforms/iapetus/build_venv.sh`. A
brief; the full pages are in `docs/platforms/iapetus/`.

- **Environment:** the Docker image `pytorch:2.14.0-cuda11.8-py312-universal`
  (custom torch for `sm_35`/`sm_50`) with the project `.venv` layered over it.
  The venv is valid **only inside the image**; never create or activate it on
  the host, whose Python is 3.14.
- **Build** a checkout's venv -- the main one or a worktree, one each -- with
  `scripts/platforms/iapetus/build_venv.sh`, run from that checkout.
- **Triton must not be installed**: the K20c is below its minimum compute
  capability. The build fails if it is importable.
- **Run** everything through the launcher, spelled out literally:
  `scripts/platforms/iapetus/run.sh python scripts/...`,
  `scripts/platforms/iapetus/run.sh python -m pytest`. The docs abbreviate it
  as an alias, `run`; don't rely on the alias in a non-interactive shell.
  Inside `run.sh bash -c '...'`, call `/workspace/.venv/bin/python`: a bare
  `python` there is the image's, without the project's packages.
- **Data, cache, runs, W&B** are in `/mnt/hdd/kna/wyformer/`, mounted by
  `run.sh` at the same paths; resolve them with `wyckoff_transformer.paths`.
- **GPUs:** 2x Tesla K20c (<5 GiB) and a GTX 750 Ti (2 GiB), driver 470, shared.
  Check `nvidia-smi`, set `CUDA_VISIBLE_DEVICES`, pass `cuda`, start with small
  batches.
- **CPU:** 6 physical cores, 30 GiB RAM, no scheduler; the ranking protocol
  runs with `--pyxtal-cores 6`.
- **Disk:** the root filesystem, holding the checkouts, is 233 GiB with about
  161 GiB free; the store's HDD has about 769 GiB free (2026-09-15). A venv is
  1.1 GiB per checkout. Keep bulky results outside the repository.
