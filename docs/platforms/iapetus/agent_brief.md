# This host is iapetus

Linked as `CLAUDE.local.md` by `scripts/build_singularity_venv.sh` when run with
`WYFORMER_PLATFORM=iapetus`. A brief; the full pages are in
`docs/platforms/iapetus/`.

- **Environment:** the Docker image `pytorch:2.14.0-cuda11.8-py312-universal`
  (custom torch for `sm_35`/`sm_50`) with the project `.venv` layered over it.
  The venv is valid **only inside the image**; never create or activate it on
  the host, whose Python is 3.14.
- **Run** everything through the launcher, spelled out literally:
  `scripts/platforms/iapetus/run.sh python scripts/...`,
  `scripts/platforms/iapetus/run.sh python -m pytest`. The docs abbreviate it
  as an alias, `run`; don't rely on the alias in a non-interactive shell.
- **GPUs:** 2x Tesla K20c (<5 GiB) and a GTX 750 Ti (2 GiB), driver 470, shared.
  Check `nvidia-smi`, set `CUDA_VISIBLE_DEVICES`, pass `cuda`, start with small
  batches. No Triton: the K20c is below its minimum compute capability.
- **CPU:** 6 physical cores, 30 GiB RAM, no scheduler; the ranking protocol
  runs with `--pyxtal-cores 6`.
- **Disk:** the root filesystem is 233 GiB with about 55 GiB free (2026-09-08);
  check before large runs and keep bulky results outside the repository.
