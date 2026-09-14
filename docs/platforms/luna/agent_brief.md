# This host is luna

Linked as `CLAUDE.local.md` by `scripts/platforms/luna/build_venv.sh`. A brief;
the full pages are in `docs/platforms/luna/`.

- **Environment:** an Apptainer container
  (`~/containers/pytorch-2.14.0-cuda12.6`) with the venv `.venv-luna` on top,
  reusing the container's torch. The venv works **only inside the container**.
- **Run** everything through the launcher, spelled out literally:
  `scripts/platforms/luna/run.sh python scripts/...`,
  `scripts/platforms/luna/run.sh pytest`. Not a bare `python` on the host, and
  not the launcher held in a variable (`$R python ...`) -- a worktree-isolated
  session refuses that.
- **Rebuild** with `scripts/platforms/luna/build_venv.sh`. Never `uv sync`: it
  would replace the container's torch.
- **GPUs:** 8x L40S, shared. **GPU 2 is faulty** (uncorrectable ECC) -- skip
  it. Check `nvidia-smi`, set `CUDA_VISIBLE_DEVICES` explicitly and pass `cuda`.
- **Long jobs:** `nohup ... &` or `tmux`. pandarallel fans out to 128 workers
  and the memory spike can kill a job tied to an interactive session.
- **W&B:** credentials in `~/.netrc`, visible inside the container.
