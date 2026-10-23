# This host is hera

Read [Hera's environment notes](environment.md) first, then the shared
[Zeus agent brief](../zeus/agent_brief.md) and its linked usage instructions.
Hera's notes take precedence over Zeus's host facts.

- The CPU and GPUs match Zeus; Hera has 499 GiB RAM. Check local GPU load.
- On 2026-10-16, NVSHMEM was installed and Zeus's storage config copied.
  `uv` is at `~/.local/bin/uv`; Hera follows Zeus on Python 3.14.
  The `compile` extra is deprecated; Torch supplies Triton.
  Python dependencies and host runtime libraries are installed; Torch import
  and a CUDA tensor calculation passed on 2026-10-16. The current MP20 DER
  pilot completed successfully with W&B run `5mu98ebn`; see environment notes.
  The old Python 3.12 venv was removed at the user's request.
- Reuse Zeus's initializer with its Python 3.14 default, then run
  `scripts/platforms/link_agent_brief.sh hera` to restore this brief.
- Use the current checkout's `.venv/bin/python`; each worktree needs its own
  venv. Resolve storage through Hera's config rather than assuming Zeus's paths.
