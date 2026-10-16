# This host is hera

Read [Hera's environment notes](environment.md) first, then the shared
[Zeus agent brief](../zeus/agent_brief.md) and its linked usage instructions.
Hera's notes take precedence over Zeus's host facts.

- The CPU and GPUs match Zeus; Hera has 499 GiB RAM. Check local GPU load.
- Initial inspection on 2026-10-16 found no project venv, `uv` on `PATH`, or
  data-store config. Complete the environment setup before running Python.
- Reuse Zeus's initializer with `UV_PYTHON=3.12`, then run
  `scripts/platforms/link_agent_brief.sh hera` to restore this brief.
- Use the current checkout's `.venv/bin/python`; each worktree needs its own
  venv. Resolve storage through Hera's config rather than assuming Zeus's paths.
