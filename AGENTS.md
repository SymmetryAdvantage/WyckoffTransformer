# Platform-specific setup
WyFormer runs on a range of machines (workstations, clusters, CI). Anything that is
true for **one** host only must not leak into the top-level `README.md`, `AGENTS.md`
or `scripts/`:
 - human/agent-readable setup & run instructions -> `docs/platforms/<platform>/`,
   split by use case (e.g. `usage.md`, `environment.md`, `troubleshooting.md`)
   once it outgrows a single `README.md`
 - scripts (container builds, env init, scheduler submission) -> `scripts/platforms/<platform>/`

`<platform>` is the short host or cluster name, e.g. `luna`, `zeus`.

Before setting up an environment, check whether `docs/platforms/<platform>/` exists
and follow it instead of improvising. When you learn something host-specific, record
it there rather than in the generic docs.
