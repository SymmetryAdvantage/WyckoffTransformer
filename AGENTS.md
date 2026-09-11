# LeMat-Bulk datasets
`lemat_bulk_fmax1_stress` is the only current variant. Use it for new work, and follow
`docs/lemat_bulk_pipeline.md` to rebuild any of it.

Every other LeMat variant in `data/` and `cache/` — `lemat_bulk_fmax1`,
`lemat_bulk_ehull`, `lemat_bulk_ehull_wp20`, and the `_pilot` subsamples — is superseded.
Use them **only** to analyse runs already trained on them, never for new training,
evaluation or caching. They predate `scripts/recover_mp_forces.py`, so they either drop
the 30,679 Materials Project rows whose archived forces are empty (a `max_force <= X` cut
drops NaN for every X) or carry a fabricated `max_force` of 0.0415 against a true median
of 0.088. `python scripts/audit_lemat_variants.py` reports what any variant on disk was
built with.

# Running code
Just running system Python won't work and can destroy the virtual environment! You must follow the machine-specific instructions! WyFormer runs on a range of machines (workstations, clusters, CI). Anything that is true for **one** host only must not leak into the top-level `README.md`, `AGENTS.md`
or `scripts/`:
 - human/agent-readable setup & run instructions -> `docs/platforms/<platform>/`,
   split by use case (e.g. `usage.md`, `environment.md`, `troubleshooting.md`)
 - scripts (container builds, env init, scheduler submission) -> `scripts/platforms/<platform>/`

`<platform>` is the short host or cluster name, e.g. `luna`, `zeus`.

Before setting up an environment, check whether `docs/platforms/<platform>/` exists
and follow it instead of improvising. When you learn something host-specific, record
it there rather than in the generic docs.
