# Platform-specific setup

Read the directory for the host you are on, and only that one:

```bash
ls "docs/platforms/$(hostname -s)"
```

If it exists, it is authoritative for this machine: follow it instead of
improvising, and record anything host-specific you learn there rather than in
the top-level `README.md` or `AGENTS.md`. Matching scripts live in
`scripts/platforms/$(hostname -s)/`.

On a **cluster**, the node name is not the platform name -- you get a different
node every job. The directory is named after the cluster, and the mapping is:

| Node name | Directory |
| --- | --- |
| `asp2a-*` (login or compute) | `aspire2a` |

If neither the host name nor a cluster entry above matches, this host is
undocumented. Set it up, then add the directory -- and a row here if it is a
cluster.

## The agent brief: `agent_brief.md` -> `CLAUDE.local.md`

Every platform directory has an `agent_brief.md`: a screenful of what an agent
must know before running anything on that host -- how to invoke Python, which GPU
to take, what never to run. The pages beside it stay the full reference.

Claude Code loads `<repo>/CLAUDE.local.md` into every session, and that file is
gitignored, so each checkout carries a relative symlink to its host's brief. The
platform's environment initialisation makes it, by calling

```bash
scripts/platforms/link_agent_brief.sh <platform>
```

-- `scripts/platforms/zeus/env_init.sh`, `scripts/platforms/luna/build_venv.sh`
and `scripts/platforms/iapetus/build_venv.sh` do so with their own name;
`scripts/build_singularity_venv.sh` builds venvs for distributable container
images rather than for one host, so it links only when given
`WYFORMER_PLATFORM=<platform>`. Run
the helper by hand to link a checkout without rebuilding its environment. It is
idempotent and leaves a hand-written `CLAUDE.local.md` alone. Claude Code copies
the link into new worktrees through `.worktreeinclude`.

The brief is not named `CLAUDE.local.md` inside `docs/platforms/` because Claude
Code picks up memory files in subdirectories when it reads files there. A tracked
`docs/platforms/luna/CLAUDE.local.md` could then feed luna's instructions to an
agent on zeus that merely opened a luna page.

A new platform needs an `agent_brief.md` and a call to the helper in its
initialisation. When a platform's pages change in a way an agent must act on,
update the brief in the same commit.

## Other hosts' pages

Do not read a sibling host's pages for guidance on this one. Nothing there
transfers: driver versions, container runtimes, filesystem layout, schedulers
and local package indices all differ, and a session cannot move between hosts.

Keep the documentation in platforms/ updated - if you encounter that something there is wrong, update the documentation as needed. WyFormer codebase and environment might also evolve - again, keep the platform-specific documentation up to date.
