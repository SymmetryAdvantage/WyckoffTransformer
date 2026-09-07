# Platform-specific setup

Read the directory for the host you are on, and only that one:

```bash
ls "docs/platforms/$(hostname -s)"
```

If it exists, it is authoritative for this machine: follow it instead of
improvising, and record anything host-specific you learn there rather than in
the top-level `README.md` or `AGENTS.md`. Matching scripts live in
`scripts/platforms/$(hostname -s)/`.

If it does not exist, this host is undocumented. Set it up, then add the
directory.

Do not read a sibling host's pages for guidance on this one. Nothing there
transfers: driver versions, container runtimes, filesystem layout, schedulers
and local package indices all differ, and a session cannot move between hosts.

Project and host environments might evolve - make sure to update this page as needed, especially when you encounter and solve issues.
