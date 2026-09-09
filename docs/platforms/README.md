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

Do not read a sibling host's pages for guidance on this one. Nothing there
transfers: driver versions, container runtimes, filesystem layout, schedulers
and local package indices all differ, and a session cannot move between hosts.

Keep the documentation in platforms/ updated - if you encounter that something there is wrong, update the documentation as needed. WyFormer codebase and environment might also evolve - again, keep the platform-specific documentation up to date.
