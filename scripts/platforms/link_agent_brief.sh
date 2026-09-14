#!/bin/bash
# Link this checkout's CLAUDE.local.md to the platform's tracked agent brief.
#
# Usage:
#     scripts/platforms/link_agent_brief.sh <platform>
#
# Claude Code and Gemini CLI / Antigravity (via .gemini/settings.json) load
# <repo>/CLAUDE.local.md into every session started in the checkout. The file
# is gitignored, so each checkout -- the main one and every worktree -- needs
# its own; the content lives in docs/platforms/<platform>/agent_brief.md, which
# is tracked, and this makes the former a relative symlink to the latter.
# Relative, so the link also resolves inside a container that mounts the
# checkout at another path.
#
# Each platform's environment initialisation calls this with its own name. It is
# idempotent and never overwrites a CLAUDE.local.md someone wrote by hand.
#
# Docs: docs/platforms/README.md
set -euo pipefail

platform=${1:?usage: link_agent_brief.sh <platform>}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
target="docs/platforms/${platform}/agent_brief.md"
link="${repo_root}/CLAUDE.local.md"

if [[ ! -f "${repo_root}/${target}" ]]; then
    echo "error: ${target} does not exist; write it before initialising ${platform}" >&2
    exit 1
fi

# A regular file is replaced only when it is a copy of the brief, as a worktree
# tool that dereferences symlinks would leave. Anything else is someone's notes.
if [[ -e "$link" && ! -L "$link" ]] && ! cmp -s "$link" "${repo_root}/${target}"; then
    echo "warning: ${link} is a hand-written file; leaving it." >&2
    echo "         Move it aside and re-run to link ${target}." >&2
    exit 0
fi

ln -sfn "$target" "$link"
echo "CLAUDE.local.md -> ${target}"
