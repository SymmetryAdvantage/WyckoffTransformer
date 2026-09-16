#!/bin/bash
# Create a new git worktree for WyFormer on ASPIRE 2A in the standard location:
# /home/users/nus/kna/scratch/WyFormer/worktrees/<name>
#
# Usage:
#     scripts/platforms/aspire2a/create_worktree.sh <name> [branch-or-commit]
#
# Example:
#     scripts/platforms/aspire2a/create_worktree.sh feature-xyz
#     scripts/platforms/aspire2a/create_worktree.sh fix-eval main
#
# Docs: docs/platforms/aspire2a/environment.md
set -euo pipefail

NAME=${1:?usage: create_worktree.sh <name> [branch-or-commit]}
BRANCH=${2:-HEAD}

WORKTREE_BASE="/home/users/nus/kna/scratch/WyFormer/worktrees"
TARGET="$WORKTREE_BASE/$NAME"

if ! command -v git >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load git/2.39.2 2>/dev/null || true
fi

if [ -e "$TARGET" ]; then
    echo "error: destination already exists: $TARGET" >&2
    exit 1
fi

mkdir -p "$WORKTREE_BASE"

echo "Creating worktree at $TARGET from $BRANCH..."
git -c core.hooksPath=/dev/null worktree add "$TARGET" "$BRANCH"

echo "Initialising worktree environment..."
if [ ! -f "$TARGET/scripts/platforms/aspire2a/env_init.sh" ]; then
    mkdir -p "$TARGET/scripts/platforms/aspire2a"
    cp "$(dirname "${BASH_SOURCE[0]}")/env_init.sh" "$TARGET/scripts/platforms/aspire2a/env_init.sh"
    chmod +x "$TARGET/scripts/platforms/aspire2a/env_init.sh"
fi
(
    cd "$TARGET"
    bash scripts/platforms/aspire2a/env_init.sh
)

echo "Worktree ready at: $TARGET"
