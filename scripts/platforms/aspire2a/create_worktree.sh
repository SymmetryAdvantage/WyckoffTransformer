#!/bin/bash
# Create a git worktree for WyFormer on ASPIRE 2A in the standard location,
# /home/users/nus/kna/scratch/WyFormer/worktrees/<name>, on a branch of the same name,
# and initialise it.
#
# Usage:
#     scripts/platforms/aspire2a/create_worktree.sh <name> [start-point]
#
#   <name>         the worktree directory and its branch. An existing branch of that
#                  name is checked out; otherwise a new one is created at start-point.
#   [start-point]  commit or branch the new branch starts from (default: HEAD of the
#                  checkout this is run from). Refused for an existing branch.
#
# Example:
#     scripts/platforms/aspire2a/create_worktree.sh sg-prediction main
#     cd /home/users/nus/kna/scratch/WyFormer/worktrees/sg-prediction
#     claude
#
# A worktree made here is never touched by Claude Code's cleanup, unlike one made by
# `claude --worktree`, so it can outlive the session that started a training chain
# from it. Docs: docs/platforms/aspire2a/usage.md#working-in-a-git-worktree
set -euo pipefail

NAME=${1:?usage: create_worktree.sh <name> [start-point]}
START=${2:-}

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

# core.hooksPath: the repository's LFS hooks call git-lfs, which is not installed here.
if git show-ref --verify --quiet "refs/heads/$NAME"; then
    if [ -n "$START" ]; then
        echo "error: branch '$NAME' already exists; a start-point only applies to a new branch" >&2
        exit 1
    fi
    echo "Creating worktree at $TARGET on the existing branch $NAME..."
    git -c core.hooksPath=/dev/null worktree add "$TARGET" "$NAME"
else
    echo "Creating worktree at $TARGET on a new branch $NAME from ${START:-HEAD}..."
    git -c core.hooksPath=/dev/null worktree add -b "$NAME" "$TARGET" "${START:-HEAD}"
fi

# A branch that predates the worktree tooling has neither this initialisation nor the
# launcher that imports a worktree's own code; say so rather than run the main checkout's.
if [ ! -f "$TARGET/scripts/platforms/aspire2a/env_init.sh" ] \
        || ! grep -q PYTHONPATH "$TARGET/scripts/platforms/aspire2a/run_in_singularity.sh"; then
    echo "error: $NAME predates the ASPIRE 2A worktree tooling: its run_in_singularity.sh" >&2
    echo "error: would import the main checkout's code. The worktree was created but not" >&2
    echo "error: initialised; merge main into $NAME there, then run scripts/platforms/aspire2a/env_init.sh." >&2
    exit 1
fi

echo "Initialising worktree environment..."
(
    cd "$TARGET"
    bash scripts/platforms/aspire2a/env_init.sh
)

echo "Worktree ready at: $TARGET"
echo "Start Claude Code there with:  cd $TARGET && claude"
