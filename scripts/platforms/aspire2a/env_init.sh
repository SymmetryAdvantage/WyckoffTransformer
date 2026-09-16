#!/bin/bash
# Initialise the WyFormer environment on ASPIRE 2A for the current checkout (main or worktree).
#
# Usage:
#     scripts/platforms/aspire2a/env_init.sh
#
# What it does:
# 1. Links this checkout's CLAUDE.local.md to docs/platforms/aspire2a/agent_brief.md.
# 2. In a worktree (located in /home/users/nus/kna/scratch/WyFormer/worktrees/<name>):
#    creates a symlink .venv pointing to the shared venv at
#    /home/project/11001786/WyFormer/WyckoffTransformer/.venv.
#    (ASPIRE 2A has a slow filesystem, so worktrees reuse the main venv).
# 3. Verifies that ~/.config/wyformer/paths.env exists and prints resolved paths.
#
# Docs: docs/platforms/aspire2a/environment.md
set -euo pipefail

MAIN_REPO="/home/project/11001786/WyFormer/WyckoffTransformer"
# Physical paths on both sides: /home/project is a symlink to /data/projects, so the
# main checkout reached through either spelling must still compare equal.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
main_repo_physical=$(cd "$MAIN_REPO" && pwd -P)
cd "$repo_root"

# 1. Link agent brief
bash scripts/platforms/link_agent_brief.sh aspire2a

# 2. Setup venv link for worktrees
if [ "$repo_root" != "$main_repo_physical" ]; then
    echo "Worktree detected at: $repo_root"
    if [ -e "$repo_root/.venv" ] && [ ! -L "$repo_root/.venv" ]; then
        # ln -sfn onto a real directory would put the link inside it.
        echo "note: $repo_root/.venv is a real venv, not a link; leaving it in place."
    else
        ln -sfn "$MAIN_REPO/.venv" "$repo_root/.venv"
        echo "Reusing shared virtual environment:"
        echo "  .venv -> $MAIN_REPO/.venv"
    fi
    if [ ! -d "$MAIN_REPO/.venv" ]; then
        echo "note: $MAIN_REPO/.venv does not exist yet. Build it in the main repo."
    fi
else
    echo "Main repository checkout at: $repo_root"
    if [ -d "$repo_root/.venv" ]; then
        echo "Virtual environment present at: $repo_root/.venv"
    else
        echo "note: .venv not yet built in main repository."
        echo "      Build it inside Singularity via:"
        echo "        singularity run --nv ~/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif \\"
        echo "            env WYFORMER_PLATFORM=aspire2a bash scripts/build_singularity_venv.sh"
    fi
fi

# 3. Verify paths.env
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/wyformer/paths.env"
if [ -f "$CONFIG_FILE" ]; then
    echo "Storage configuration ($CONFIG_FILE):"
    # shellcheck source=scripts/wyformer_paths.sh
    . "$repo_root/scripts/wyformer_paths.sh"
    for key in WYFORMER_DATA WYFORMER_CACHE WYFORMER_RUNS WANDB_DIR; do
        val=$(wyformer_path "$key" 2>/dev/null || echo "<unset>")
        echo "  $key = $val"
    done
else
    echo "warning: $CONFIG_FILE not found! Storage paths will fall back." >&2
fi
