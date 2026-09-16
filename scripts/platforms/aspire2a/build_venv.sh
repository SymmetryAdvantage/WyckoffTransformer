#!/bin/bash
# ---------------------------------------------------------------------------
# (Re)build the one venv every checkout and job on ASPIRE 2A shares, at
# /home/project/11001786/WyFormer/WyckoffTransformer/.venv -- project storage, which
# scratch's purge policy does not reach.
#
#   bash scripts/platforms/aspire2a/build_venv.sh [--no-swap]
#
# Run from a login node (it downloads), from any checkout; it always builds for the
# main one. What it does:
#
#   1. builds a fresh venv beside the current one, at .venv.build-<timestamp>, inside
#      the container with scripts/build_singularity_venv.sh: dependencies resolved from
#      pyproject.toml plus the `relax` (ORB, MACE) and `dev` (pytest) extras,
#      relocatable so it can be renamed;
#   2. that script's import check has to pass, or nothing is swapped;
#   3. moves the current .venv (a directory or a symlink) aside to
#      .venv.previous-<timestamp> and renames the new one into its place -- two
#      renames, so no job ever sees a half-built venv;
#   4. makes the new venv read-only (store_lock.sh).
#
# --no-swap stops after step 2 and leaves the new venv at .venv.build-<timestamp>.
#
# Every job that starts after the swap uses the new venv, including the next link of
# a chain that started on the old one. A process already running keeps the modules it
# has imported, but anything it imports later comes from the new venv, so swap when
# that matters least. Remove .venv.previous-* once nothing runs on it:
# `store_lock.sh unlock` it first.
# ---------------------------------------------------------------------------
set -euo pipefail

MAIN_REPO=/home/project/11001786/WyFormer/WyckoffTransformer
SIF=${SIF:-/home/users/nus/kna/pytorch_2.14.0-cuda12.6-cudnn9-devel.sif}
UV_CACHE=${UV_CACHE_DIR:-/scratch/users/nus/kna/WyFormer/uv-cache}   # regenerable, so scratch
EXTRAS=${VENV_EXTRAS:-relax dev}

SWAP=1
case "${1:-}" in
    --no-swap) SWAP=0 ;;
    '') ;;
    -h|--help) sed -n '3,/^# ---/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
esac

if ! command -v singularity >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load singularity
fi

STAMP=$(date +%Y%m%d-%H%M%S)
NEW=".venv.build-$STAMP"
cd "$MAIN_REPO"
mkdir -p "$UV_CACHE"

echo "building $MAIN_REPO/$NEW (extras: ${EXTRAS:-none})"
SINGULARITY_NO_EVAL=1 singularity run --nv \
    --bind "/home/project,/data/projects" \
    "$SIF" env \
        REPO_DIR="$MAIN_REPO" VENV_DIR="$NEW" VENV_EXTRAS="$EXTRAS" VENV_RELOCATABLE=1 \
        UV_CACHE_DIR="$UV_CACHE" WYFORMER_PLATFORM=aspire2a \
        bash scripts/build_singularity_venv.sh

if [ "$SWAP" -eq 0 ]; then
    echo "built $MAIN_REPO/$NEW; not swapped (--no-swap)"
    exit 0
fi

if [ -e .venv ] || [ -L .venv ]; then
    mv .venv ".venv.previous-$STAMP"
    echo "moved the previous .venv aside: $MAIN_REPO/.venv.previous-$STAMP"
fi
mv "$NEW" .venv
echo "swapped in: $MAIN_REPO/.venv"

bash scripts/platforms/aspire2a/store_lock.sh lock "$MAIN_REPO/.venv"
