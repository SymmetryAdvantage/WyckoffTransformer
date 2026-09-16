#!/bin/bash
# ---------------------------------------------------------------------------
# Keep the shared cache store and the shared .venv read-only, so that nothing
# overwrites or deletes them by accident while the jobs of several worktrees read
# them at once.
#
#   bash scripts/platforms/aspire2a/store_lock.sh lock   [PATH...]
#   bash scripts/platforms/aspire2a/store_lock.sh open   DIR...
#   bash scripts/platforms/aspire2a/store_lock.sh unlock PATH...
#   bash scripts/platforms/aspire2a/store_lock.sh status [PATH...]
#
#   lock    remove every write bit under PATH, files and directories alike. Nothing
#           under it can then be overwritten, created, renamed or deleted -- `rm -rf`
#           included -- until it is unlocked.
#   open    make DIR itself writable again, and nothing inside it: new files can be
#           added to DIR while the files already there stay read-only. The way to add
#           a cache next to the ones jobs are reading.
#   unlock  give the owner write access to everything under PATH again. For
#           deliberately rebuilding or replacing what is there.
#   status  count the entries under PATH that anyone can still write to.
#
# With no PATH, lock and status act on the defaults: the cache store
# (WYFORMER_CACHE from ~/.config/wyformer/paths.env) and the main checkout's
# .venv. open and unlock always need an explicit path: widening access should name
# what it widens. Relock with `lock` when done.
#
# This guards against accidents, not against intent: the owner can always chmod.
# It works the same on GPFS (/home/project) and Lustre (/scratch), which is what
# was checked on 2026-09-16. See docs/platforms/aspire2a/usage.md.
# ---------------------------------------------------------------------------
set -euo pipefail

MAIN_REPO=/home/project/11001786/WyFormer/WyckoffTransformer

die() { echo "error: $*" >&2; exit 1; }

usage() { sed -n '3,/^# ---/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

default_paths() {
    # shellcheck source=scripts/wyformer_paths.sh
    . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../wyformer_paths.sh"
    wyformer_path WYFORMER_CACHE || die "cannot resolve WYFORMER_CACHE"
    echo "$MAIN_REPO/.venv"
}

# Symlinks resolved first: `chmod -R` given a symlink changes its target, and a
# worktree's .venv is a symlink to the main one.
resolve() {
    local p
    for p in "$@"; do
        [ -e "$p" ] || die "no such path: $p"
        readlink -f "$p"
    done
}

[ $# -ge 1 ] || { usage; exit 2; }
action=$1; shift

case "$action" in
    lock|status)
        if [ $# -eq 0 ]; then
            mapfile -t targets < <(default_paths)
        else
            mapfile -t targets < <(resolve "$@")
        fi ;;
    open|unlock)
        [ $# -ge 1 ] || die "$action needs an explicit path"
        mapfile -t targets < <(resolve "$@") ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown action '$action' (lock, open, unlock, status)" ;;
esac

for target in "${targets[@]}"; do
    case "$action" in
        lock)
            chmod -R a-w "$target"
            echo "locked    $target" ;;
        open)
            [ -d "$target" ] || die "$target is not a directory; open adds files to a directory"
            chmod u+w "$target"
            echo "opened    $target (existing files stay read-only)" ;;
        unlock)
            chmod -R u+w "$target"
            echo "unlocked  $target -- relock with: $0 lock $target" ;;
        status)
            writable=$(find "$target" -perm /222 -not -type l | wc -l)
            echo "$writable writable entries under $target" ;;
    esac
done
