#!/bin/bash
# Replicate the data/cache stores between two machines.
#
#   store_sync.sh <remote> [pull|push] [--go] [subpath ...]
#
# `<remote>` is an ssh host alias -- put the addresses in ~/.ssh/config so this
# script carries a name rather than a link-local scope-id or a ProxyJump chain.
# See docs/data_store.md, and docs/platforms/<host>/ for how to reach each one.
#
# Defaults are the safe ones: it pulls, and it only reports. Pass `push` to send,
# and `--go` to actually transfer. Subpaths are store-relative and restrict the
# transfer, e.g. `data/mp_20 cache/mp_20`; with none, both stores are synced.
#
# Each side's store comes from that machine's own config file,
# ~/.config/wyformer/paths.env (docs/data_store.md), so there is one source of
# truth per machine and no host specifics in the repository. The remote's file
# is read with a plain `cat`: nothing depends on which shell profile it sources.
#
# Only the stores move. Datasets tracked by git -- plainly or through LFS --
# reach every machine through git and stay in <repo>/data; `runs` is per-machine.
set -euo pipefail

die() { printf 'store_sync: %s\n' "$*" >&2; exit 1; }

remote=""
direction="pull"
go=0
subpaths=()

for arg in "$@"; do
    case "$arg" in
        pull|push) direction="$arg" ;;
        --go) go=1 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        -*) die "unknown option $arg" ;;
        *) if [[ -z "$remote" ]]; then remote="$arg"; else subpaths+=("$arg"); fi ;;
    esac
done
[[ -n "$remote" ]] || die "no remote given; try --help"

for sub in "${subpaths[@]:-}"; do
    [[ -z "$sub" ]] && continue
    case "$sub" in
        data|data/*|cache|cache/*) ;;
        *) die "subpath must start with data/ or cache/: $sub" ;;
    esac
done

# shellcheck source=scripts/wyformer_paths.sh
. "$(dirname "$(readlink -f "$0")")/wyformer_paths.sh"

# Local roots. Deliberately no fallback to the checkout: syncing into the wrong
# directory is expensive to undo, so an unconfigured machine is an error.
local_data=$(wyformer_path WYFORMER_DATA) || exit 1
local_cache=$(wyformer_path WYFORMER_CACHE) || exit 1
[[ -n "$local_data" && -n "$local_cache" ]] \
    || die "no store configured here: create $(wyformer_config_file); see docs/data_store.md"

echo "reading the store config on $remote ..."
remote_config=$(ssh "$remote" 'cat "${XDG_CONFIG_HOME:-$HOME/.config}/wyformer/paths.env"') \
    || die "cannot read ~/.config/wyformer/paths.env on $remote; create it there first"
remote_data=$(wyformer_config_value WYFORMER_DATA <<< "$remote_config") \
    || die "$remote's paths.env does not set WYFORMER_DATA"
remote_cache=$(wyformer_config_value WYFORMER_CACHE <<< "$remote_config") \
    || die "$remote's paths.env does not set WYFORMER_CACHE"
for value in "$remote_data" "$remote_cache"; do
    [[ "$value" == /* && "$value" != *'$'* ]] || die "$remote's paths.env: '$value' is not a literal absolute path"
done

# -a keeps the relative symlinks in data/ as symlinks; -L would dereference the
#    tolerance-variant aliases and inflate the transfer by gigabytes.
# -W whole-file: every large file here is compressed or binary, so rsync's delta
#    algorithm cannot win and only costs a read of both sides. No -z either.
# --update never overwrites a file that is newer on the receiver. Without
#    versioning a two-sided diff cannot say which side is right, so this plus
#    pull-by-default plus dry-run-by-default is the whole safety story.
# --partial-dir resumes a dropped transfer. It is mutually exclusive with
#    --inplace; see docs/data_store.md for when to swap them.
# No --delete, ever, by default: it turns a wrong-direction sync into data loss.
rsync_flags=(-aHW --update --info=progress2 --human-readable
             --partial-dir=.rsync-partial)
[[ $go -eq 1 ]] || rsync_flags+=(--dry-run --itemize-changes)

stamp=$(date +%Y%m%d-%H%M%S)

sync_one() {
    local store="$1" local_root="$2" remote_root="$3" rel="$4"
    local src dst backup_root
    # Trailing slashes matter: <src>/ copies the contents into <dst>.
    if [[ "$direction" == "pull" ]]; then
        src="$remote:${remote_root%/}/${rel}"
        dst="${local_root%/}/${rel}"
        backup_root="${local_root%/}/.rsync-backup/$stamp"
    else
        src="${local_root%/}/${rel}"
        dst="$remote:${remote_root%/}/${rel}"
        backup_root="${remote_root%/}/.rsync-backup/$stamp"
    fi
    src="${src%/}/"
    [[ $go -eq 1 && "$direction" == "pull" ]] && mkdir -p "$dst"
    printf '\n=== %s %s: %s -> %s\n' "$direction" "$store" "$src" "$dst"
    rsync "${rsync_flags[@]}" --backup --backup-dir="$backup_root" "$src" "$dst"
}

if [[ ${#subpaths[@]} -eq 0 ]]; then
    sync_one data "$local_data" "$remote_data" ""
    sync_one cache "$local_cache" "$remote_cache" ""
else
    for sub in "${subpaths[@]}"; do
        case "$sub" in
            data|data/*) sync_one data "$local_data" "$remote_data" "${sub#data}" ;;
            cache|cache/*) sync_one cache "$local_cache" "$remote_cache" "${sub#cache}" ;;
            *) die "subpath must start with data/ or cache/: $sub" ;;
        esac
    done
fi

if [[ $go -eq 0 ]]; then
    printf '\nThat was a dry run. Re-run with --go to transfer.\n'
fi
