# Sourced, not executed: the shell half of wyckoff_transformer.paths.
#
#   . "$REPO/scripts/wyformer_paths.sh"
#   RUNS_DIR=$(wyformer_path WYFORMER_RUNS "$REPO/runs") || exit 1
#
# Reads the same ${XDG_CONFIG_HOME:-~/.config}/wyformer/paths.env with the same
# rules, so a launcher and the Python it starts cannot disagree about where
# things are: the environment wins; then the config file, which is
# authoritative when it exists (a key it omits is an error, never the
# fallback); the caller's fallback only when there is no config file at all.
# The file is docker's --env-file format, so values are literal absolute paths.
# See docs/data_store.md.

wyformer_config_file() {
    printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/wyformer/paths.env"
}

# wyformer_config_value KEY < env-file
# Print KEY's value from env-file text on stdin (the last one, as docker takes
# it); exit status 1 when KEY is not set. Lines without '=' are skipped: in
# docker's format they pass a host variable through.
wyformer_config_value() {
    awk -v key="$1" '
        { sub(/^[[:space:]]+/, ""); sub(/[[:space:]]+$/, "") }
        $0 == "" || /^#/ || index($0, "=") == 0 { next }
        substr($0, 1, index($0, "=") - 1) == key {
            value = substr($0, index($0, "=") + 1); found = 1
        }
        END { if (found) print value; exit !found }'
}

# wyformer_path KEY [FALLBACK]
# Print where KEY points on this machine. Non-zero, with a message on stderr,
# when the config file exists but is incomplete or holds a non-literal path.
wyformer_path() {
    local key=$1 fallback=${2:-} file value
    if [ -n "${!key:-}" ]; then
        printf '%s\n' "${!key}"
        return 0
    fi
    file=$(wyformer_config_file)
    if [ ! -f "$file" ]; then
        printf '%s\n' "$fallback"
        return 0
    fi
    if ! value=$(wyformer_config_value "$key" < "$file"); then
        echo "error: $file exists, so it is authoritative, but does not set $key" >&2
        return 1
    fi
    case "$value" in
        *'$'* | [!/]* | '')
            echo "error: $file: $key=$value must be a literal absolute path" >&2
            return 1 ;;
    esac
    printf '%s\n' "$value"
}
