#!/usr/bin/env bash
# Run the WyFormer de novo ranking protocol sequentially for multiple runs on iapetus.
#
# Runs can be specified as CLI positional arguments, read from one or more files,
# or read from stdin.
#
# Examples:
#   scripts/platforms/iapetus/protocol.sh chemsys_e_form_sg_adamw_wsd-20260921-220630
#   scripts/platforms/iapetus/protocol.sh run1 run2 run3
#   scripts/platforms/iapetus/protocol.sh -f runs.txt
#   scripts/platforms/iapetus/protocol.sh runs.txt
#   cat runs.txt | scripts/platforms/iapetus/protocol.sh -
#   scripts/platforms/iapetus/protocol.sh -f runs.txt --skip-generate
#   scripts/platforms/iapetus/protocol.sh -f runs.txt --dry-run
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN="$SCRIPT_DIR/run.sh"

if [[ ! -x "$RUN" ]]; then
    echo "error: launcher not found or not executable at $RUN" >&2
    exit 1
fi

PYXTAL_CORES=6
DEVICES="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2"
OUTPUT_BASE="generated"
OUTPUT_DIR=""
STOP_ON_ERROR=0
DRY_RUN=0

declare -a RUNS=()
declare -a EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage:
    scripts/platforms/iapetus/protocol.sh [options] <run-id-1> [run-id-2 ...]
    scripts/platforms/iapetus/protocol.sh [options] -f <runs-file>
    scripts/platforms/iapetus/protocol.sh [options] <runs-file>
    cat <runs-file> | scripts/platforms/iapetus/protocol.sh [options] -

Runs the de novo ranking protocol sequentially for multiple runs on iapetus.

Arguments:
    <run-id> ...               One or more W&B run IDs to evaluate sequentially.
    <runs-file>                File containing run IDs (one per line, # comments ignored).

Options:
    -f, --file FILE            Read run IDs from FILE (or '-' for stdin). Repeatable.
    --pyxtal-cores N           CPU cores for PyXtal structure generation (default: 6).
    --devices DEVS             CUDA devices string (default: cuda:0,cuda:0,cuda:1,cuda:1,cuda:2).
    --output-base DIR          Base directory for outputs (default: generated).
                               Each run outputs to <DIR>/<run-id>/protocol.
    --output-dir DIR           Custom output directory or template ({run}, {run_id}, %s).
    --stop-on-error            Stop immediately if any run fails.
    --continue-on-error        Continue running remaining runs if a run fails (default).
    -n, --dry-run              Print the commands without executing them.
    -h, --help                 Show this help message.

Common wyformer-protocol-wandb options passed through:
    --skip-generate            Reuse existing wyckoff_genes.json.gz in output dir.
    --no-upload                Skip write-back to W&B.
    --n-genes N                Number of genes to generate and evaluate (default: 1000).
    --temperature T            Sampling temperature (default: 1.0).
    --stages STAGES            Comma-separated stages (screen,generate,relax,score).
    --mlip MLIP                Scoring MLIP (default: orb_conserv_inf).
    --prerelax-mlip MLIP       Pre-relaxation MLIP (e.g. nep89).
    --condition NAME=VALUE     Condition target, e.g. energy_above_hull=0.
    --condition-value VAL      Shorthand for condition value.
    --allow-fewer              Keep whatever valid genes were generated.
    --resume / --no-resume     Resume from existing progress (default: resume).
    --                         Pass any remaining arguments directly to wyformer-protocol-wandb.

Examples:
    # Single run:
    scripts/platforms/iapetus/protocol.sh chemsys_e_form_sg_adamw_wsd-20260921-220630

    # Multiple runs from CLI arguments:
    scripts/platforms/iapetus/protocol.sh run1 run2 run3

    # Runs read from file:
    scripts/platforms/iapetus/protocol.sh -f runs.txt
    scripts/platforms/iapetus/protocol.sh runs.txt

    # Preview commands without executing:
    scripts/platforms/iapetus/protocol.sh -f runs.txt --dry-run
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

read_runs_from_file() {
    local file="$1"
    if [[ "$file" == "-" ]]; then
        file="/dev/stdin"
    fi
    if [[ ! -e "$file" && "$file" != "/dev/stdin" ]]; then
        die "file not found: '$file'"
    fi
    if [[ ! -r "$file" ]]; then
        die "cannot read file: '$file'"
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line//$'\r'/}"
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        if [[ -z "$line" ]]; then
            continue
        fi
        for token in $line; do
            RUNS+=("$token")
        done
    done < "$file"
}

format_duration() {
    local seconds=$1
    local h=$((seconds / 3600))
    local m=$(((seconds % 3600) / 60))
    local s=$((seconds % 60))
    if (( h > 0 )); then
        printf "%dh %dm %ds" "$h" "$m" "$s"
    elif (( m > 0 )); then
        printf "%dm %ds" "$m" "$s"
    else
        printf "%ds" "$s"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -f|--file)
            [[ $# -ge 2 ]] || die "--file requires a file path"
            read_runs_from_file "$2"
            shift 2
            ;;
        --file=*)
            read_runs_from_file "${1#*=}"
            shift
            ;;
        --pyxtal-cores)
            [[ $# -ge 2 ]] || die "--pyxtal-cores requires an integer"
            PYXTAL_CORES="$2"
            shift 2
            ;;
        --pyxtal-cores=*)
            PYXTAL_CORES="${1#*=}"
            shift
            ;;
        --devices)
            [[ $# -ge 2 ]] || die "--devices requires a device string"
            DEVICES="$2"
            shift 2
            ;;
        --devices=*)
            DEVICES="${1#*=}"
            shift
            ;;
        --output-base)
            [[ $# -ge 2 ]] || die "--output-base requires a directory"
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --output-base=*)
            OUTPUT_BASE="${1#*=}"
            shift
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "--output-dir requires a directory or pattern"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --stop-on-error)
            STOP_ON_ERROR=1
            shift
            ;;
        --continue-on-error)
            STOP_ON_ERROR=0
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -)
            read_runs_from_file "-"
            shift
            ;;
        # Common wyformer-protocol-wandb boolean flags
        --skip-generate|--no-upload|--allow-fewer|--resume|--no-resume|\
        --release-symmetry|--no-release-symmetry|--rattle|--no-rattle|\
        --retry-failed|--allow-incomplete|\
        --prescreen-release-symmetry|--no-prescreen-release-symmetry|\
        --prescreen-rattle|--no-prescreen-rattle)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        # Common wyformer-protocol-wandb flags that take an argument
        --stages|--mlip|--prerelax-mlip|--temperature|--guidance-scale|\
        --condition|--condition-value|--n-genes|--pyxtal-timeout|\
        --pyxtal-tol-factor|--cores|--workers-per-device|--n-trials|\
        --fmax|--relax-timeout|--relax-from|--prerelax-fmax|\
        --prerelax-max-expansion|--trial-multiplier|--limit|\
        --prescreen-mlip|--prescreen-fmax|--prescreen-dedup|\
        --prescreen-energy-tol|--prescreen-select|--basinhop-mlip|\
        --basinhop-steps|--basinhop-temperature|--basinhop-stdev|\
        --basinhop-strain-stdev|--template-index|--template-candidates|\
        --reference-cache|--reference-splits|--reference-fingerprint-cache|\
        --lemat-cif-csv|--system-prior)
            [[ $# -ge 2 ]] || die "$1 requires an argument"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --from-artifact)
            if [[ $# -ge 2 && "$2" != -* && ! -f "$2" ]]; then
                EXTRA_ARGS+=("$1" "$2")
                shift 2
            else
                EXTRA_ARGS+=("$1")
                shift 1
            fi
            ;;
        --*=*|-*=*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        -*)
            die "unrecognized option: $1 (use -- to pass arbitrary options to wyformer-protocol-wandb)"
            ;;
        *)
            if [[ "$1" == "-" ]]; then
                read_runs_from_file "-"
            elif [[ -f "$1" ]]; then
                read_runs_from_file "$1"
            else
                RUNS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
    die "no run IDs specified (pass runs as arguments or via --file). Run with --help for usage."
fi

# Deduplicate runs while preserving order
declare -A seen_runs=()
declare -a unique_runs=()
for r in "${RUNS[@]}"; do
    if [[ -z "${seen_runs[$r]:-}" ]]; then
        seen_runs[$r]=1
        unique_runs+=("$r")
    else
        echo "info: skipping duplicate run ID '$r'" >&2
    fi
done
RUNS=("${unique_runs[@]}")

# Trap Ctrl+C cleanly
trap 'echo ""; echo "Batch cancelled by user (SIGINT). Exiting..."; exit 130' INT TERM

total_runs=${#RUNS[@]}

echo "======================================================================"
echo "WyFormer Protocol Batch Runner on iapetus"
echo "----------------------------------------------------------------------"
echo "Total runs     : $total_runs"
echo "Runs           : ${RUNS[*]}"
echo "PyXtal cores   : $PYXTAL_CORES"
echo "Devices        : $DEVICES"
echo "Output base    : $OUTPUT_BASE"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output dir pat : $OUTPUT_DIR"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra args     : ${EXTRA_ARGS[*]}"
fi
echo "Error policy   : $([[ $STOP_ON_ERROR -eq 1 ]] && echo 'stop on error' || echo 'continue on error')"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Mode           : DRY RUN"
fi
echo "======================================================================"

declare -a succeeded=()
declare -a failed=()
declare -a run_durations=()
declare -a run_exit_codes=()

batch_start=$(date +%s)

for idx in "${!RUNS[@]}"; do
    run_id="${RUNS[$idx]}"
    run_num=$((idx + 1))

    # Resolve output directory
    if [[ -n "$OUTPUT_DIR" ]]; then
        if [[ "$OUTPUT_DIR" =~ \{run\}|\{run_id\}|%s ]]; then
            out_dir="${OUTPUT_DIR//\{run_id\}/$run_id}"
            out_dir="${out_dir//\{run\}/$run_id}"
            out_dir="${out_dir//%s/$run_id}"
        elif [[ $total_runs -eq 1 ]]; then
            out_dir="$OUTPUT_DIR"
        else
            out_dir="${OUTPUT_DIR}/${run_id}/protocol"
        fi
    else
        out_dir="${OUTPUT_BASE}/${run_id}/protocol"
    fi

    cmd=(
        "$RUN"
        wyformer-protocol-wandb
        "$run_id"
        --output-dir "$out_dir"
        --pyxtal-cores "$PYXTAL_CORES"
        --devices "$DEVICES"
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo ""
    echo "======================================================================"
    echo "[$run_num/$total_runs] Starting protocol for run: $run_id"
    echo "Time       : $(date -Is)"
    echo "Output dir : $out_dir"
    echo "Command    : ${cmd[*]}"
    echo "======================================================================"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] Would execute:"
        echo "  ${cmd[*]}"
        succeeded+=("$run_id")
        run_durations+=("0s")
        run_exit_codes+=(0)
        continue
    fi

    t_start=$(date +%s)
    set +e
    "${cmd[@]}"
    rc=$?
    set -e
    t_end=$(date +%s)
    duration=$(( t_end - t_start ))
    dur_str=$(format_duration "$duration")

    run_durations+=("$dur_str")
    run_exit_codes+=("$rc")

    if [[ $rc -eq 0 ]]; then
        echo "----------------------------------------------------------------------"
        echo "[$run_num/$total_runs] SUCCESS: $run_id (took $dur_str)"
        echo "----------------------------------------------------------------------"
        succeeded+=("$run_id")
    else
        echo "----------------------------------------------------------------------"
        echo "[$run_num/$total_runs] FAILED: $run_id (exit code $rc, took $dur_str)" >&2
        echo "----------------------------------------------------------------------"
        failed+=("$run_id")
        if [[ $STOP_ON_ERROR -eq 1 ]]; then
            echo "error: stopping batch execution due to failure (--stop-on-error)" >&2
            break
        fi
    fi
done

batch_end=$(date +%s)
batch_duration=$(( batch_end - batch_start ))
batch_dur_str=$(format_duration "$batch_duration")

echo ""
echo "======================================================================"
echo "WyFormer Protocol Batch Summary on iapetus"
echo "======================================================================"
echo "Total runs processed: ${#succeeded[@]} succeeded, ${#failed[@]} failed (out of $total_runs)"
echo "Total batch time    : $batch_dur_str"
echo "----------------------------------------------------------------------"
for idx in "${!RUNS[@]}"; do
    r="${RUNS[$idx]}"
    if [[ $idx -lt ${#run_exit_codes[@]} ]]; then
        rc="${run_exit_codes[$idx]}"
        dur="${run_durations[$idx]}"
        if [[ $rc -eq 0 ]]; then
            printf "  [SUCCESS] %-50s (time: %s)\n" "$r" "$dur"
        else
            printf "  [FAILED]  %-50s (exit: %d, time: %s)\n" "$r" "$rc" "$dur"
        fi
    else
        printf "  [SKIPPED] %-50s\n" "$r"
    fi
done
echo "======================================================================"

if [[ ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
