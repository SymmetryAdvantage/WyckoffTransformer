#!/usr/bin/env bash
# Run one step of a batch pipeline; if it ends abnormally, hand it to Claude Code to debug,
# then try it again.
#
#   bash scripts/platforms/aspire2a/self_debug.sh <step-name> <command> [args...]
#
# The step's output is teed to $SELF_DEBUG_DIR/<step>.attempt<N>.log. On a non-zero exit,
# `claude -p` is started in this checkout in permission mode `auto`, told what failed and
# where the log is, and asked to find the cause, fix it here, test the fix and commit it
# (train_in_pbs.sh runs committed code only). Its transcript goes to
# $SELF_DEBUG_DIR/<step>.claude<N>.log. The step is then re-run. After SELF_DEBUG_MAX_FIXES
# (default 2) repairs the step's last exit code is returned, so a pipeline stops instead of
# looping. A nested call never spawns Claude: SELF_DEBUG_DEPTH guards that.
set -uo pipefail

step=${1:?usage: self_debug.sh <step-name> <command> [args...]}
shift
REPO_DIR=$(git rev-parse --show-toplevel)
SELF_DEBUG_DIR=${SELF_DEBUG_DIR:-/scratch/users/nus/kna/WyFormer/logs/self_debug/${PBS_JOBID:-local}}
MAX_FIXES=${SELF_DEBUG_MAX_FIXES:-2}
CLAUDE_BIN=${CLAUDE_BIN:-$HOME/.local/bin/claude}
mkdir -p "$SELF_DEBUG_DIR"

attempt=0
while :; do
    attempt=$((attempt + 1))
    log="$SELF_DEBUG_DIR/$step.attempt$attempt.log"
    echo "[self_debug] $(date -Is) step '$step' attempt $attempt: $*" | tee "$log"
    "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        echo "[self_debug] step '$step' succeeded on attempt $attempt"
        exit 0
    fi
    echo "[self_debug] step '$step' exited $rc (log: $log)"
    if [ "${SELF_DEBUG_DEPTH:-0}" -ge 1 ] || [ "$attempt" -gt "$MAX_FIXES" ]; then
        echo "[self_debug] giving up on '$step' after $attempt attempts"
        exit "$rc"
    fi
    if [ ! -x "$CLAUDE_BIN" ]; then
        echo "[self_debug] no claude at $CLAUDE_BIN; cannot self-debug"
        exit "$rc"
    fi

    prompt=$(cat <<EOF
You are debugging a failed step of an unattended PBS batch pipeline on ASPIRE 2A. Nobody is
watching; make the step work, or stop and explain why it cannot.

- Job: ${PBS_JOBID:-not in PBS}, on $(hostname), checkout $REPO_DIR (branch $(git -C "$REPO_DIR" branch --show-current), commit $(git -C "$REPO_DIR" rev-parse --short HEAD)).
- Step: '$step', attempt $attempt, exit code $rc.
- Command: $*
- Full output: $log (read it first; the end usually has the traceback).

Rules:
- Read CLAUDE.local.md / AGENTS.md in the checkout and follow them. Python runs only inside the
  container: bash scripts/platforms/aspire2a/run_in_singularity.sh python ...; never uv/pip/pytest
  on the host, never touch the shared .venv.
- Fix the cause in THIS checkout only. Do not edit other worktrees or the main checkout, do not
  qdel or modify any job, do not unlock or rebuild any cache other than the one this step builds.
- Test the fix (the relevant unit tests under src/, and a cheap re-run of the failing piece if
  possible), then git commit it here with a message explaining the cause. The wrapper re-runs the
  step as soon as you exit; train_in_pbs.sh refuses uncommitted changes.
- If the failure is transient (node, filesystem, network, W&B), fix nothing and just say so.
- Finish with a short report: cause, fix, commit, and anything a human must check.
EOF
)
    echo "[self_debug] $(date -Is) starting Claude to debug '$step' (transcript: $SELF_DEBUG_DIR/$step.claude$attempt.log)"
    (cd "$REPO_DIR" && SELF_DEBUG_DEPTH=1 "$CLAUDE_BIN" -p "$prompt" --permission-mode auto) \
        > "$SELF_DEBUG_DIR/$step.claude$attempt.log" 2>&1
    echo "[self_debug] Claude exited $?; now at commit $(git -C "$REPO_DIR" rev-parse --short HEAD); retrying '$step'"
done
