#!/usr/bin/env bash
# rdr_paperbench_retry.sh — wrap rdr_paperbench.py with watchdog-kill retry.
#
# The RDR controller fires os._exit(124) when the per-cluster watchdog detects
# the Claude SDK aclose() deadlock.  This wrapper retries the run (up to
# RDR_MAX_RETRIES attempts, default 3) passing --resume on every retry so
# completed cluster checkpoints are preserved and not re-executed.
#
# Usage:
#   bash scripts/rdr_paperbench_retry.sh sequential-neural-score-estimation \
#       --provider anthropic --sandbox docker
#
# Environment:
#   RDR_MAX_RETRIES  maximum retry attempts (default: 3)
#
set -uo pipefail

max="${RDR_MAX_RETRIES:-3}"
attempts=0
extra_args=""

while [ "$attempts" -lt "$max" ]; do
    .venv/bin/python scripts/rdr_paperbench.py "$@" $extra_args
    ec=$?
    if [ "$ec" -ne 124 ]; then
        exit "$ec"
    fi
    echo "[rdr_paperbench_retry] watchdog kill (exit 124); attempt $((attempts + 1))/$max with --resume" >&2
    attempts=$((attempts + 1))
    extra_args="--resume"
done

echo "[rdr_paperbench_retry] gave up after $max attempts (all ended in watchdog kill)" >&2
exit 124
