#!/usr/bin/env bash
# Fail if any of the Phase-D-migrated hardcoded patterns regress.
#
# Each pattern below corresponds to a constant that used to live in the
# frontend and is now sourced from the backend topology / per-run state /
# user preferences. Re-introducing the literal loses that value.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
patterns=(
  'const NODES: WorkflowNode\[\] = \['
  'const EDGES: Array<\[string'
  'INTERNAL_AGENT_NAMES'
  'DEMO_AGENT_NAMES'
  'const GATE_COORDS'
  '"opt", "bb", "aug", "hor", "div"'
  '<option value="sonnet">'
  '(^|[^-])width: 1200px'
  '(^|[^-])height: 640px'
  '1740'
)

for p in "${patterns[@]}"; do
    hits=$(grep -rnE "$p" frontend/src 2>/dev/null \
        | grep -v "node_modules\|\.test\.\|__fixtures__\|/\*\|^[^:]*://" \
        | grep -v "//.*$p" || true)
    if [ -n "$hits" ]; then
        echo "forbidden pattern reappeared: $p"
        echo "$hits"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "Phase D regression - see hits above. Each pattern is one we"
    echo "migrated to API or run-state. Re-introducing it loses the value."
    exit 1
fi
echo "no-hardcoding sweep clean"
