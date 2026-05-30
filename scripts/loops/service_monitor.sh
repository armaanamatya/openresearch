#!/usr/bin/env bash
# Service-validation background monitor (2026-05-30 long-run validation session).
# Appends one timestamped health line every MONITOR_INTERVAL seconds to
# logs/service-validation/operator.log. Fail-soft: a probe failure logs an ALERT,
# never crashes the loop. No secrets are read or printed.
#
# Launch:  MONITOR_INTERVAL=90 bash scripts/loops/service_monitor.sh   (run_in_background)
# Stop:    kill the launched PID (single PID — do NOT killpg/pkill).
set -u

ROOT="/Volumes/CS_Stuff/openresearch"
LOG="$ROOT/logs/service-validation/operator.log"
INTERVAL="${MONITOR_INTERVAL:-90}"
PY="$ROOT/.venv/bin/python"

mkdir -p "$(dirname "$LOG")"

probe_http() {
  "$PY" - "$1" <<'PYEOF'
import sys, urllib.request, urllib.error
try:
    with urllib.request.urlopen(sys.argv[1], timeout=4) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:
    print("DOWN")
PYEOF
}

listening() { lsof -i :"$1" -P -n 2>/dev/null | grep -q LISTEN && echo up || echo DOWN; }

scan_active_runs() {
  "$PY" - <<'PYEOF'
import json, glob, os
from datetime import datetime, timezone
now = datetime.now(timezone.utc).timestamp()
out = []
for p in glob.glob("/Volumes/CS_Stuff/openresearch/runs/*/demo_status.json"):
    try:
        s = json.load(open(p))
    except Exception:
        continue
    if s.get("status") in ("queued", "running"):
        ev = os.path.join(os.path.dirname(p), "dashboard_events.jsonl")
        gap = "?"
        try:
            gap = str(int(now - os.path.getmtime(ev))) + "s"
        except Exception:
            pass
        out.append(f"{os.path.basename(os.path.dirname(p))}:evgap={gap}")
print(",".join(out) if out else "none")
PYEOF
}

echo "$(date '+%F %T') monitor started (interval=${INTERVAL}s)" >> "$LOG"
while true; do
  ts="$(date '+%F %T')"
  be_port="$(listening 8000)"
  fe_port="$(listening 3000)"
  be_health="$(probe_http http://127.0.0.1:8000/health)"
  active="$(scan_active_runs)"
  echo "$ts backend[port=$be_port health=$be_health] frontend[port=$fe_port] active_runs[$active]" >> "$LOG"
  [ "$be_port" = DOWN ] || [ "$be_health" = DOWN ] && echo "$ts ALERT backend unhealthy (port=$be_port health=$be_health)" >> "$LOG"
  [ "$fe_port" = DOWN ] && echo "$ts ALERT frontend down (port 3000 not listening)" >> "$LOG"
  sleep "$INTERVAL"
done
