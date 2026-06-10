#!/usr/bin/env python3
"""watch_run.py — live, crash-aware monitor for an RLM reproduction run.

Tails ``runs/<id>/{dashboard_events.jsonl, experiment_runs.jsonl, demo_status.json}``
and surfaces *progress* + *crashes with full diagnosis* so a failed run is seen and
understood immediately instead of being buried in JSONL. On any experiment failure it
prints a CRASH REPORT (failure_class + the failing commands + the full log/traceback
tail) and the actionable next step; on a terminal run status it prints the verdict and
exits.

Usage:
    .venv/bin/python scripts/watch_run.py [PROJECT_ID] [--interval S] [--exit-on-crash]
                                          [--max-min N] [--log PATH]
Defaults: PROJECT_ID=prj_09047604e591d969, --interval 15, runs until the run reaches a
terminal status (or --max-min elapses, 0 = unbounded).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Must cover EVERY terminal RunStatus (live_runs.py RunStatus Literal):
# killed (CLI signal handler) and interrupted (orphan sweep) were missing,
# so the monitor looped forever on those runs unless --max-min was set.
_TERMINAL = {"completed", "failed", "stopped", "killed", "interrupted"}


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _new_lines(p: Path, seen: int) -> tuple[list[str], int]:
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], seen
    return lines[seen:], len(lines)


def _summarize_event(e: dict) -> str | None:
    ev = e.get("event")
    if ev == "rubric_score":
        d = e.get("data", e)
        return f"  📊 rubric_score: {d.get('overall_score', d.get('score'))}"
    if ev == "run_warning":
        return f"  ⚠️  run_warning[{e.get('code')}]: {str(e.get('message',''))[:160]}"
    if ev == "candidate_proposed":
        return "  💡 candidate_proposed"
    if ev == "candidate_outcome":
        return "  ⚖️  candidate_outcome"
    if ev == "experiment_completed":
        d = e.get("data", {})
        ok = d.get("success")
        return f"  🧪 experiment_completed success={ok} metrics={d.get('metrics')}"
    if ev == "sub_rlm_spawned":
        return "  ↳ sub_rlm spawned"
    if ev == "run_complete":
        return "  ✅ run_complete"
    return None


def _crash_report(entry: dict) -> str:
    logs = (entry.get("logs") or "")
    out = [
        "",
        "🛑 ====================== EXPERIMENT CRASH ======================",
        f"   success      : {entry.get('success')}",
        f"   failure_class: {entry.get('failure_class')}",
        f"   error        : {str(entry.get('error') or '')[:200]}",
        f"   metrics      : {entry.get('metrics')}",
        "   --- log / traceback tail (last 2500 chars) ---",
        logs[-2500:] if logs else "   (no logs captured)",
        "🛑 ============================================================",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_id", nargs="?", default="prj_09047604e591d969")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--exit-on-crash", action="store_true")
    ap.add_argument("--max-min", type=float, default=0.0, help="0 = unbounded")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--log", default="", help="extra run log to scan for tracebacks")
    a = ap.parse_args()

    rd = Path(a.runs_root) / a.project_id
    devents, eruns, dstatus = (rd / "dashboard_events.jsonl",
                               rd / "experiment_runs.jsonl",
                               rd / "demo_status.json")
    print(f"watch_run: monitoring {rd} (interval={a.interval}s, exit_on_crash={a.exit_on_crash})", flush=True)
    seen_ev, seen_exp = 0, 0
    last_status, crashes, tick = None, 0, 0
    deadline = time.monotonic() + a.max_min * 60 if a.max_min else None

    while True:
        # 1) experiment results — the crash surface
        new_exp, seen_exp = _new_lines(eruns, seen_exp)
        for ln in new_exp:
            try:
                entry = json.loads(ln)
            except Exception:
                continue
            if entry.get("success") is False:
                crashes += 1
                print(_crash_report(entry), flush=True)
                if a.exit_on_crash:
                    print(f"watch_run: --exit-on-crash → stopping after crash #{crashes}", flush=True)
                    return 2
            else:
                print(f"  ✅ experiment OK metrics={entry.get('metrics')}", flush=True)

        # 2) dashboard progress + warnings
        new_ev, seen_ev = _new_lines(devents, seen_ev)
        for ln in new_ev:
            try:
                e = json.loads(ln)
            except Exception:
                continue
            s = _summarize_event(e)
            if s:
                print(s, flush=True)

        # 3) status — terminal → done
        st = _read_json(dstatus) or {}
        status = st.get("status")
        if status != last_status:
            print(f"  • status → {status} (verdict={st.get('verdict')}, process={st.get('process_status')})", flush=True)
            last_status = status
        if status in _TERMINAL:
            print(f"watch_run: run reached terminal status '{status}' "
                  f"(error={st.get('error')!r}); crashes seen={crashes}. Exiting.", flush=True)
            return 0

        tick += 1
        if tick % 4 == 0:  # periodic heartbeat
            print(f"  … events={seen_ev} experiments={seen_exp} crashes={crashes} status={status}", flush=True)
        if deadline and time.monotonic() > deadline:
            print(f"watch_run: --max-min reached; crashes seen={crashes}. Exiting.", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
