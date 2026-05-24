#!/usr/bin/env python3
"""Live TUI aggregator for ReproLab runs.

Usage:
    python scripts/run_monitor.py <project_id> [<project_id> ...]
    python scripts/run_monitor.py --auto    # finds latest active runs

One screen, four panes:
    [header]  active processes + pod count + last event ages per run
    [events]  meaningful SSE events from runs/<id>/dashboard_events.jsonl
              (skips heartbeat / iteration_heartbeat so noise stays out)
    [pod]     last 8 lines of <id>/code/outputs/<run_id>/exec.log when present
    [warn]    sticky pane for run_warning + run_complete events

Refresh: 1 s. Quit: Ctrl-C.

Pure stdlib — no curses, no textual, no asyncio.  Uses ANSI escape codes for
in-place updates so it works in any modern terminal (and over SSH).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ANSI escape codes
ESC = "\x1b["
CLEAR = ESC + "2J" + ESC + "H"
HIDE_CURSOR = ESC + "?25l"
SHOW_CURSOR = ESC + "?25h"
BOLD = ESC + "1m"
DIM = ESC + "2m"
RESET = ESC + "0m"
RED = ESC + "31m"
GREEN = ESC + "32m"
YELLOW = ESC + "33m"
BLUE = ESC + "34m"
MAGENTA = ESC + "35m"
CYAN = ESC + "36m"
GREY = ESC + "90m"

EVENTS_DIR = Path("runs")

# Events worth surfacing (drop pure heartbeat noise).
MEANINGFUL_EVENTS = {
    "primitive_call",
    "experiment_completed",
    "rubric_score",
    "run_complete",
    "run_failed",
    "run_warning",
    "gpu_resolved",
    "gpu_escalated",
    "candidate_proposed",
    "candidate_outcome",
    "sub_rlm_spawned",
    "sub_rlm_complete",
    "worker_report_started",
    "worker_report_completed",
}

EVENT_COLORS = {
    "run_warning": RED,
    "run_failed": RED,
    "run_complete": MAGENTA,
    "rubric_score": GREEN,
    "experiment_completed": CYAN,
    "gpu_resolved": BLUE,
    "gpu_escalated": YELLOW,
    "candidate_proposed": BLUE,
    "candidate_outcome": BLUE,
    "sub_rlm_spawned": GREY,
    "sub_rlm_complete": GREY,
    "primitive_call": GREY,
    "worker_report_started": DIM + CYAN,
    "worker_report_completed": DIM + CYAN,
}


def _now() -> float:
    return time.time()


def _short_ts(iso_or_unix: Any) -> str:
    if isinstance(iso_or_unix, (int, float)):
        return datetime.fromtimestamp(float(iso_or_unix), tz=timezone.utc).strftime("%H:%M:%S")
    if isinstance(iso_or_unix, str):
        try:
            return datetime.fromisoformat(iso_or_unix.replace("Z", "+00:00")).strftime("%H:%M:%S")
        except ValueError:
            return iso_or_unix[-13:-5] if len(iso_or_unix) > 14 else iso_or_unix
    return "??:??:??"


def _age_seconds(path: Path) -> float | None:
    try:
        return _now() - path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return None


def _summarise_event(line: str) -> str | None:
    """Return a one-line summary of a meaningful event, or None to skip."""
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    ev = d.get("event") or d.get("type") or ""
    primitive = d.get("primitive") or ""
    # Skip noise
    if ev == "iteration_heartbeat":
        return None
    if ev == "primitive_call" and primitive == "heartbeat":
        return None
    if ev not in MEANINGFUL_EVENTS:
        return None
    ts = d.get("ts") or d.get("timestamp") or ""
    color = EVENT_COLORS.get(ev, "")
    data = d.get("data") or d
    parts = [f"{_short_ts(ts):>8} {color}{ev:<22}{RESET}"]
    if primitive:
        status = d.get("status") or ""
        parts.append(f"{primitive}({status})")
    if ev == "rubric_score":
        parts.append(f"score={d.get('score', 0):.3f}/{d.get('target', 0):.2f}")
    if ev == "experiment_completed":
        s = data.get("success")
        err = (data.get("error") or "")[:60]
        parts.append(f"success={s}{(' err='+err) if err else ''}")
    if ev == "gpu_escalated":
        parts.append(f"{data.get('from_sku')}→{data.get('to_sku')} ({data.get('reason')})")
    if ev == "gpu_resolved":
        parts.append(f"{data.get('short_name')} {data.get('vram_gb')}GB")
    if ev == "run_warning":
        stale = data.get("stale_seconds") or 0
        reason = data.get("reason") or "?"
        parts.append(f"{reason} stale={stale:.0f}s")
    if ev == "run_complete":
        parts.append(f"status={d.get('status')}")
    return "  ".join(parts)


def _read_new_lines(fh, buf: list[str]) -> None:
    while True:
        line = fh.readline()
        if not line:
            return
        s = _summarise_event(line.strip())
        if s:
            buf.append(s)


def _latest_run_dir(project: Path) -> Path | None:
    outputs = project / "code" / "outputs"
    if not outputs.is_dir():
        return None
    subdirs = [p for p in outputs.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _tail_file(path: Path, n: int = 8) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return fh.readlines()[-n:]
    except (OSError, FileNotFoundError):
        return []


def _render(projects: list[Path], event_buf: dict[str, list[str]]) -> str:
    out: list[str] = []
    # Header
    out.append(f"{BOLD}ReproLab Live Monitor{RESET}  ({datetime.now().strftime('%H:%M:%S')})  Ctrl-C to quit\n")
    out.append("─" * 100)
    # Per-run status
    for project in projects:
        events_file = project / "dashboard_events.jsonl"
        last_ev_age = _age_seconds(events_file)
        run_dir = _latest_run_dir(project)
        exec_log = (run_dir / "exec.log") if run_dir else None
        exec_age = _age_seconds(exec_log) if exec_log else None
        n_events = 0
        try:
            with events_file.open() as fh:
                n_events = sum(1 for _ in fh)
        except OSError:
            pass
        line = f"  {BOLD}{project.name}{RESET}  events={n_events}"
        if last_ev_age is not None:
            color = GREEN if last_ev_age < 60 else (YELLOW if last_ev_age < 300 else RED)
            line += f"  last_ev={color}{last_ev_age:.0f}s{RESET}"
        if exec_age is not None:
            color = GREEN if exec_age < 60 else (YELLOW if exec_age < 300 else RED)
            line += f"  exec_log={color}{exec_age:.0f}s{RESET}"
        else:
            line += f"  exec_log={DIM}—{RESET}"
        out.append(line)
    out.append("─" * 100)
    # Events pane — interleaved across all projects
    out.append(f"{BOLD}Events:{RESET}")
    rows: list[tuple[float, str, str]] = []
    for project in projects:
        for ev in event_buf.get(project.name, [])[-25:]:
            rows.append((_now(), project.name[:18], ev))
    for ts, tag, ev in rows[-25:]:
        out.append(f"  [{tag:<18}] {ev}")
    if not rows:
        out.append(f"  {DIM}(no meaningful events yet){RESET}")
    out.append("")
    # Pod exec tails
    for project in projects:
        run_dir = _latest_run_dir(project)
        if run_dir is None:
            continue
        exec_log = run_dir / "exec.log"
        if not exec_log.exists():
            continue
        tail = _tail_file(exec_log, n=6)
        if not tail:
            continue
        out.append(f"{BOLD}exec.log ({project.name}/{run_dir.name}):{RESET}")
        for ln in tail:
            ln = ln.rstrip()
            if not ln:
                continue
            out.append(f"  {DIM}{ln[:120]}{RESET}")
        out.append("")
    return "\n".join(out)


def _resolve_auto() -> list[Path]:
    """Find ALL project dirs whose dashboard_events.jsonl was touched < 5 min ago."""
    if not EVENTS_DIR.is_dir():
        return []
    candidates: list[tuple[float, Path]] = []
    for d in EVENTS_DIR.iterdir():
        if not d.is_dir():
            continue
        events = d / "dashboard_events.jsonl"
        if not events.exists():
            continue
        age = _age_seconds(events)
        if age is None or age > 300:
            continue
        candidates.append((age, d))
    candidates.sort()
    return [d for _, d in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_ids", nargs="*", help="project_id directory name(s)")
    parser.add_argument("--auto", action="store_true", help="auto-discover active project dirs")
    parser.add_argument("--refresh", type=float, default=1.0, help="refresh interval seconds")
    args = parser.parse_args()

    if args.auto:
        projects = _resolve_auto()
    else:
        projects = [EVENTS_DIR / pid for pid in args.project_ids]

    if not projects:
        print("No projects found. Pass --auto or specify project_id(s).", file=sys.stderr)
        return 2

    print(f"Monitoring {len(projects)} run(s): {', '.join(p.name for p in projects)}")
    time.sleep(0.5)

    fhs: dict[str, Any] = {}
    event_buf: dict[str, list[str]] = {p.name: [] for p in projects}
    try:
        for p in projects:
            events_file = p / "dashboard_events.jsonl"
            try:
                fh = events_file.open()
                fh.seek(0, 2)  # seek to end — show only NEW events
                fhs[p.name] = fh
            except OSError:
                pass

        sys.stdout.write(HIDE_CURSOR)
        while True:
            for p in projects:
                fh = fhs.get(p.name)
                if fh is None:
                    # File may have been created since we started
                    events_file = p / "dashboard_events.jsonl"
                    if events_file.exists():
                        fh = events_file.open()
                        fhs[p.name] = fh
                if fh is not None:
                    _read_new_lines(fh, event_buf[p.name])
            sys.stdout.write(CLEAR + _render(projects, event_buf))
            sys.stdout.flush()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        return 0
    finally:
        for fh in fhs.values():
            try: fh.close()
            except: pass
        sys.stdout.write(SHOW_CURSOR + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
