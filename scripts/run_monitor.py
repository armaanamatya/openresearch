#!/usr/bin/env python3
"""Dev-facing TUI for ReproLab runs.

One terminal, every signal a dev needs to keep an eye on while a run is
in flight: per-run health, pod state + cost, rubric trajectory, watchdog
warnings, primitive-cache hit rate, recent meaningful events, exec.log
tail.  Pure stdlib — no curses / textual / asyncio.

Usage:
    python scripts/run_monitor.py --auto                     # active runs <5min old
    python scripts/run_monitor.py prj_xxx [prj_yyy ...]      # explicit
    python scripts/run_monitor.py --auto --no-pods           # skip RunPod API polls
    python scripts/run_monitor.py --auto --refresh 2         # 2s refresh

Quit with Ctrl-C.  Best viewed in a 120-column-wide terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── ANSI ─────────────────────────────────────────────────────────────────
ESC = "\x1b["
CLEAR = ESC + "2J" + ESC + "H"
HOME = ESC + "H"
HIDE_CURSOR = ESC + "?25l"
SHOW_CURSOR = ESC + "?25h"
B = ESC + "1m"
DIM = ESC + "2m"
R = ESC + "0m"
RED = ESC + "31m"
GRN = ESC + "32m"
YEL = ESC + "33m"
BLU = ESC + "34m"
MAG = ESC + "35m"
CYN = ESC + "36m"
GRY = ESC + "90m"
BG_GRY = ESC + "100m"

EVENTS_DIR = Path("runs")

# Events worth surfacing
MEANINGFUL = {
    "primitive_call", "experiment_completed", "rubric_score", "run_complete",
    "run_failed", "run_warning", "gpu_resolved", "gpu_escalated",
    "candidate_proposed", "candidate_outcome", "sub_rlm_spawned",
    "sub_rlm_complete", "worker_report_started", "worker_report_completed",
}
EVENT_COLOR = {
    "run_warning": RED, "run_failed": RED, "run_complete": MAG,
    "rubric_score": GRN, "experiment_completed": CYN,
    "gpu_resolved": BLU, "gpu_escalated": YEL,
    "candidate_proposed": BLU, "candidate_outcome": BLU,
    "sub_rlm_spawned": GRY, "sub_rlm_complete": GRY,
    "primitive_call": GRY, "worker_report_started": DIM + CYN,
    "worker_report_completed": DIM + CYN,
}

# ── helpers ──────────────────────────────────────────────────────────────


def _now() -> float: return time.time()


def _short_ts(ts: Any) -> str:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%H:%M:%S")
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
        except ValueError:
            return "??:??:??"
    return "??:??:??"


def _age_seconds(path: Path) -> float | None:
    try:
        return _now() - path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return None


def _color_age(seconds: float | None) -> str:
    if seconds is None: return f"{DIM}—{R}"
    if seconds < 60: return f"{GRN}{seconds:.0f}s{R}"
    if seconds < 300: return f"{YEL}{seconds:.0f}s{R}"
    return f"{RED}{seconds:.0f}s{R}"


def _humansec(s: float) -> str:
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"


def _term_width(default: int = 120) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


# ── state extraction ─────────────────────────────────────────────────────


def _ps_for_project(project_id: str) -> dict | None:
    """Find the python orchestrator pid for this project_id via ps + reading args."""
    try:
        r = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,pmem,args", "--no-headers"],
            capture_output=True, text=True, timeout=2,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if r.returncode != 0:
        return None
    # We look for an active python invocation whose project_id will be
    # written to the runs/<id>/demo_status.json once it's running.
    # Simpler: pid matches if it wrote to this project's events_file recently.
    return None  # not the primary signal; we use events_file mtime instead


def _read_event_summaries(events_file: Path, buf: deque, fh, max_buf: int) -> dict:
    """Read NEW lines from events_file into buf. Return summary stats."""
    stats = {
        "n_events": 0, "latest_phase": None, "iteration": None,
        "rubric_score": None, "rubric_target": None, "rubric_areas": [],
        "last_warn": None, "last_failure": None, "pod_id": None,
        "gpu_short_name": None, "gpu_cost_per_hr": None,
        "experiment_status": None,
        "candidate_count": 0, "warn_count": 0,
    }
    try:
        with events_file.open() as f0:
            stats["n_events"] = sum(1 for _ in f0)
    except OSError:
        pass

    # Append any new lines via the persistent fh handle.
    if fh is not None:
        while True:
            line = fh.readline()
            if not line:
                break
            buf.append(line.rstrip())

    # Walk the entire file once per refresh for derived state.
    try:
        with events_file.open() as f0:
            for line in f0:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = d.get("event") or d.get("type") or ""
                data = d.get("data") or {}
                primitive = d.get("primitive") or ""
                if ev == "primitive_call":
                    if primitive and primitive != "heartbeat":
                        stats["latest_phase"] = primitive
                elif ev == "rubric_score":
                    score = d.get("score")
                    if score is not None:
                        stats["rubric_score"] = float(score)
                        stats["rubric_target"] = float(d.get("target") or 0)
                        stats["rubric_areas"] = d.get("areas") or []
                elif ev == "run_warning":
                    stats["warn_count"] += 1
                    stats["last_warn"] = data
                elif ev == "experiment_completed":
                    success = data.get("success")
                    stats["experiment_status"] = ("ok" if success else "fail")
                    if not success and (data.get("error") or "").strip():
                        stats["last_failure"] = data.get("error", "")[:120]
                elif ev == "gpu_resolved":
                    stats["gpu_short_name"] = data.get("short_name")
                elif ev == "candidate_proposed":
                    stats["candidate_count"] += 1
                elif ev == "iteration_heartbeat":
                    n = d.get("iteration") or d.get("counter")
                    if n is not None:
                        try: stats["iteration"] = int(n)
                        except: pass
    except OSError:
        pass
    return stats


def _summarise_line(line: str) -> str | None:
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    ev = d.get("event") or d.get("type") or ""
    primitive = d.get("primitive") or ""
    if ev == "iteration_heartbeat":
        return None
    if ev == "primitive_call" and primitive == "heartbeat":
        return None
    if ev not in MEANINGFUL:
        return None
    ts = d.get("ts") or d.get("timestamp") or ""
    color = EVENT_COLOR.get(ev, "")
    data = d.get("data") or {}
    parts = [f"{_short_ts(ts):>8} {color}{ev:<22}{R}"]
    if primitive:
        status = d.get("status") or ""
        parts.append(f"{primitive}({status})")
    if ev == "rubric_score":
        parts.append(f"score={float(d.get('score', 0)):.3f}/{float(d.get('target', 0)):.2f}")
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


def _latest_run_dir(project: Path) -> Path | None:
    outputs = project / "code" / "outputs"
    if not outputs.is_dir(): return None
    subdirs = [p for p in outputs.iterdir() if p.is_dir()]
    if not subdirs: return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _tail_file(path: Path, n: int = 6) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return fh.readlines()[-n:]
    except (OSError, FileNotFoundError):
        return []


def _read_metrics(project: Path) -> dict | None:
    """Try to find a metrics.json — first in pod artifact dir, then code/."""
    for candidate in [
        *[(rd / "metrics.json") for rd in [_latest_run_dir(project)] if rd],
        project / "code" / "metrics.json",
    ]:
        try:
            return json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    return None


def _cache_stats(project: Path) -> dict[str, int]:
    cache = project / "rlm_state" / "primitive_cache.jsonl"
    counts: dict[str, int] = {}
    try:
        with cache.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = d.get("primitive", "?")
                counts[p] = counts.get(p, 0) + 1
    except (OSError, FileNotFoundError):
        return {}
    return counts


def _watchdog_signals(project: Path) -> dict:
    """Read the three watchdog signals."""
    out: dict[str, float | None] = {}
    run_dir = _latest_run_dir(project)
    if run_dir:
        out["exec_log"] = _age_seconds(run_dir / "exec.log")
        out["heartbeat"] = _age_seconds(run_dir / ".heartbeat")
    else:
        out["exec_log"] = out["heartbeat"] = None
    out["sse_event"] = _age_seconds(project / "dashboard_events.jsonl")
    return out


# ── RunPod pods ──────────────────────────────────────────────────────────


_POD_CACHE: dict = {"ts": 0.0, "pods": []}


def _runpod_pods(api_key: str, ttl: float = 15.0) -> list[dict]:
    """Cached GET /v1/pods so we don't hammer the RunPod API on every refresh."""
    if _now() - _POD_CACHE["ts"] < ttl:
        return _POD_CACHE["pods"]
    if not api_key:
        return []
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://rest.runpod.io/v1/pods",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
    except Exception:
        return _POD_CACHE.get("pods", [])
    pods = data if isinstance(data, list) else data.get("pods", []) or []
    _POD_CACHE["ts"] = _now()
    _POD_CACHE["pods"] = pods
    return pods


def _pods_for_project(pods: list[dict], project_id: str) -> list[dict]:
    return [p for p in pods if project_id in (p.get("name") or "")]


# ── rendering ────────────────────────────────────────────────────────────


def _render(
    projects: list[Path],
    state: dict[str, dict],
    event_buf: dict[str, deque],
    pod_data: list[dict],
    show_pods: bool,
) -> str:
    width = _term_width()
    out: list[str] = []

    # ── Top bar ────────────────────────────────────────────────────────
    out.append(
        f"{BG_GRY}{B} ReproLab Dev Monitor "
        f"  {datetime.now().strftime('%H:%M:%S')}  "
        f"runs={len(projects)}  pods={len(pod_data)}  Ctrl-C quits {R}"
    )
    out.append("")

    # ── Per-run summary ────────────────────────────────────────────────
    out.append(f"{B}━━ Runs ━━{R}")
    for project in projects:
        s = state.get(project.name, {})
        sigs = _watchdog_signals(project)
        rubric = s.get("rubric_score")
        target = s.get("rubric_target") or 0.6
        phase = s.get("latest_phase") or "—"
        wcount = s.get("warn_count", 0)
        cache_n = sum(_cache_stats(project).values())

        # status mark
        ev_age = sigs["sse_event"]
        if ev_age is None: mark = f"{DIM}∅{R}"
        elif ev_age < 60: mark = f"{GRN}●{R}"
        elif ev_age < 600: mark = f"{YEL}●{R}"
        else: mark = f"{RED}●{R}"

        rubric_str = "—"
        if rubric is not None:
            color = GRN if rubric >= target else (YEL if rubric >= target * 0.5 else RED)
            rubric_str = f"{color}{rubric:.3f}{R}/{target:.2f}"

        # pod
        pods_here = _pods_for_project(pod_data, project.name)
        pod_str = ""
        if pods_here:
            p = pods_here[0]
            pod_str = f"  pod={p.get('id','?')[:10]}({p.get('desiredStatus','?')})"

        line = (
            f"  {mark} {B}{project.name[:30]:<30}{R}  "
            f"phase={CYN}{phase:<22}{R}  "
            f"events={s.get('n_events', 0):>3}  "
            f"rubric={rubric_str:>20}  "
            f"warn={wcount}  "
            f"cache={cache_n}  "
            f"last_ev={_color_age(ev_age)}"
            f"{pod_str}"
        )
        out.append(line)

        # Watchdog signal row
        out.append(
            f"      {DIM}signals:{R} "
            f"exec_log={_color_age(sigs['exec_log'])}  "
            f"heartbeat={_color_age(sigs['heartbeat'])}  "
            f"sse_event={_color_age(sigs['sse_event'])}"
        )

        # Rubric area breakdown (if available)
        if s.get("rubric_areas"):
            for area in s["rubric_areas"][:6]:
                name = area.get("area", "?")[:42]
                ascore = float(area.get("score", 0))
                weight = float(area.get("weight", 0))
                status = area.get("status", "?")
                bar_width = 12
                fill = int(ascore * bar_width)
                bar = "█" * fill + "░" * (bar_width - fill)
                color = GRN if ascore >= 0.7 else (YEL if ascore >= 0.4 else RED)
                out.append(f"      {DIM}{name:<42}{R}  {color}{bar}{R}  {ascore:.2f}  w={weight:.2f}  {status}")

        # Last warning (sticky)
        if s.get("last_warn"):
            w = s["last_warn"]
            reason = w.get("reason", "?")
            stale = w.get("stale_seconds") or 0
            freshest = w.get("freshest_signal", "?")
            out.append(
                f"      {RED}⚠  last_warn:{R} {reason}  stale={stale:.0f}s  freshest_signal={freshest}"
            )

        # Last failure (sticky)
        if s.get("last_failure"):
            out.append(f"      {RED}✗  last_failure:{R} {s['last_failure']}")

        out.append("")

    # ── Pods table ─────────────────────────────────────────────────────
    if show_pods and pod_data:
        out.append(f"{B}━━ RunPod ({len(pod_data)} pod) ━━{R}")
        out.append(f"  {DIM}{'id':<14}  {'status':<10}  {'gpu':<26}  $/hr   name{R}")
        total = 0.0
        for p in pod_data:
            cost = p.get("costPerHr")
            try: total += float(cost) if cost else 0.0
            except (TypeError, ValueError): pass
            gpu_t = p.get("gpuTypeIds") or [p.get("machine", {}).get("gpuTypeId", "?")]
            gpu_str = (gpu_t[0] if isinstance(gpu_t, list) else gpu_t)[:26]
            name = (p.get("name") or "?")[:60]
            out.append(
                f"  {p.get('id','?')[:14]:<14}  "
                f"{p.get('desiredStatus','?'):<10}  "
                f"{gpu_str:<26}  "
                f"{cost or '—'}    {name}"
            )
        out.append(f"  {B}total burn rate: ${total:.2f}/hr{R}")
        out.append("")

    # ── Recent meaningful events ───────────────────────────────────────
    out.append(f"{B}━━ Recent events ━━{R}")
    rows: list[tuple[str, str]] = []
    for project in projects:
        for ev in list(event_buf.get(project.name, []))[-15:]:
            s = _summarise_line(ev)
            if s:
                rows.append((project.name[:18], s))
    for tag, ev in rows[-15:]:
        out.append(f"  [{tag:<18}] {ev}")
    if not rows:
        out.append(f"  {DIM}(no meaningful events yet){R}")
    out.append("")

    # ── exec.log tails ─────────────────────────────────────────────────
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
        out.append(f"{B}━━ exec.log ({project.name}/{run_dir.name[:10]}) ━━{R}")
        for ln in tail:
            ln = ln.rstrip()
            if not ln:
                continue
            out.append(f"  {DIM}{ln[:width - 4]}{R}")
        out.append("")

    # ── Lab UI URLs ────────────────────────────────────────────────────
    out.append(f"{B}━━ Lab UI links ━━{R}")
    for project in projects:
        out.append(f"  {DIM}http://localhost:3000/lab?projectId={project.name}{R}")

    return "\n".join(out)


def _resolve_auto() -> list[Path]:
    if not EVENTS_DIR.is_dir(): return []
    cs: list[tuple[float, Path]] = []
    for d in EVENTS_DIR.iterdir():
        if not d.is_dir(): continue
        f = d / "dashboard_events.jsonl"
        if not f.exists(): continue
        a = _age_seconds(f)
        if a is None or a > 300: continue
        cs.append((a, d))
    cs.sort()
    return [d for _, d in cs]


def _api_key_from_env() -> str:
    # Inline .env-aware read because users may not have dotenv loaded
    from_env = os.environ.get("REPROLAB_RUNPOD_API_KEY", "") or os.environ.get("RUNPOD_API_KEY", "")
    if from_env: return from_env
    try:
        for line in Path(".env").read_text().splitlines():
            if line.startswith("REPROLAB_RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip()
    except (OSError, FileNotFoundError):
        pass
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_ids", nargs="*")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--no-pods", action="store_true", help="skip RunPod API polls")
    parser.add_argument("--refresh", type=float, default=1.0)
    parser.add_argument("--buf-size", type=int, default=200)
    args = parser.parse_args()

    projects = _resolve_auto() if args.auto else [EVENTS_DIR / pid for pid in args.project_ids]
    if not projects:
        print("No projects found. Use --auto or pass project_id(s).", file=sys.stderr)
        return 2

    print(f"Monitoring {len(projects)} run(s): {', '.join(p.name for p in projects)}")
    time.sleep(0.3)

    api_key = "" if args.no_pods else _api_key_from_env()
    show_pods = bool(api_key)
    fhs: dict[str, Any] = {}
    event_buf: dict[str, deque] = {p.name: deque(maxlen=args.buf_size) for p in projects}

    try:
        for p in projects:
            f = p / "dashboard_events.jsonl"
            try:
                fh = f.open()
                fh.seek(0, 2)
                fhs[p.name] = fh
            except OSError:
                pass

        sys.stdout.write(HIDE_CURSOR)
        while True:
            state: dict[str, dict] = {}
            for p in projects:
                fh = fhs.get(p.name)
                if fh is None:
                    f = p / "dashboard_events.jsonl"
                    if f.exists():
                        fh = f.open()
                        fhs[p.name] = fh
                state[p.name] = _read_event_summaries(p / "dashboard_events.jsonl",
                                                     event_buf[p.name], fh,
                                                     args.buf_size)
            pod_data = _runpod_pods(api_key) if show_pods else []
            sys.stdout.write(CLEAR + _render(projects, state, event_buf, pod_data, show_pods))
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
