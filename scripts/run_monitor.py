#!/usr/bin/env python3
"""Dev-facing TUI for ReproLab runs (v3).

Information-dense single-screen monitor for every signal a dev needs
while debugging a paper-reproduction run.  Pure stdlib (no curses /
textual / asyncio).

Panes (top to bottom):
  Header        local time, run + pod counts, total burn rate, controls
  RunPod        every pod: id, GPU class, status, $/hr, uptime, cost-so-far,
                owner project (parsed from name).
  Per-paper     for each active run:
                  - project + pid + etime + current phase + iteration
                  - watchdog signal ages
                  - rubric score with trend arrow (vs previous iteration)
                  - per-area progress bars
                  - top weak leaves with justifications
                  - latest failure: failure_class + suggested_fix
                  - latest run_warning (sticky)
                  - if completed: final_report summary
  Original      rubric tree from generated_rubric.json (areas + leaves +
   rubric       weights)
  Events        meaningful event stream (skips heartbeat noise)
  exec.log      last 6 lines from each active pod (Lane B streamed)
  Lab UI        click-through URLs

Usage:
  python scripts/run_monitor.py --auto                      active runs <5min old
  python scripts/run_monitor.py prj_xxx [prj_yyy ...]       explicit
  python scripts/run_monitor.py --auto --no-pods            skip RunPod API polls
  python scripts/run_monitor.py --auto --refresh 2          slower refresh
  python scripts/run_monitor.py --auto --no-rubric-tree     hide original rubric
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── ANSI ─────────────────────────────────────────────────────────────────
ESC = "\x1b["
CLEAR = ESC + "2J" + ESC + "H"
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
    if s < 3600: return f"{int(s//60)}m{int(s%60)}s"
    return f"{int(s//3600)}h{int((s%3600)//60)}m"


def _term_width(default: int = 140) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _bar(score: float, width: int = 14) -> str:
    score = max(0.0, min(1.0, float(score)))
    fill = int(score * width)
    return "█" * fill + "░" * (width - fill)


def _score_color(score: float, target: float = 0.6) -> str:
    if score >= target: return GRN
    if score >= target * 0.5: return YEL
    return RED


def _trend_arrow(curr: float, prev: float | None) -> str:
    if prev is None: return ""
    diff = curr - prev
    if abs(diff) < 0.001: return f"{DIM}↔{R}"
    if diff > 0: return f"{GRN}↑{diff:+.3f}{R}"
    return f"{RED}↓{diff:+.3f}{R}"


# ── per-run state extraction ─────────────────────────────────────────────


def _extract_run_state(events_file: Path) -> dict:
    """Walk dashboard_events.jsonl once and pull every signal we display.

    Returns a dict with: n_events, latest_phase, iteration, rubric_history
    (list of (iteration, score, target, areas, weak_leaves)), last_warn,
    last_failure (failure_class + suggested_fix), gpu_short_name,
    candidate_count, warn_count, run_complete (when present).
    """
    out: dict = {
        "n_events": 0,
        "latest_phase": None,
        "iteration": None,
        "rubric_history": [],
        "last_warn": None,
        "last_failure": None,
        "last_failure_class": None,
        "last_failure_fix": None,
        "experiment_status": None,
        "gpu_short_name": None,
        "gpu_escalations": [],
        "candidate_count": 0,
        "warn_count": 0,
        "run_complete": None,
    }
    try:
        with events_file.open() as f:
            for line in f:
                out["n_events"] += 1
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = d.get("data") or {}
                ev = d.get("event") or d.get("type") or ""
                primitive = d.get("primitive") or ""
                status = d.get("status") or ""
                if ev == "primitive_call" and primitive and primitive != "heartbeat":
                    out["latest_phase"] = f"{primitive}({status})"
                elif ev == "iteration_heartbeat":
                    try: out["iteration"] = int(d.get("iteration") or d.get("counter") or out["iteration"] or 0)
                    except: pass
                elif ev == "rubric_score":
                    out["rubric_history"].append({
                        "iteration": d.get("iteration"),
                        "score": float(d.get("score") or 0),
                        "target": float(d.get("target") or 0.6),
                        "areas": d.get("areas") or [],
                        "weak_leaves": d.get("weak_leaves") or [],
                    })
                elif ev == "run_warning":
                    out["warn_count"] += 1
                    out["last_warn"] = data
                elif ev == "experiment_completed":
                    out["experiment_status"] = "ok" if data.get("success") else "fail"
                    if not data.get("success"):
                        out["last_failure"] = (data.get("error") or "")[:240]
                        out["last_failure_class"] = data.get("failure_class")
                        out["last_failure_fix"] = data.get("suggested_fix")
                elif ev == "gpu_resolved":
                    out["gpu_short_name"] = data.get("short_name")
                elif ev == "gpu_escalated":
                    out["gpu_escalations"].append(
                        f"{data.get('from_sku')}→{data.get('to_sku')} ({data.get('reason')})"
                    )
                elif ev == "candidate_proposed":
                    out["candidate_count"] += 1
                elif ev == "run_complete":
                    out["run_complete"] = d
    except OSError:
        pass
    return out


def _latest_run_dir(project: Path) -> Path | None:
    outputs = project / "code" / "outputs"
    if not outputs.is_dir(): return None
    subs = [p for p in outputs.iterdir() if p.is_dir()]
    return max(subs, key=lambda p: p.stat().st_mtime) if subs else None


def _tail_file(path: Path, n: int = 6) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except (OSError, FileNotFoundError):
        return []


def _cache_count(project: Path) -> int:
    cache = project / "rlm_state" / "primitive_cache.jsonl"
    try:
        with cache.open() as f:
            return sum(1 for _ in f)
    except (OSError, FileNotFoundError):
        return 0


def _watchdog_signals(project: Path) -> dict:
    out: dict[str, float | None] = {}
    rd = _latest_run_dir(project)
    if rd:
        out["exec_log"] = _age_seconds(rd / "exec.log")
        out["heartbeat"] = _age_seconds(rd / ".heartbeat")
    else:
        out["exec_log"] = out["heartbeat"] = None
    out["sse_event"] = _age_seconds(project / "dashboard_events.jsonl")
    return out


def _final_report(project: Path) -> dict | None:
    try:
        return json.loads((project / "final_report.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _rubric_tree(project: Path) -> dict | None:
    """Try generated_rubric.json (LLM-derived) then fall back to nothing."""
    for candidate in [project / "generated_rubric.json",
                      project / "rubric.json"]:
        try:
            return json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _walk_rubric_leaves(node: dict, depth: int = 0) -> list[tuple[int, str, float]]:
    """Return [(depth, title, weight)] for every leaf in the rubric tree."""
    if not isinstance(node, dict):
        return []
    out: list[tuple[int, str, float]] = []
    title = (node.get("title") or node.get("area") or node.get("name") or "?")[:64]
    weight = float(node.get("weight") or 0)
    children = node.get("sub_tasks") or node.get("children") or []
    if not children:
        # leaf
        out.append((depth, title, weight))
    else:
        if depth == 0:
            # root — list its sub_tasks
            for child in children:
                if isinstance(child, dict):
                    out.extend(_walk_rubric_leaves(child, depth=0))
        else:
            out.append((depth, title, weight))
            for child in children:
                if isinstance(child, dict):
                    out.extend(_walk_rubric_leaves(child, depth + 1))
    return out


# ── RunPod ───────────────────────────────────────────────────────────────


_POD_CACHE: dict = {"ts": 0.0, "pods": []}
_POD_FIRST_SEEN: dict[str, float] = {}


def _api_key_from_env() -> str:
    k = os.environ.get("REPROLAB_RUNPOD_API_KEY", "") or os.environ.get("RUNPOD_API_KEY", "")
    if k: return k
    try:
        for line in Path(".env").read_text().splitlines():
            if line.startswith("REPROLAB_RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip()
    except (OSError, FileNotFoundError):
        pass
    return ""


def _runpod_pods(api_key: str, ttl: float = 15.0) -> list[dict]:
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
    # Track first-seen for cost-so-far estimate when RunPod's createdAt is missing
    seen_ids = {p.get("id") for p in pods}
    for pid in seen_ids:
        _POD_FIRST_SEEN.setdefault(pid, _now())
    return pods


def _pod_owner_label(pod_name: str, projects: list[Path]) -> str:
    for p in projects:
        if p.name in (pod_name or ""):
            return p.name
    if "ui-rlm" in (pod_name or ""):
        return "(UI side pod)"
    return "?"


def _pod_uptime_seconds(pod: dict) -> float | None:
    """Try createdAt, fall back to first-seen-in-this-session."""
    created = pod.get("createdAt") or ""
    if created:
        raw = created.strip()
        raw = re.sub(r"\s+[A-Z]{2,5}$", "", raw)
        raw = re.sub(r"\s+([+-])(\d{2})(\d{2})$", r"\1\2:\3", raw)
        if raw.endswith("Z"): raw = raw[:-1] + "+00:00"
        if not re.search(r"[+\-]\d{2}:\d{2}$", raw): raw = raw + "+00:00"
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(raw)
            return max(0.0, _now() - dt.timestamp())
        except (ValueError, AttributeError):
            pass
    seen = _POD_FIRST_SEEN.get(pod.get("id") or "")
    if seen is not None:
        return _now() - seen
    return None


def _pod_gpu_label(pod: dict) -> str:
    cost_raw = pod.get("costPerHr")
    try:
        cost = float(cost_raw) if cost_raw else 0
    except (TypeError, ValueError):
        cost = 0
    # Map by SECURE price → GPU class.  Best-effort heuristic since RunPod's
    # list endpoint doesn't surface gpuTypeId.
    if cost < 0.20: return "CPU-only"
    if cost < 0.40: return "RTX 4090(community)"
    if cost <= 0.75: return "RTX 4090(secure)"
    if cost <= 0.90: return "L40S"
    if cost <= 1.50: return "A100 40GB"
    if cost <= 2.10: return "A100 80GB"
    return f"≈${cost:.2f}/hr GPU"


# ── rendering ────────────────────────────────────────────────────────────


def _render_pod_table(pods: list[dict], projects: list[Path]) -> list[str]:
    out: list[str] = []
    if not pods:
        out.append(f"  {DIM}(no pods){R}")
        return out
    out.append(f"  {DIM}{'id':<14}  {'status':<10}  {'gpu':<22}  {'$/hr':>6}  {'uptime':>8}  {'cost':>7}  owner{R}")
    total = 0.0
    for p in pods:
        cost_raw = p.get("costPerHr")
        try: cost = float(cost_raw) if cost_raw else 0.0
        except (TypeError, ValueError): cost = 0.0
        total += cost
        up = _pod_uptime_seconds(p) or 0.0
        spend = cost * up / 3600.0
        gpu = _pod_gpu_label(p)
        owner = _pod_owner_label(p.get("name", ""), projects)
        out.append(
            f"  {p.get('id','?')[:14]:<14}  "
            f"{p.get('desiredStatus','?'):<10}  "
            f"{gpu:<22}  "
            f"{cost:>6.2f}  "
            f"{_humansec(up):>8}  "
            f"${spend:>6.2f}  "
            f"{owner}"
        )
    out.append(f"  {B}burn rate: ${total:.2f}/hr{R}")
    return out


def _render_run_card(
    project: Path, state: dict, sigs: dict, pods: list[dict],
) -> list[str]:
    out: list[str] = []
    n_events = state["n_events"]
    phase = state["latest_phase"] or "—"
    pods_here = [p for p in pods if project.name in (p.get("name") or "")]
    pod_str = ""
    if pods_here:
        p = pods_here[0]
        pod_str = (
            f"  {DIM}pod:{R} {p.get('id','?')[:10]}({p.get('desiredStatus','?')})"
            f"  {_pod_gpu_label(p)}  ${p.get('costPerHr','?')}/hr"
        )

    iter_str = f"  iter={state['iteration']}" if state.get("iteration") else ""
    out.append(
        f"  {B}{project.name}{R}  pid?  "
        f"phase={CYN}{phase}{R}  events={n_events}{iter_str}{pod_str}"
    )
    out.append(
        f"    {DIM}signals:{R} "
        f"exec_log={_color_age(sigs['exec_log'])}  "
        f"heartbeat={_color_age(sigs['heartbeat'])}  "
        f"sse_event={_color_age(sigs['sse_event'])}  "
        f"cache={_cache_count(project)}"
    )

    # GPU escalation history (compact)
    if state.get("gpu_escalations"):
        latest = state["gpu_escalations"][-3:]
        out.append(f"    {DIM}gpu_escalations:{R} " + "  →  ".join(latest))

    # Rubric history with trend
    hist = state.get("rubric_history") or []
    if hist:
        latest = hist[-1]
        prev_score = hist[-2]["score"] if len(hist) > 1 else None
        score = latest["score"]; target = latest["target"]
        color = _score_color(score, target)
        trend = _trend_arrow(score, prev_score)
        out.append(
            f"    {B}rubric:{R} {color}{score:.3f}{R}/{target:.2f}  "
            f"[{color}{_bar(score)}{R}]  iter#{latest.get('iteration', len(hist))}  "
            f"{trend}"
        )
        # Per-area progress
        for area in latest.get("areas", [])[:6]:
            name = (area.get("area") or "?")[:42]
            ascore = float(area.get("score") or 0)
            weight = float(area.get("weight") or 0)
            astatus = area.get("status") or "?"
            acolor = _score_color(ascore, 0.7)
            out.append(
                f"      {DIM}{name:<42}{R}  {acolor}{_bar(ascore, 14)}{R}  "
                f"{ascore:.2f}  w={weight:.2f}  {astatus}"
            )
        # Top weak leaves with justification
        weak = latest.get("weak_leaves") or []
        if weak:
            out.append(f"      {DIM}top weak leaves:{R}")
            for leaf in weak[:5]:
                lid = (leaf.get("id") or "?")[:32]
                lscore = float(leaf.get("score") or 0)
                just = (leaf.get("justification") or "")[:90]
                out.append(f"        [{_score_color(lscore, 0.7)}{lscore:.2f}{R}] {lid:<32}  {DIM}{just}{R}")

    # Latest failure with class + suggested fix (NEW)
    if state.get("last_failure"):
        out.append("")
        klass = state.get("last_failure_class") or "unknown"
        fix = state.get("last_failure_fix") or ""
        out.append(f"    {RED}✗ last_failure:{R} {DIM}{state['last_failure']}{R}")
        out.append(f"      {YEL}failure_class:{R} {klass}")
        if fix:
            out.append(f"      {YEL}suggested_fix:{R} {DIM}{fix[:160]}{R}")

    # Latest run_warning (sticky)
    if state.get("last_warn"):
        w = state["last_warn"]
        out.append(
            f"    {RED}⚠ last_warn:{R} {w.get('reason','?')}  "
            f"stale={(w.get('stale_seconds') or 0):.0f}s  "
            f"freshest={w.get('freshest_signal','?')}"
        )

    # Final report (when run_complete fired)
    rc = state.get("run_complete")
    if rc:
        final = _final_report(project) or {}
        out.append("")
        out.append(f"    {B}{MAG}final_report:{R}  status={rc.get('status')}")
        if final.get("rubric_score") is not None:
            out.append(f"      rubric_score={final.get('rubric_score'):.3f}  cost_usd={final.get('cost_usd', 0)}")
        if final.get("mode"):
            out.append(f"      mode={final.get('mode')}  models={final.get('models')}")
        if final.get("verdict"):
            out.append(f"      verdict={(final.get('verdict') or '')[:200]}")

    return out


def _render_rubric_tree(project: Path) -> list[str]:
    tree = _rubric_tree(project)
    if not tree:
        return []
    out: list[str] = []
    out.append(f"  {B}{project.name}{R} target={tree.get('target_score', 0.6)}")
    # Walk top-level areas
    areas = tree.get("sub_tasks") or tree.get("areas") or []
    for area in areas[:6]:
        if not isinstance(area, dict):
            continue
        title = (area.get("title") or area.get("area") or "?")[:60]
        weight = float(area.get("weight") or 0)
        out.append(f"    {CYN}{title:<60}{R}  w={weight:.3f}")
        children = area.get("sub_tasks") or area.get("children") or []
        for leaf in children[:5]:
            if not isinstance(leaf, dict):
                continue
            ltitle = (leaf.get("title") or leaf.get("area") or "?")[:74]
            lweight = float(leaf.get("weight") or 0)
            out.append(f"      {DIM}- {ltitle:<74}  w={lweight:.3f}{R}")
        if len(children) > 5:
            out.append(f"      {DIM}... +{len(children)-5} more leaves{R}")
    return out


def _summarise_event(line: str) -> str | None:
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    ev = d.get("event") or ""
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
        parts.append(f"{primitive}({d.get('status','')})")
    if ev == "rubric_score":
        parts.append(f"score={float(d.get('score',0)):.3f}/{float(d.get('target',0)):.2f}")
    if ev == "experiment_completed":
        s = data.get("success"); err = (data.get("error") or "")[:60]
        klass = data.get("failure_class") or ""
        klass_part = f" [{klass}]" if klass else ""
        parts.append(f"success={s}{klass_part}{(' err='+err) if err else ''}")
    if ev == "gpu_escalated":
        parts.append(f"{data.get('from_sku')}→{data.get('to_sku')} ({data.get('reason')})")
    if ev == "run_warning":
        parts.append(f"{data.get('reason')} stale={(data.get('stale_seconds') or 0):.0f}s")
    if ev == "run_complete":
        parts.append(f"status={d.get('status')}")
    return "  ".join(parts)


def _render(
    projects: list[Path],
    state: dict[str, dict],
    event_buf: dict[str, deque],
    pods: list[dict],
    show_pods: bool,
    show_rubric_tree: bool,
) -> str:
    out: list[str] = []

    # Header
    burn = sum(float(p.get("costPerHr") or 0) for p in pods)
    out.append(
        f"{BG_GRY}{B} ReproLab Dev Monitor "
        f"  {datetime.now().strftime('%H:%M:%S')}  "
        f"runs={len(projects)}  pods={len(pods)}  burn=${burn:.2f}/hr  Ctrl-C quits {R}"
    )

    # Pods
    if show_pods:
        out.append("")
        out.append(f"{B}━━ RunPod ━━{R}")
        out.extend(_render_pod_table(pods, projects))

    # Per-paper cards
    for project in projects:
        out.append("")
        out.append(f"{B}━━ {project.name} ━━{R}")
        sigs = _watchdog_signals(project)
        out.extend(_render_run_card(project, state.get(project.name, {}), sigs, pods))

    # Original rubric
    if show_rubric_tree:
        any_tree = False
        for project in projects:
            tree_lines = _render_rubric_tree(project)
            if tree_lines:
                if not any_tree:
                    out.append("")
                    out.append(f"{B}━━ Original rubric ━━{R}")
                    any_tree = True
                out.extend(tree_lines)

    # Recent events
    out.append("")
    out.append(f"{B}━━ Recent events ━━{R}")
    rows: list[tuple[str, str]] = []
    for project in projects:
        for ev in list(event_buf.get(project.name, []))[-12:]:
            s = _summarise_event(ev)
            if s:
                rows.append((project.name[:18], s))
    for tag, ev in rows[-12:]:
        out.append(f"  [{tag:<18}] {ev}")
    if not rows:
        out.append(f"  {DIM}(no meaningful events yet){R}")

    # exec.log tails
    for project in projects:
        rd = _latest_run_dir(project)
        if rd is None: continue
        log = rd / "exec.log"
        if not log.exists(): continue
        tail = _tail_file(log, n=5)
        if not tail: continue
        out.append("")
        out.append(f"{B}━━ exec.log ({project.name}/{rd.name[:10]}) ━━{R}")
        for ln in tail:
            ln = ln.rstrip()
            if not ln: continue
            out.append(f"  {DIM}{ln[:170]}{R}")

    # UI links
    out.append("")
    out.append(f"{B}━━ Lab UI ━━{R}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_ids", nargs="*")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--no-pods", action="store_true")
    parser.add_argument("--no-rubric-tree", action="store_true")
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
                fh = f.open(); fh.seek(0, 2); fhs[p.name] = fh
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
                        fh = f.open(); fhs[p.name] = fh
                if fh is not None:
                    while True:
                        line = fh.readline()
                        if not line: break
                        event_buf[p.name].append(line.rstrip())
                state[p.name] = _extract_run_state(p / "dashboard_events.jsonl")
            pods = _runpod_pods(api_key) if show_pods else []
            sys.stdout.write(CLEAR + _render(projects, state, event_buf, pods, show_pods, not args.no_rubric_tree))
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
