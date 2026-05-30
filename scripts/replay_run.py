#!/usr/bin/env python3
"""Read-only run-replay / postmortem harness (Phase 7g / Gap-3).

Ingests the per-run artifacts under ``runs/<project_id>/`` and prints a verdict:
the longest gap between dashboard events, dangling ``sub_rlm_spawned`` (the wedge
shape), the prompt-cache read ratio, and whether the final report claims a
success-ish verdict with no experiment evidence (FM-004).

No LLM, no network, no new persistence — pure analysis of existing JSONL/JSON.

Usage:
    .venv/bin/python scripts/replay_run.py runs/<project_id>
    .venv/bin/python scripts/replay_run.py <project_id>            # resolves under runs/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Surface a wedge: an open sub-call or an event gap beyond this is suspicious.
_GAP_WARN_SECONDS = 120.0


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        return rows
    return rows


def _parse_ts(row: dict) -> datetime | None:
    s = row.get("timestamp") or row.get("ts")
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_type(row: dict) -> str:
    return str(row.get("event") or row.get("type") or "")


def _has_experiment_evidence(run_dir: Path) -> bool:
    for row in _read_jsonl(run_dir / "experiment_runs.jsonl"):
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return True
    return False


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Return a structured postmortem for one run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    events = _read_jsonl(run_dir / "dashboard_events.jsonl")

    # Max gap between consecutive timestamped events.
    times = [t for t in (_parse_ts(e) for e in events) if t is not None]
    times.sort()
    max_gap = 0.0
    for a, b in zip(times, times[1:]):
        max_gap = max(max_gap, (b - a).total_seconds())

    # Dangling sub_rlm_spawned (no matching sub_rlm_complete).
    spawned = sum(1 for e in events if _event_type(e) == "sub_rlm_spawned")
    completed = sum(1 for e in events if _event_type(e) == "sub_rlm_complete")
    dangling = max(0, spawned - completed)

    # Final report + evidence/verdict mismatch (FM-004).
    fr_path = run_dir / "final_report.json"
    has_final_report = fr_path.exists()
    final_report: dict = {}
    if has_final_report:
        try:
            final_report = json.loads(fr_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            final_report = {}
    verdict = str(final_report.get("verdict") or "")
    baseline_metrics = final_report.get("baseline_metrics") or {}
    evidence = _has_experiment_evidence(run_dir)
    evidence_verdict_mismatch = (
        verdict in {"reproduced", "partial"} and not baseline_metrics and not evidence
    )

    # Prompt-cache read ratio.
    total_read = total_input = 0
    for row in _read_jsonl(run_dir / "cost_ledger.jsonl"):
        total_read += int(row.get("cache_read_input_tokens", 0) or 0)
        total_input += int(row.get("input_tokens", 0) or 0)
    cache_ratio = (total_read / (total_read + total_input)) if (total_read + total_input) else None

    status = ""
    ds_path = run_dir / "demo_status.json"
    if ds_path.exists():
        try:
            status = str(json.loads(ds_path.read_text(encoding="utf-8")).get("status") or "")
        except (OSError, json.JSONDecodeError):
            status = ""

    flags: list[str] = []
    if dangling > 0:
        flags.append(f"{dangling} dangling sub_rlm_spawned (no matching complete) — wedge shape")
    if max_gap >= _GAP_WARN_SECONDS:
        flags.append(f"max event gap {max_gap/60:.1f} min (>= {_GAP_WARN_SECONDS:.0f}s)")
    if evidence_verdict_mismatch:
        flags.append(f"verdict '{verdict}' with empty baseline_metrics and no experiment evidence (FM-004)")
    if cache_ratio is not None and cache_ratio < 0.5:
        flags.append(f"prompt-cache read ratio {cache_ratio:.3f} < 0.5 — prompt re-billed each iteration")

    return {
        "run_dir": str(run_dir),
        "status": status,
        "events": len(events),
        "max_event_gap_seconds": max_gap,
        "sub_rlm_spawned": spawned,
        "sub_rlm_complete": completed,
        "dangling_subcalls": dangling,
        "has_final_report": has_final_report,
        "verdict": verdict,
        "evidence_verdict_mismatch": evidence_verdict_mismatch,
        "cache_read_ratio": cache_ratio,
        "flags": flags,
    }


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = Path("runs") / arg
    return candidate


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_dir = _resolve_run_dir(argv[1])
    try:
        report = analyze_run(run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    if report["flags"]:
        print("\nFLAGS:", file=sys.stderr)
        for f in report["flags"]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nNo issues flagged.", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
