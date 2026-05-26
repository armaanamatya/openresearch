"""Capture and read per-run timing data for the k-NN estimator.

`write_timing_json` is called from `write_final_report_rlm` immediately after
the `.preserved` marker is stamped.  It reads `dashboard_events.jsonl` and
`final_report.json` (already written) to produce `timing.json`:

    {
      "schema_version": 1,
      "wall_clock_s": 1234,
      "primitive_wall_clock_s": {"understand_section": 12.3, ...},
      "primitive_call_counts": {"understand_section": 3, ...},
      "gpu_hours": 0.34,
      "gpu_type": "rtx4090",
      "gpu_count": 1,
      "iterations": 4,
      "rubric_score": 0.42,
      "computed_at_utc": "..."
    }

`load_preserved_timings` walks the runs root and returns a list of dicts for
use by the k-NN estimator.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §timing
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

TIMING_SCHEMA_VERSION = 1


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _extract_primitive_wall_clocks(
    dashboard_path: Path,
) -> tuple[dict[str, float], dict[str, int]]:
    """Parse dashboard_events.jsonl to extract per-primitive wall-clock seconds
    and call counts.

    Pairs `status=start` with the next `status=ok|error` for each primitive
    by walking events in order and matching on primitive name.  Uses the
    `timestamp` field of the event pair to compute elapsed seconds.
    """
    if not dashboard_path.exists():
        return {}, {}

    # Collect all primitive_call events in order.
    events: list[dict] = []
    try:
        for line in dashboard_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "primitive_call":
                events.append(ev)
    except OSError:
        return {}, {}

    # Match start → ok/error pairs per primitive.
    # Use a stack per primitive name to handle (unlikely) interleaving.
    in_flight: dict[str, list[datetime]] = {}
    wall_clock_s: dict[str, float] = {}
    call_counts: dict[str, int] = {}

    for ev in events:
        prim = ev.get("primitive")
        status = ev.get("status")
        ts_raw = ev.get("timestamp")
        if not prim or not status or not ts_raw:
            continue
        ts = _parse_iso(ts_raw)
        if ts is None:
            continue

        if status == "start":
            in_flight.setdefault(prim, []).append(ts)
        elif status in ("ok", "error", "timeout"):
            stack = in_flight.get(prim, [])
            if stack:
                start_ts = stack.pop()
                elapsed = (ts - start_ts).total_seconds()
                if elapsed >= 0:
                    wall_clock_s[prim] = wall_clock_s.get(prim, 0.0) + elapsed
                    call_counts[prim] = call_counts.get(prim, 0) + 1

    return wall_clock_s, call_counts


def _extract_gpu_hours(project_dir: Path, gpu_count: int) -> float:
    """Estimate GPU hours from experiment_runs.jsonl timestamps.

    Each successful experiment_runs.jsonl entry represents one container
    execution on the GPU.  We use the run's timestamp as a proxy for
    completion time; a true start timestamp is not recorded in the current
    schema, so we fall back to wall_clock_s × gpu_count / 3600 when no
    experiment_runs timestamps are available.

    This is a conservative approximation: it treats each experiment_runs
    row as a distinct GPU-pod session, using the overall run wall-clock
    as a proxy when individual start times are not available.
    """
    exp_log = project_dir / "experiment_runs.jsonl"
    if not exp_log.exists():
        return 0.0

    # Collect all timestamps from experiment_runs.jsonl.
    timestamps: list[datetime] = []
    try:
        for line in exp_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_iso(row.get("timestamp"))
            if ts is not None:
                timestamps.append(ts)
    except OSError:
        return 0.0

    # Can't compute duration without at least two timestamps.
    if len(timestamps) < 2:
        return 0.0

    total_s = (max(timestamps) - min(timestamps)).total_seconds()
    return max(0.0, total_s * gpu_count / 3600.0)


def write_timing_json(project_dir: Path) -> Path | None:
    """Compute and write `runs/<id>/timing.json`.

    Reads final_report.json, dashboard_events.jsonl, rlm_state/gpu_plan.json,
    and experiment_runs.jsonl.  All reads are best-effort: missing files
    produce conservative zeros rather than raising.

    Returns the path written, or None if nothing could be produced.
    """
    project_dir = Path(project_dir)

    # --- Wall clock from final_report.json ---
    final_report_path = project_dir / "final_report.json"
    wall_clock_s: float = 0.0
    iterations: int = 0
    rubric_score: float | None = None
    started_at: str | None = None
    completed_at: str | None = None

    try:
        fr = json.loads(final_report_path.read_text(encoding="utf-8"))
        started_at = fr.get("started_at")
        completed_at = fr.get("completed_at")
        iterations = int(fr.get("iterations") or 0)
        rubric = fr.get("rubric") or {}
        overall = rubric.get("overall_score")
        if isinstance(overall, (int, float)):
            rubric_score = float(overall)
        # Wall clock
        started_dt = _parse_iso(started_at)
        completed_dt = _parse_iso(completed_at)
        if started_dt and completed_dt:
            wall_clock_s = max(0.0, (completed_dt - started_dt).total_seconds())
    except Exception:  # noqa: BLE001
        pass

    # If final_report.json lacks started_at/completed_at, fall back to
    # demo_status.json.
    if wall_clock_s == 0.0:
        try:
            ds = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
            started_dt2 = _parse_iso(ds.get("startedAt"))
            completed_dt2 = _parse_iso(ds.get("completedAt") or ds.get("updatedAt"))
            if started_dt2 and completed_dt2:
                wall_clock_s = max(0.0, (completed_dt2 - started_dt2).total_seconds())
        except Exception:  # noqa: BLE001
            pass

    # --- GPU info from gpu_plan.json ---
    gpu_type = "unknown"
    gpu_count = 1
    gpu_usd_per_hr = 0.0
    try:
        gp = json.loads(
            (project_dir / "rlm_state" / "gpu_plan.json").read_text(encoding="utf-8")
        )
        gpu_type = gp.get("short_name") or "unknown"
        gpu_count = int(gp.get("gpu_count") or 1)
        gpu_usd_per_hr = float(gp.get("sku_usd_per_hr") or 0.0)
    except Exception:  # noqa: BLE001
        pass

    # --- Per-primitive wall-clock ---
    prim_wc, prim_counts = _extract_primitive_wall_clocks(
        project_dir / "dashboard_events.jsonl"
    )

    # --- GPU hours ---
    # Prefer: sum of per-primitive run_experiment wall-clock × gpu_count
    # Fallback: experiment_runs.jsonl timestamp range × gpu_count
    gpu_hours = 0.0
    run_exp_s = prim_wc.get("run_experiment", 0.0)
    if run_exp_s > 0 and gpu_usd_per_hr > 0:
        gpu_hours = run_exp_s * gpu_count / 3600.0
    else:
        gpu_hours = _extract_gpu_hours(project_dir, gpu_count)

    timing = {
        "schema_version": TIMING_SCHEMA_VERSION,
        "wall_clock_s": round(wall_clock_s, 1),
        "primitive_wall_clock_s": {k: round(v, 3) for k, v in prim_wc.items()},
        "primitive_call_counts": prim_counts,
        "gpu_hours": round(gpu_hours, 4),
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "iterations": iterations,
        "rubric_score": rubric_score,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = project_dir / "timing.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    logger.info("timing: wrote %s (wall_clock_s=%.0f, iterations=%d)", out_path, wall_clock_s, iterations)
    return out_path


def load_preserved_timings(runs_root: Path) -> list[dict]:
    """Load all `timing.json` files from preserved runs.

    Returns a list of dicts (one per preserved run that has a timing.json).
    Silently skips runs without timing.json or with parse errors.
    """
    runs_root = Path(runs_root)
    result: list[dict] = []
    try:
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            if not (run_dir / ".preserved").exists():
                continue
            timing_path = run_dir / "timing.json"
            if not timing_path.exists():
                continue
            try:
                timing = json.loads(timing_path.read_text(encoding="utf-8"))
                timing["_run_id"] = run_dir.name
                result.append(timing)
            except Exception:  # noqa: BLE001
                continue
    except OSError:
        pass
    return result
