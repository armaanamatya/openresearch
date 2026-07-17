"""Per-experiment GPU-efficiency ledger — display-only efficiency instrumentation.

WHY: ``experiment_runs.jsonl`` records WHAT an experiment did (success, metrics,
logs) but not HOW LONG it occupied a GPU or what it likely cost — there is no
per-experiment efficiency signal today. This module adds that signal as a
strictly separate, additive artifact:

  - ``runs/<id>/gpu_ledger.jsonl`` — one row per ``run_experiment`` call:
    ``{experiment_run_id, start_ts, end_ts, gpu_hours, provider,
    rate_usd_per_hr, est_cost_usd}``.
  - ``aggregate_gpu_cost(project_dir)`` folds those rows into a display-only
    summary (``{total_gpu_hours, total_est_cost_usd, by_experiment}``) for the
    scorecard / UI.

This is DELIBERATELY NOT the token ``cost_ledger.jsonl`` (that ledger tracks
per-primitive LLM $ spend; this one tracks GPU wall-clock $ spend) — the two are
orthogonal spend axes and must not be conflated into one schema.

NORTH STAR: this is a DISPLAY-ONLY efficiency metric. It is never read by the
verdict authority, never folds into ``meets_target``/the rubric, and never gates
anything — per the harness's "evidence, not grade" invariant, no new signal may
raise or lower a reproduction verdict.

GATING: ``OPENRESEARCH_GPU_LEDGER`` (default OFF). Off ⇒ ``append_gpu_ledger`` is
a pure no-op (returns False, writes nothing); the ``primitives.py`` stamp seam
(Task 3b) likewise adds none of ``start_ts``/``end_ts``/``gpu_plan``/``retry_id``
to the ``experiment_runs.jsonl`` row — byte-identical to today.

FAIL-SOFT: every public function is wrapped so a bad timestamp, a missing
directory, or any other unexpected input degrades to a safe value/no-op rather
than raising into the run.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.agents.rlm.feature_flags import env_truthy

_LEDGER_FILENAME = "gpu_ledger.jsonl"


def gpu_ledger_enabled() -> bool:
    """Master gate for the per-experiment GPU ledger (default OFF)."""
    return env_truthy("OPENRESEARCH_GPU_LEDGER")


def _parse_iso_ts(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp; ``None`` on anything unparseable.

    Fail-soft by design: a malformed/placeholder timestamp must degrade to a
    zero-hour reading, never raise into the run.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _gpu_hours(start_ts: object, end_ts: object) -> float:
    """Wall-clock hours between two ISO timestamps; 0.0 if unparseable/negative."""
    start = _parse_iso_ts(start_ts)
    end = _parse_iso_ts(end_ts)
    if start is None or end is None:
        return 0.0
    try:
        delta_s = (end - start).total_seconds()
    except Exception:  # noqa: BLE001 — defensive; stay fail-soft even here
        return 0.0
    if delta_s <= 0:
        return 0.0
    return round(delta_s / 3600.0, 6)


def _atomic_append_line(path: Path, line: str) -> None:
    """Durably append ``line`` (no trailing newline) to ``path``.

    Read-existing / write-temp-with-full-content / os.replace: the whole new
    content (existing bytes + the new line) is written to a temp file in the same
    directory and atomically swapped in, so a crash mid-write can never leave a
    torn line in the durable log.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 — treat unreadable as empty, never crash
            existing = ""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(existing)
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(line)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)  # atomic swap onto the durable log
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:  # noqa: BLE001
            pass


def append_gpu_ledger(
    project_dir: Path | str,
    *,
    experiment_run_id: str,
    start_ts: str,
    end_ts: str,
    gpu_plan: dict[str, Any] | None,
    provider: str,
    rate_usd_per_hr: float,
) -> bool:
    """Atomically append one GPU-cost row to ``runs/<id>/gpu_ledger.jsonl``.

    Display-only efficiency instrumentation, entirely separate from the token
    ``cost_ledger``. ``gpu_plan`` is accepted for caller symmetry with the
    ``experiment_runs.jsonl`` stamp (Task 3b) but is not itself persisted onto
    the ledger row (the row schema is exactly ``{experiment_run_id, start_ts,
    end_ts, gpu_hours, provider, rate_usd_per_hr, est_cost_usd}``).

    Off-flag (or any internal failure) returns False and writes nothing; never
    raises into the run.
    """
    del gpu_plan  # accepted for interface symmetry; not part of the row schema
    if not gpu_ledger_enabled():
        return False
    try:
        hours = _gpu_hours(start_ts, end_ts)
        try:
            rate = float(rate_usd_per_hr or 0.0)
        except (TypeError, ValueError):
            rate = 0.0
        row = {
            "experiment_run_id": str(experiment_run_id or ""),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "gpu_hours": hours,
            "provider": str(provider or "unknown"),
            "rate_usd_per_hr": rate,
            "est_cost_usd": round(hours * rate, 6),
        }
        path = Path(project_dir) / _LEDGER_FILENAME
        _atomic_append_line(path, json.dumps(row, default=str))
        return True
    except Exception:  # noqa: BLE001 — the GPU ledger must never break the run
        return False


def aggregate_gpu_cost(project_dir: Path | str) -> dict[str, Any]:
    """Fold ``gpu_ledger.jsonl`` into a display-only cost summary.

    Returns ``{total_gpu_hours, total_est_cost_usd, by_experiment}`` where
    ``by_experiment[experiment_run_id]`` is ``{gpu_hours, est_cost_usd}`` (summed
    across every row for that id — e.g. repeated repair-loop calls).

    Never raises: an absent/corrupt ledger degrades to an all-zero summary. NEVER
    a fitness term — this is read by the scorecard/UI only.
    """
    empty = {"total_gpu_hours": 0.0, "total_est_cost_usd": 0.0, "by_experiment": {}}
    try:
        path = Path(project_dir) / _LEDGER_FILENAME
        if not path.exists():
            return empty
        total_hours = 0.0
        total_cost = 0.0
        by_experiment: dict[str, dict[str, float]] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except Exception:  # noqa: BLE001 — tolerate a torn/corrupt line
                continue
            if not isinstance(row, dict):
                continue
            exp_id = str(row.get("experiment_run_id") or "")
            if not exp_id:
                continue
            hours = float(row.get("gpu_hours") or 0.0)
            cost = float(row.get("est_cost_usd") or 0.0)
            total_hours += hours
            total_cost += cost
            bucket = by_experiment.setdefault(exp_id, {"gpu_hours": 0.0, "est_cost_usd": 0.0})
            bucket["gpu_hours"] = round(bucket["gpu_hours"] + hours, 6)
            bucket["est_cost_usd"] = round(bucket["est_cost_usd"] + cost, 6)
        return {
            "total_gpu_hours": round(total_hours, 6),
            "total_est_cost_usd": round(total_cost, 6),
            "by_experiment": by_experiment,
        }
    except Exception:  # noqa: BLE001 — aggregation must never break the run
        return empty


__all__ = ["gpu_ledger_enabled", "append_gpu_ledger", "aggregate_gpu_cost"]
