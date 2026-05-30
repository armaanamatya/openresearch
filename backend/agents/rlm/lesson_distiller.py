"""Per-paper negative lessons (MUSE-lite) — deterministic cross-run failure memory.

See docs/superpowers/specs/2026-05-30-negative-lessons-design.md.

When an *agent-correctable* failure recurs across separate runs of the same paper
(BUG-NEW-042 prose-in-Dockerfile, a missing per-model metric), this carries a
structured, class-tagged lesson forward so the next run's implementer avoids it.
Mined deterministically from the *already-classified* ``failure_class`` rows in
``experiment_runs.jsonl`` — NO LLM. Stored runtime-only at
``runs/_lessons/<arxiv_id>.json`` (gitignored via ``/runs/``). Fail-soft,
flag-gated (``REPROLAB_NEGATIVE_LESSONS``, default off).

Guarantees (spec §4/§5/§7):
  * ``suggested_fix`` is ALWAYS re-derived from the classifier's canonical map,
    never the agent-authored ``experiment_runs.jsonl::suggested_fix`` field.
  * Promotion is recurrence-gated: a lesson is ``candidate`` until
    ``occurrences >= _promote_threshold`` (default 2; structurally-deterministic
    classes like ``dockerfile_invalid`` promote at 1).
  * Retirement is opportunity-aware off ``experiment_runs.jsonl`` ONLY (the
    conservative, reliable signal): a lesson ages only when its gating phase was
    reached and it did not fire; retired at ``staleness >= 3``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Final

from backend.agents.rlm.failure_classifier import suggested_fix_for_class

logger = logging.getLogger(__name__)

_VERSION: Final[str] = "v1"
_ENABLE_ENV_VAR: Final[str] = "REPROLAB_NEGATIVE_LESSONS"
_LESSONS_SUBDIR: Final[str] = "_lessons"

_lock = threading.Lock()

# Agent-correctable classes only — environmental/transient classes won't recur
# deterministically and would be prompt junk (spec §3).
_LESSON_WORTHY_CLASSES: Final[frozenset[str]] = frozenset({
    "missing_module", "torch_redundancy", "requirements_not_found",
    "missing_dataset", "syntax_error", "scope_shape_violation",
    "contract_violation", "dockerfile_invalid",
})

# Recurrence-gated promotion: default 2; structurally-deterministic classes at 1.
_PROMOTE_THRESHOLD: Final[dict[str, int]] = {"dockerfile_invalid": 1}
_DEFAULT_PROMOTE_THRESHOLD: Final[int] = 2

_RETIRE_STALENESS: Final[int] = 3
_MAX_INJECT: Final[int] = 5
_FIX_TRUNC: Final[int] = 200

# Each class only has a chance to fire in a specific phase; ages only then.
_PHASE_ATTEMPTED: Final[str] = "EXPERIMENT_ATTEMPTED"
_PHASE_SUCCEEDED: Final[str] = "EXPERIMENT_SUCCEEDED"
_OPPORTUNITY_PHASE: Final[dict[str, str]] = {
    "dockerfile_invalid": _PHASE_ATTEMPTED,
    "torch_redundancy": _PHASE_ATTEMPTED,
    "requirements_not_found": _PHASE_ATTEMPTED,
    "missing_module": _PHASE_ATTEMPTED,
    "missing_dataset": _PHASE_ATTEMPTED,
    "syntax_error": _PHASE_ATTEMPTED,
    "scope_shape_violation": _PHASE_SUCCEEDED,
    "contract_violation": _PHASE_SUCCEEDED,
}


def is_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV_VAR, "").strip().lower() in ("1", "on", "true", "yes")


def _promote_threshold(klass: str) -> int:
    return _PROMOTE_THRESHOLD.get(klass, _DEFAULT_PROMOTE_THRESHOLD)


def _lessons_path(runs_root: Path, arxiv_id: str) -> Path:
    return runs_root / _LESSONS_SUBDIR / f"{arxiv_id}.json"


def _empty(arxiv_id: str) -> dict:
    return {"version": _VERSION, "arxiv_id": arxiv_id, "lessons": []}


def _load(runs_root: Path, arxiv_id: str) -> dict:
    p = _lessons_path(runs_root, arxiv_id)
    if not p.exists():
        return _empty(arxiv_id)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty(arxiv_id)
    if not isinstance(obj, dict) or not isinstance(obj.get("lessons"), list):
        return _empty(arxiv_id)
    obj.setdefault("version", _VERSION)
    obj.setdefault("arxiv_id", arxiv_id)
    return obj


def _persist(runs_root: Path, arxiv_id: str, obj: dict) -> None:
    d = runs_root / _LESSONS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    p = _lessons_path(runs_root, arxiv_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def read_lessons(runs_root: Path, arxiv_id: str | None) -> dict:
    """Return the per-paper lessons object; empty when off / no id / error."""
    if not is_enabled() or not arxiv_id or not isinstance(runs_root, Path):
        return _empty(arxiv_id or "")
    try:
        return _load(runs_root, arxiv_id)
    except Exception:  # noqa: BLE001 — fail-soft; a read must never raise
        return _empty(arxiv_id or "")


def _read_experiment_rows(project_dir: Path) -> list[dict]:
    path = project_dir / "experiment_runs.jsonl"
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                rows.append(entry)
    except OSError:
        return rows
    return rows


def _fired_classes(rows: list[dict]) -> set[str]:
    fired: set[str] = set()
    for e in rows:
        fc = e.get("failure_class")
        if isinstance(fc, str) and fc in _LESSON_WORTHY_CLASSES:
            fired.add(fc)
    return fired


def _phases_reached(rows: list[dict]) -> set[str]:
    """Conservative, reliable phase set derived from experiment_runs.jsonl only."""
    phases: set[str] = set()
    if rows:
        phases.add(_PHASE_ATTEMPTED)
    for e in rows:
        if e.get("success") is True:
            m = e.get("metrics")
            if isinstance(m, dict) and m:
                phases.add(_PHASE_SUCCEEDED)
                break
    return phases


def _classifier_fix(klass: str) -> str:
    return (suggested_fix_for_class(klass) or "")[:_FIX_TRUNC]


def mine_lessons(project_dir: Path, runs_root: Path, arxiv_id: str | None,
                 *, run_id: str | None = None) -> None:
    """Post-run: upsert/retire per-paper lessons from this run's experiment rows.

    Fail-soft; no-op when disabled or ``arxiv_id`` is None.
    """
    if not is_enabled():
        return
    if not arxiv_id or not isinstance(project_dir, Path) or not isinstance(runs_root, Path):
        return
    try:
        rows = _read_experiment_rows(project_dir)
        fired = _fired_classes(rows)
        phases = _phases_reached(rows)
        run_id = run_id or project_dir.name
        with _lock:
            obj = _load(runs_root, arxiv_id)
            lessons: list[dict] = obj.setdefault("lessons", [])
            by_class: dict[str, dict] = {
                l["failure_class"]: l for l in lessons
                if isinstance(l, dict) and isinstance(l.get("failure_class"), str)
            }

            # --- Promotion pass: classes that fired this run -----------------
            for klass in fired:
                l = by_class.get(klass)
                if l is None:
                    l = {
                        "failure_class": klass,
                        "suggested_fix": _classifier_fix(klass),
                        "suggested_fix_source": "classifier",
                        "occurrences": 0,
                        "status": "candidate",
                        "first_seen_run": run_id,
                        "last_seen_run": run_id,
                        "staleness": 0,
                    }
                    lessons.append(l)
                    by_class[klass] = l
                l["occurrences"] = int(l.get("occurrences", 0)) + 1
                l["last_seen_run"] = run_id
                l["staleness"] = 0
                # Always re-derive the fix from the classifier (never agent prose).
                l["suggested_fix"] = _classifier_fix(klass)
                l["suggested_fix_source"] = "classifier"
                l["status"] = (
                    "active" if l["occurrences"] >= _promote_threshold(klass) else "candidate"
                )

            # --- Retirement pass: opportunity-aware staleness ----------------
            survivors: list[dict] = []
            for l in lessons:
                klass = l.get("failure_class")
                if klass in fired:
                    survivors.append(l)
                    continue
                phase = _OPPORTUNITY_PHASE.get(klass)
                if phase is not None and phase in phases:
                    l["staleness"] = int(l.get("staleness", 0)) + 1
                if int(l.get("staleness", 0)) >= _RETIRE_STALENESS:
                    continue  # retire (drop)
                survivors.append(l)
            obj["lessons"] = survivors

            _persist(runs_root, arxiv_id, obj)
    except Exception:  # noqa: BLE001 — mining MUST NOT block the run
        logger.exception("lesson_distiller.mine_lessons failed")


def render_block(runs_root: Path, arxiv_id: str | None) -> str:
    """Return the implementer-guidance block of ACTIVE lessons (or '' when off/empty)."""
    if not is_enabled() or not arxiv_id or not isinstance(runs_root, Path):
        return ""
    try:
        obj = _load(runs_root, arxiv_id)
    except Exception:  # noqa: BLE001 — fail-soft
        return ""
    active = [
        l for l in obj.get("lessons", [])
        if isinstance(l, dict) and l.get("status") == "active"
    ]
    if not active:
        return ""
    active.sort(
        key=lambda l: (int(l.get("occurrences", 0)), str(l.get("last_seen_run", ""))),
        reverse=True,
    )
    lines = ["NEGATIVE LESSONS FROM PRIOR RUNS OF THIS PAPER — avoid repeating these failures:"]
    for l in active[:_MAX_INJECT]:
        fix = str(l.get("suggested_fix") or "")[:_FIX_TRUNC]
        lines.append(f"- [{l.get('failure_class')}] {fix} (seen {int(l.get('occurrences', 0))}×)")
    return "\n".join(lines) + "\n"


__all__ = ["is_enabled", "mine_lessons", "read_lessons", "render_block"]
