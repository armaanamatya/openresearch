"""MUSE-lite per-paper negative lessons (FLAG-2, ``OPENRESEARCH_NEGATIVE_LESSONS``).

Cross-run, per-paper failure memory. After a run finalizes, mine the
agent-correctable ``failure_class`` rows from ``experiment_runs.jsonl`` into
``runs/_lessons/<arxiv_id>.json``; the next run of the same ``arxiv_id`` injects
the *active* lessons into the implementer's guidance (advisory only).

Guardrails:
- **Recurrence-gated promotion**: a lesson is "active" only at ``occurrences>=2``,
  EXCEPT ``dockerfile_invalid`` (active at 1 — a malformed Dockerfile is
  deterministically wrong, no need to wait for a repeat).
- **Classifier-sourced fix only**: the advice text is the failure classifier's
  ``suggested_fix`` — never agent prose.
- **Opportunity-aware retirement**: a lesson whose class did not recur in a run
  that *did* run experiments accrues staleness; at ``staleness>=3`` it is
  removed (simplification of "phase reached": we proxy phase with "the run
  produced any experiment rows", and note it here).
- **Caps**: at most ``MAX_LESSONS`` active lessons, each ``MAX_FIX_LEN`` chars.

Default OFF: when the flag is unset, mining is a no-op and the block is "".
Needs ``arxiv_id``; absent → no-op. Fail-soft throughout. The config shim
aliases ``OPENRESEARCH_NEGATIVE_LESSONS`` <-> ``OPENRESEARCH_NEGATIVE_LESSONS``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# agent-correctable failure classes worth remembering across runs (mirrors
# primitives._CODEX_AGENT_CORRECTABLE_FAILURES; duplicated here to avoid a heavy
# import in the implementer-prompt path).
CORRECTABLE = frozenset(
    {
        "syntax_error",
        "missing_module",
        "requirements_not_found",
        "dockerfile_invalid",
        "contract_violation",
        "scope_shape_violation",
    }
)
PROMOTE_AT_ONE = frozenset({"dockerfile_invalid"})

MAX_LESSONS = 5
MAX_FIX_LEN = 200
RETIRE_STALENESS = 3


def _enabled() -> bool:
    return os.environ.get("OPENRESEARCH_NEGATIVE_LESSONS", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _store_path(runs_root: Path | str, arxiv_id: str) -> Path:
    safe = "".join(c for c in arxiv_id if c.isalnum() or c in "._-")[:80]
    return Path(runs_root) / "_lessons" / f"{safe}.json"


def _is_promoted(lesson: dict[str, Any]) -> bool:
    occ = int(lesson.get("occurrences", 0))
    if lesson.get("failure_class") in PROMOTE_AT_ONE:
        return occ >= 1
    return occ >= 2


def _read_store(path: Path) -> dict[str, dict[str, Any]]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _scan_failures(project_dir: Path | str) -> tuple[dict[str, str], int]:
    """Return ({failure_class: suggested_fix}, n_experiment_rows) for this run."""
    seen: dict[str, str] = {}
    rows = 0
    path = Path(project_dir) / "experiment_runs.jsonl"
    try:
        if not path.exists():
            return seen, 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rows += 1
            if row.get("success"):
                continue
            fclass = row.get("failure_class")
            if fclass in CORRECTABLE:
                fix = str(row.get("suggested_fix") or "").strip()[:MAX_FIX_LEN]
                if fix:
                    seen[fclass] = fix  # last suggested_fix for the class wins
    except Exception:
        return seen, rows
    return seen, rows


def mine_lessons(
    project_dir: Path | str,
    runs_root: Path | str,
    arxiv_id: str | None,
) -> None:
    """Update the per-paper lesson store from this run. No-op unless enabled and
    ``arxiv_id`` is known. Fail-soft."""
    if not _enabled() or not arxiv_id:
        return
    try:
        seen, n_rows = _scan_failures(project_dir)
        path = _store_path(runs_root, arxiv_id)
        store = _read_store(path)

        for fclass, fix in seen.items():
            entry = store.setdefault(
                fclass, {"failure_class": fclass, "occurrences": 0, "staleness": 0}
            )
            entry["occurrences"] = int(entry.get("occurrences", 0)) + 1
            entry["staleness"] = 0
            entry["suggested_fix"] = fix  # refresh to the latest classifier fix

        # Opportunity-aware retirement: only when the run actually ran
        # experiments (proxy for "reached the phase") and the class didn't recur.
        if n_rows > 0:
            for fclass in list(store):
                if fclass in seen:
                    continue
                store[fclass]["staleness"] = int(store[fclass].get("staleness", 0)) + 1
                if store[fclass]["staleness"] >= RETIRE_STALENESS:
                    store.pop(fclass, None)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        return  # fail-soft: lesson bookkeeping must never break finalize


def active_lessons(runs_root: Path | str, arxiv_id: str | None) -> list[dict[str, Any]]:
    """Return promoted lessons (recurrence-gated), highest-occurrence first,
    capped at MAX_LESSONS. Empty when disabled / unknown paper / none promoted."""
    if not _enabled() or not arxiv_id:
        return []
    store = _read_store(_store_path(runs_root, arxiv_id))
    promoted = [v for v in store.values() if _is_promoted(v)]
    promoted.sort(key=lambda v: int(v.get("occurrences", 0)), reverse=True)
    return promoted[:MAX_LESSONS]


def negative_lessons_block(runs_root: Path | str, arxiv_id: str | None) -> str:
    """Render active lessons as an implementer-guidance block, or "" when none."""
    lessons = active_lessons(runs_root, arxiv_id)
    if not lessons:
        return ""
    lines = [
        "NEGATIVE LESSONS (this paper failed these ways before — avoid them):",
    ]
    for ls in lessons:
        fix = str(ls.get("suggested_fix") or "").strip()[:MAX_FIX_LEN]
        lines.append(f"- [{ls.get('failure_class')}] {fix}")
    return "\n".join(lines)
