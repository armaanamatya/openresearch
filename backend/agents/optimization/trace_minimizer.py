"""Turn a completed RLM run's artifacts into a GEPA reflective-dataset record.

Schema: ``docs/superpowers/specs/2026-05-25-gepa-prompt-optimization-design.md`` §4.4.
Redaction rules: never leak paper corpus, secrets, env vars, REPL locals, or
artifact bodies. Stdout/stderr capped to first 200 + last 200 chars. Hash
``runs/...`` paths to short tags.

Output cap: ~8 KB per record after redaction. The minimizer trims if it
exceeds.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_MAX_RECORD_BYTES = 8 * 1024
_SNIPPET_MAX = 120
_TAIL_HEAD = 200
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
)
_PATH_PATTERN = re.compile(r"(?:runs|/tmp|/Users/[^/]+|/Volumes/[^/]+)/[\w./\-]+")


def _scrub(text: str) -> str:
    """Replace secrets + paths with redacted tags."""
    if not isinstance(text, str):
        return str(text)
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("<REDACTED:secret>", out)
    out = _PATH_PATTERN.sub(_hash_path_tag, out)
    return out


def _hash_path_tag(match: re.Match[str]) -> str:
    short = hashlib.sha256(match.group(0).encode()).hexdigest()[:8]
    return f"<path:{short}>"


def _truncate(text: str | None, *, limit: int = _SNIPPET_MAX) -> str:
    if not text:
        return ""
    t = _scrub(text)
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _truncate_stdout(text: str | None) -> str:
    if not text:
        return ""
    t = _scrub(text)
    if len(t) <= _TAIL_HEAD * 2:
        return t
    return f"{t[:_TAIL_HEAD]}\n...[{len(t) - _TAIL_HEAD * 2} chars elided]...\n{t[-_TAIL_HEAD:]}"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def _hermes_section(final_report: dict, run_dir: Path) -> dict:
    """Extract Hermes status + redacted unsupported_claims + evidence snippets."""
    hermes = final_report.get("hermes_audit") or {}
    status = hermes.get("status", "unavailable")
    unsupported = [_truncate(s, limit=200) for s in (hermes.get("unsupported_claims") or [])][:5]
    snippets: list[str] = []
    for ref in (hermes.get("evidence_refs") or [])[:3]:
        if isinstance(ref, dict):
            snippets.append(_truncate(ref.get("snippet", ""), limit=_SNIPPET_MAX))
    return {
        "status": status,
        "unsupported_claims": unsupported,
        "evidence_refs_snippets": snippets,
    }


def _hermes_multiplier(status: str) -> float:
    """G1: anti-Goodhart Hermes-clamped score multiplier (spec §6 G1)."""
    return {
        "grounded": 1.0,
        "caveat": 0.5,
        "unsupported": 0.0,
        "unavailable": 0.0,
        "system_error": 0.0,
    }.get(status, 0.0)


def build_record(
    *,
    component_id: str,
    example_id: str,
    paper_archetype: str,
    run_dir: Path,
    rubric_before: dict | None = None,
    weak_leaves_before: list[dict] | None = None,
    candidate_output: dict | None = None,
) -> dict:
    """Build one §4.4 reflective-dataset record from a completed eval's run_dir.

    Reads ``final_report.json``, ``cost_ledger.jsonl``, and
    ``dashboard_events.jsonl`` from ``run_dir``. Missing files → degraded but
    non-empty record (status carried forward as ``unavailable``, score 0.0).
    """
    final_report = _read_json(run_dir / "final_report.json") or {}
    rubric_after = (final_report.get("rubric") or final_report.get("rubric_score") or {})
    overall_after = float(rubric_after.get("overall_score") or 0.0)
    areas_after = {k: v for k, v in (rubric_after.get("areas") or {}).items()}
    weak_after = [
        {"name": str(leaf.get("name", "")), "score": float(leaf.get("score", 0.0))}
        for leaf in (rubric_after.get("weak_leaves") or [])[:10]
    ]

    events = _read_jsonl(run_dir / "dashboard_events.jsonl")
    forced_iter = sum(1 for e in events if e.get("event_type") == "run_warning"
                      and (e.get("payload") or {}).get("code") == "forced_iteration")
    blanket_declines = sum(1 for e in events if e.get("event_type") == "candidate_outcome"
                           and (e.get("payload") or {}).get("outcome") == "declined")

    experiments = _read_jsonl(run_dir / "experiment_runs.jsonl")
    last_exp = experiments[-1] if experiments else {}
    run_success = bool(last_exp.get("success"))
    repair_summaries = [
        _truncate(str(e.get("repair_context") or e.get("error") or ""), limit=200)
        for e in experiments if not e.get("success")
    ][:3]

    hermes = _hermes_section(final_report, run_dir)
    hermes_clamped = overall_after * _hermes_multiplier(hermes["status"])

    overall_before = float((rubric_before or {}).get("overall_score") or 0.0)
    areas_before = (rubric_before or {}).get("areas") or {}
    rubric_delta_areas = {
        k: float(areas_after.get(k, 0.0)) - float(areas_before.get(k, 0.0))
        for k in set(list(areas_after.keys()) + list(areas_before.keys()))
    }

    record = {
        "component_id": component_id,
        "example_id": example_id,
        "paper_archetype": paper_archetype,
        "score": round(hermes_clamped, 4),
        "hermes_clamped_score": round(hermes_clamped, 4),
        "input": {
            "rubric_overall_before": round(overall_before, 4),
            "rubric_areas_before": {k: round(float(v), 4) for k, v in areas_before.items()},
            "weak_leaves_before": [
                {**w, "rationale": _truncate(w.get("rationale"), limit=200)}
                for w in (weak_leaves_before or [])[:5]
            ],
            "current_results_digest": _truncate(
                json.dumps(final_report.get("metrics") or {}, default=str), limit=400
            ),
        },
        "candidate_output": candidate_output or {},
        "execution_trace": {
            "rubric_overall_after": round(overall_after, 4),
            "rubric_delta_areas": {k: round(v, 4) for k, v in rubric_delta_areas.items()},
            "weak_leaves_after": weak_after,
            "run_experiment_success": run_success,
            "repair_summaries": repair_summaries,
            "forced_iteration_warnings": forced_iter,
            "blanket_decline_count": blanket_declines,
            "hermes": hermes,
        },
    }

    blob = json.dumps(record, default=str)
    if len(blob.encode("utf-8")) > _MAX_RECORD_BYTES:
        # Trim non-essential sections until we fit.
        record["input"]["weak_leaves_before"] = record["input"]["weak_leaves_before"][:2]
        record["execution_trace"]["weak_leaves_after"] = record["execution_trace"]["weak_leaves_after"][:3]
        record["execution_trace"]["repair_summaries"] = record["execution_trace"]["repair_summaries"][:1]
        record["input"]["current_results_digest"] = _truncate(
            record["input"]["current_results_digest"], limit=200
        )
    return record


__all__ = ["build_record", "_hermes_multiplier"]
