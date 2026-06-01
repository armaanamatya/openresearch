"""Proxy metric for implement_baseline optimization (no sub-agent call)."""
from __future__ import annotations

_REQUIRED_PLAN_KEYS = frozenset({
    "mode", "code_path", "dockerfile_path", "diff_summary",
    "commands_to_run", "assumptions_applied",
})


def code_plan_structural_validity(output: object) -> tuple[float, str]:
    """Score structural validity of an implement_baseline output dict.

    Used as a PROXY metric during GEPA optimization — no actual sub-agent call.
    Score = fraction of required fields present and non-empty.
    """
    if not isinstance(output, dict):
        return 0.0, "not a dict"
    scores = []
    for key in _REQUIRED_PLAN_KEYS:
        val = output.get(key)
        if val is None:
            scores.append(0.0)
        elif isinstance(val, (str, list)) and len(val) == 0:
            scores.append(0.0)
        else:
            scores.append(1.0)
    score = sum(scores) / len(scores)
    missing = [k for k in _REQUIRED_PLAN_KEYS if not output.get(k)]
    return score, f"missing_fields={missing}" if missing else "all fields present"
