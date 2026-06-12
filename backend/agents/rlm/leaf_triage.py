"""Deterministic weak-leaf triage — turn grader justifications into repair plans.

Every low-scoring run this campaign lost most of its points the same way: not
to failed training but to evidence-legibility gaps the grader names in prose —
a figure never rendered from data already on disk, a measured fact never
written into provenance.json, per-cell results never aggregated, a protocol
detail (whitening/dropout) implemented nowhere the grader can see, or one cell
whose result contradicts the paper because of a bad hyperparameter. The
2026-06-11/12 Adam runs recovered 0.0→0.716 purely by an operator reading
``weak_leaves`` and steering those exact repairs.

This module automates that triage, deterministically and paper-agnostically:
classify each weak leaf's justification text into a repair class, GROUND the
classification against actual disk state (a "render the figure" directive is
only issued when the underlying data demonstrably exists), attach a cost class
(``none`` = aggregation/rendering/provenance, no retraining; ``targeted_rerun``
= one family/cell; ``review`` = needs judgment), and order the plan by cost
then severity — cheapest, highest-impact repairs first.

Consumers: ``verify_against_rubric`` attaches the plan to its result (the root
sees it immediately) and persists ``rlm_state/leaf_triage.json``;
``baseline_implementation`` appends a compact directive block to the next
implementer prompt. Zero LLM calls; fail-soft everywhere; default ON with the
``REPROLAB_LEAF_TRIAGE=0`` escape hatch (env_pin precedent — this is a
correctness/efficiency rail, not an experiment).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_FLAG = "REPROLAB_LEAF_TRIAGE"
STATE_FILE = "leaf_triage.json"

_MAX_DIRECTIVES = 8
_MAX_DIRECTIVE_CHARS = 320
_WEAK_THRESHOLD = 0.6

# Repair classes, cheapest first. Order is the plan's sort key.
_COST_ORDER = {"none": 0, "targeted_rerun": 1, "review": 2}

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Result contradicts the paper — almost always one cell's hyperparameters.
    ("result_quality", re.compile(
        r"contradict|ranked (last|worst)|does not (hold|match|reach)|wrong "
        r"(ordering|direction)|opposite (of|to)|underperform|fails? the paper'?s",
        re.IGNORECASE)),
    # A figure/table artifact the grader looked for and did not find.
    ("render_artifact", re.compile(
        r"no figure|missing figure|figure[- ]?\d+[- ]?style artifact|not "
        r"rendered|no .{0,30}(plot|curve|figure|chart)|fig[_ ]?\d.{0,40}"
        r"(missing|absent|no evidence)|(missing|absent|lacks).{0,40}fig",
        re.IGNORECASE)),
    # Per-cell/per-family results that never reached the canonical aggregate.
    ("aggregation_gap", re.compile(
        r"not aggregated|(missing|absent|empty).{0,30}(per_model|per-model)|"
        r"only one experiment|metrics\.json.{0,40}(missing|lacks|contains no)|"
        r"cell director(y|ies).{0,60}(but|yet).{0,40}(no|not|missing)",
        re.IGNORECASE)),
    # Measured/derivable facts the text-only grader could not verify.
    ("provenance_gap", re.compile(
        r"not (recorded|confirmed|verifiable|evidenced|documented)|only an "
        r"assumption|no (evidence|record) (of|that|for)|cannot (be )?verif|"
        r"provenance\.json.{0,40}(lacks|missing|no )|nothing .{0,40}evidences",
        re.IGNORECASE)),
    # The paper's protocol detail (preprocessing/regularisation) absent.
    ("protocol_gap", re.compile(
        r"whitening|dropout|augment|normali[sz]ation|preprocess|weight.?decay|"
        r"learning.?rate schedule|warm.?up",
        re.IGNORECASE)),
)

_DIRECTIVES: dict[str, str] = {
    "render_artifact": (
        "RENDER, don't retrain: the underlying data exists on disk ({evidence}) "
        "— render the named figure/table from it and emit its axis sidecar "
        "(emit_figure_sidecar) so the text-only grader can read the axes."
    ),
    "provenance_gap": (
        "RECORD, don't retrain: the named facts already exist in your "
        "config/code — write them into provenance.json experiments "
        "(emit_provenance: epochs, batch_size, arch, per-optimizer hparams, "
        "init_fingerprint, preprocessing flags)."
    ),
    "aggregation_gap": (
        "AGGREGATE, don't retrain: per-cell results exist under code/outputs/ "
        "({evidence}) — fold them into metrics.json per_model (+ history) so "
        "the scorer sees every measured cell."
    ),
    "protocol_gap": (
        "TARGETED re-run: implement the named protocol detail (whitening / "
        "dropout / augmentation / schedule) in THAT family only, re-run just "
        "that family/cell, and record the flag in provenance.json — never the "
        "whole matrix."
    ),
    "result_quality": (
        "TARGETED re-run: the result contradicts the paper — check the "
        "hyperparameters for THAT cell against the champion configs in your "
        "PRIOR-ATTEMPT MEASURED EVIDENCE (a bad lr is the usual cause) and "
        "re-run only that cell."
    ),
    "review": (
        "Needs judgment — read the justification; no deterministic repair "
        "template applies."
    ),
}

_COST: dict[str, str] = {
    "render_artifact": "none",
    "provenance_gap": "none",
    "aggregation_gap": "none",
    "protocol_gap": "targeted_rerun",
    "result_quality": "targeted_rerun",
    "review": "review",
}


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() not in ("0", "false", "off")


def _disk_facts(project_dir: Path) -> dict[str, Any]:
    """Cheap, fail-soft facts grounding the classification."""
    facts: dict[str, Any] = {
        "outputs_metrics": 0,
        "has_history": False,
        "has_curves": False,
        "has_sweep": False,
        "has_provenance": False,
    }
    try:
        code = project_dir / "code"
        facts["outputs_metrics"] = sum(
            1 for _ in (code / "outputs").glob("*/*/metrics.json")
        ) if (code / "outputs").is_dir() else 0
        facts["has_curves"] = (
            any((code / "outputs").glob("*/*/training_curves.json"))
            if (code / "outputs").is_dir() else False
        ) or (code / "training_curves.json").is_file()
        facts["has_provenance"] = any(code.glob("**/provenance.json"))
        mpath = code / "metrics.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if isinstance(m, dict):
                facts["has_history"] = bool(m.get("history"))
                facts["has_sweep"] = any(
                    isinstance(k, str) and "sweep" in k.lower() for k in m
                )
    except Exception:  # noqa: BLE001 — facts are advisory grounding
        logger.debug("leaf_triage: disk facts failed", exc_info=True)
    return facts


def _classify(text: str) -> str:
    for klass, pat in _PATTERNS:
        if pat.search(text):
            return klass
    return "review"


def _ground(klass: str, facts: dict[str, Any]) -> str:
    """Downgrade a no-cost class when the disk can't actually support it."""
    if klass == "render_artifact" and not (
        facts.get("has_history") or facts.get("has_curves") or facts.get("has_sweep")
    ):
        return "protocol_gap"  # data may genuinely not exist → produce it, scoped
    if klass == "aggregation_gap" and not facts.get("outputs_metrics"):
        return "review"
    return klass


def _evidence_note(klass: str, facts: dict[str, Any]) -> str:
    if klass == "render_artifact":
        bits = [n for n, k in (
            ("history block", "has_history"),
            ("training_curves", "has_curves"),
            ("sweep data", "has_sweep"),
        ) if facts.get(k)]
        return ", ".join(bits) or "metrics on disk"
    if klass == "aggregation_gap":
        return f"{facts.get('outputs_metrics', 0)} per-cell metrics.json files"
    return ""


def triage_weak_leaves(
    weak_leaves: list[dict],
    project_dir: Path | str,
    *,
    max_directives: int = _MAX_DIRECTIVES,
) -> dict[str, Any]:
    """Classify weak leaves into a cost-ordered repair plan. Never raises."""
    try:
        project_dir = Path(project_dir)
        facts = _disk_facts(project_dir)
        plan: list[dict[str, Any]] = []
        for leaf in weak_leaves or []:
            if not isinstance(leaf, dict):
                continue
            try:
                score = float(leaf.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score >= _WEAK_THRESHOLD:
                continue
            text = " ".join(
                str(leaf.get(k) or "") for k in ("justification", "requirement", "label")
            )
            klass = _ground(_classify(text), facts)
            directive = _DIRECTIVES[klass].format(
                evidence=_evidence_note(klass, facts) or "data on disk"
            )[:_MAX_DIRECTIVE_CHARS]
            plan.append({
                "leaf_id": str(leaf.get("id") or "")[:16],
                "score": score,
                "repair_class": klass,
                "cost": _COST[klass],
                "directive": directive,
                "justification": str(leaf.get("justification") or "")[:200],
            })
        plan.sort(key=lambda d: (_COST_ORDER.get(d["cost"], 9), d["score"]))
        plan = plan[:max_directives]
        n_free = sum(1 for d in plan if d["cost"] == "none")
        summary = (
            f"{len(plan)} weak leaves triaged: {n_free} repairable with NO "
            f"retraining (render/record/aggregate), "
            f"{sum(1 for d in plan if d['cost'] == 'targeted_rerun')} need a "
            f"targeted re-run, "
            f"{sum(1 for d in plan if d['cost'] == 'review')} need judgment."
        ) if plan else ""
        return {"plan": plan, "facts": facts, "summary": summary}
    except Exception:  # noqa: BLE001 — triage is advisory, never fatal
        logger.debug("leaf_triage: triage failed", exc_info=True)
        return {"plan": [], "facts": {}, "summary": ""}


def persist(project_dir: Path | str, triage: dict[str, Any]) -> None:
    try:
        state_dir = Path(project_dir) / "rlm_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / STATE_FILE).write_text(
            json.dumps(triage, indent=2), encoding="utf-8"
        )
    except OSError:
        logger.debug("leaf_triage: persist failed", exc_info=True)


def guidance_block(project_dir: Path | str, *, max_chars: int = 1600) -> str:
    """Compact directive block for the implementer prompt (empty when N/A)."""
    if not is_enabled():
        return ""
    try:
        path = Path(project_dir) / "rlm_state" / STATE_FILE
        if not path.is_file():
            return ""
        triage = json.loads(path.read_text(encoding="utf-8"))
        plan = triage.get("plan") or []
        if not plan:
            return ""
        lines = [
            "\n\nLEAF REPAIR PLAN (deterministic triage of the last verify's weak "
            "leaves — ordered cheapest first; most points are recoverable WITHOUT "
            "retraining):",
        ]
        for d in plan:
            lines.append(
                f"  [{d.get('cost')}] leaf {d.get('leaf_id')} "
                f"(score {d.get('score')}): {d.get('directive')}"
            )
        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[: max_chars - 15].rstrip() + "\n  (truncated)"
        return block + "\n"
    except Exception:  # noqa: BLE001
        logger.debug("leaf_triage: guidance block failed", exc_info=True)
        return ""


__all__ = [
    "ENV_FLAG",
    "guidance_block",
    "is_enabled",
    "persist",
    "triage_weak_leaves",
]
