"""Deterministic next-action card (feature #18).

WHY this module exists
----------------------
``recommend_next_tool`` (``primitives.py``) spends a PAID ``ctx.llm_client.complete``
call to advise the root on which tool to use next. But most of the time the next
action is NOT a judgement call — it is mechanically determined by the run's
lifecycle state: if there is no baseline yet, the next tool is *always*
``implement_baseline``; if the environment is not built, it is *always*
``build_environment``; and so on.  In those unambiguous cases the LLM call is
pure waste (money + latency + a non-deterministic answer for a deterministic
question).

This module derives a compact **state/action card** — lifecycle stage + last
primitive outcome + coverage gaps + remaining budget — purely from disk/ledger
state (NO paper corpus, NO LLM), and maps the inferred lifecycle stage onto the
single unambiguous next tool.  The stage logic is REUSED from
``root_progress.infer_required_stage`` (the same ladder the forced-iteration
backstop and ``run.py`` already use) — it is not duplicated here.

Gated behind ``OPENRESEARCH_ACTION_CARDS`` (default OFF).  When the flag is off,
or when the next action is ambiguous (no clear required stage), or on ANY error,
the caller falls through to the existing LLM path so behaviour is byte-for-byte
unchanged.  Fail-soft is mandatory: this module never raises into the primitive.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from backend.agents.rlm.root_progress import REQUIRED_STAGES, infer_required_stage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.agents.rlm.context import RunContext

__all__ = ["action_cards_enabled", "build_action_card"]


def action_cards_enabled() -> bool:
    """Whether the deterministic next-action card short-circuit is enabled.

    Reads ``OPENRESEARCH_ACTION_CARDS`` from the environment; truthy values are
    ``"1"``/``"true"``/``"yes"`` (case-insensitive).  Default OFF — unset means
    ``recommend_next_tool`` keeps calling the LLM, byte-for-byte unchanged.
    """
    return os.environ.get("OPENRESEARCH_ACTION_CARDS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


# The single unambiguous next tool for each lifecycle stage.  ``can_finalize`` is
# intentionally ABSENT: once the run is scored, whether to finalize, propose
# improvements, or iterate is a genuine judgement call — that case must fall
# through to the LLM, never short-circuit.
_STAGE_TO_TOOL: dict[str, str] = {
    "need_baseline": "implement_baseline",
    "need_environment": "build_environment",
    "need_experiment": "run_experiment",
    "need_verification": "verify_against_rubric",
}

_STAGE_REASON: dict[str, str] = {
    "need_baseline": "no baseline implementation exists yet",
    "need_environment": "code exists but the environment is not built",
    "need_experiment": "environment is ready but no experiment has run",
    "need_verification": "an experiment ran but it has not been scored",
}


def _code_path_exists(ctx: "RunContext") -> bool:
    """Whether a usable baseline implementation exists on disk.

    Mirror of ``run.py``'s closure of the same name (kept inline to keep this
    module import-light + side-effect-free): a non-empty ``code/commands.json``
    JSON list AND ≥1 runnable source file under ``code/``.
    """
    code_dir = ctx.project_dir / "code"
    commands_path = code_dir / "commands.json"
    try:
        commands = json.loads(commands_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(commands, list) or not commands:
        return False
    runnable_suffixes = {".py", ".sh", ".bash", ".ps1"}
    for file in code_dir.rglob("*"):
        if not file.is_file() or file.name == "commands.json":
            continue
        if file.suffix.lower() in runnable_suffixes or file.name in {
            "Dockerfile",
            "Makefile",
        }:
            return True
    return False


def _env_built(ctx: "RunContext") -> bool:
    """Whether the environment build has succeeded for this run.

    For ``docker``/``auto`` an explicit ``build_environment`` ok-row is required;
    for ``local``/``runpod``/``azure``/``gcp`` env-build is a no-op success → treat
    as built.  Mirror of ``run.py``'s closure of the same name.
    """
    mode = getattr(ctx.sandbox_mode, "value", str(ctx.sandbox_mode or "")).lower()
    if mode in ("docker", "auto"):
        try:
            return ctx.cost_ledger.session_call_count("build_environment") > 0
        except Exception:  # noqa: BLE001
            return False
    return True


def _run_experiment_count(ctx: "RunContext") -> int:
    """In-process ``run_experiment`` count (fail-soft → 0)."""
    ledger = getattr(ctx, "cost_ledger", None)
    if ledger is None:
        return 0
    try:
        counter = getattr(ledger, "session_call_count", None)
        if callable(counter):
            return int(counter("run_experiment"))
    except Exception:  # noqa: BLE001
        return 0
    return 0


def _remaining_budget(ctx: "RunContext") -> float | None:
    """Remaining wall-clock seconds (fail-soft → None)."""
    try:
        return ctx.remaining_s()
    except Exception:  # noqa: BLE001
        return None


def build_action_card(situation: str, ctx: "RunContext") -> dict[str, Any] | None:
    """Build a deterministic next-action card, or ``None`` when ambiguous.

    Returns a recommendation dict in the SAME shape ``recommend_next_tool``
    returns from its LLM path (``tool``/``reason``/``alternatives``/``outcome``)
    when the next lifecycle stage maps to a single unambiguous tool.  Returns
    ``None`` — signalling the caller to fall through to the LLM — when:

    * the inferred stage is ``can_finalize`` (a judgement call), or
    * the inferred stage is not one we map (defensive), or
    * anything goes wrong building the card.

    Pure: reads only on-disk/ledger run state + remaining budget; NEVER reads the
    paper corpus and NEVER calls the LLM.  Never raises (fail-soft).
    """
    try:
        stage = infer_required_stage(
            primitives=[],
            code_path_exists=_code_path_exists(ctx),
            env_built=_env_built(ctx),
            total_run_experiments=_run_experiment_count(ctx),
            total_verifications=(
                1 if getattr(ctx, "latest_rubric_score", None) is not None else 0
            ),
        )
    except Exception:  # noqa: BLE001 — never break the primitive; fall through to LLM
        return None

    if stage not in REQUIRED_STAGES:  # defensive — unknown stage → ambiguous
        return None

    tool = _STAGE_TO_TOOL.get(stage)
    if tool is None:
        # ``can_finalize`` (or any unmapped stage) is ambiguous — let the LLM decide.
        return None

    # Compact card: stage + last outcome + coverage gaps + remaining budget. These
    # are surfaced in ``reason`` so the deterministic answer is self-describing.
    last_outcome = getattr(ctx, "last_primitive_outcome", None)
    remaining_s = _remaining_budget(ctx)
    coverage_gaps = [s for s in _STAGE_TO_TOOL if s != stage]

    reason = (
        f"deterministic: lifecycle stage={stage} ({_STAGE_REASON.get(stage, '')}); "
        f"next tool is unambiguous"
    )

    return {
        "tool": tool,
        "reason": reason,
        "alternatives": [],
        "outcome": "ok",
        "deterministic": True,
        "card": {
            "stage": stage,
            "last_primitive_outcome": last_outcome,
            "coverage_gaps": coverage_gaps,
            "remaining_budget_s": remaining_s,
        },
    }
