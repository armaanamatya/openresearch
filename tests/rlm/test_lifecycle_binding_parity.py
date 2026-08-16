"""Parity test: lifecycle-driven, binding-wrapped verify emits rubric_score
with iteration >= 1.

Task 4 of the lifecycle-primary hardening plan.

WHY: Tasks 1-3 tested the driver against raw mock tools that bypass
``binding.wrap_primitive``.  This test proves the KEYSTONE end-to-end:
when the lifecycle driver runs a ``binding.wrap_primitive``-wrapped
``verify_against_rubric``, the emitted ``rubric_score`` event carries
``iteration >= 1``.  Without Task 1's counter-advance in ``_step``, the
event would carry iteration=0.

Context construction follows the ``make_context`` pattern from
``tests/rlm/conftest.py`` exactly (same fields, same factory).  We do NOT
use the conftest fixture here so the test is self-contained and readable
in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path


from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm import binding
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.lifecycle_driver import drive_lifecycle_chain
from backend.agents.rlm.sse_bridge import make_emit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLlmClient:
    """Minimal LlmClient stub — primitives under test do not call the LLM."""

    def complete(self, *, system: str, user: str) -> str:  # noqa: ARG002
        return "{}"


def _build_ctx(project_dir: Path, runs_root: Path) -> RunContext:
    """Build a RunContext in the same way conftest.make_context does."""
    project_dir.mkdir(parents=True, exist_ok=True)
    dashboard = DashboardEmitter("test_proj", runs_root)
    return RunContext(
        project_id="test_proj",
        project_dir=project_dir,
        runs_root=runs_root,
        dashboard=dashboard,
        emit=make_emit(dashboard),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl",
            project_id="test_proj",
            attach_path=True,
        ),
        llm_client=_FakeLlmClient(),
        provider="anthropic",
        model="test-model",
    )


def _read_events(project_dir: Path) -> list[dict]:
    """Read all dashboard events from the JSONL file."""
    path = project_dir / "dashboard_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln]


# ---------------------------------------------------------------------------
# The keystone parity test
# ---------------------------------------------------------------------------


def test_lifecycle_driver_binding_parity_rubric_iteration(tmp_path: Path) -> None:
    """A lifecycle-driven, binding-wrapped verify_against_rubric emits a
    rubric_score event with iteration >= 1.

    Flow:
    1. Build a real RunContext with current_iteration=0.
    2. Wrap a fake verify tool via binding.wrap_primitive.
    3. Drive the lifecycle chain at start_stage="need_verification" so only
       the verify step runs (one _step call → counter advance → iteration=1).
    4. Collect rubric_score events from ctx.emit's output (dashboard_events.jsonl).
    5. Assert the event carries iteration >= 1 (NOT 0).
    """
    project_dir = tmp_path / "test_proj"
    ctx = _build_ctx(project_dir, tmp_path)

    # Confirm the context starts at iteration 0
    assert ctx.current_iteration == 0

    # fake verify returns a fully scored result with areas so binding's
    # rubric-emit branch fires (it requires: score is not None AND
    # isinstance(areas, list)).
    def _fake_verify(*args, **kwargs):  # noqa: ANN001,ANN002,ANN003
        return {
            "overall_score": 0.9,
            "target_score": 0.7,
            "meets_target": True,
            "areas": [],         # empty list is fine; branch checks isinstance(areas, list)
        }

    # Wrap via binding.wrap_primitive — this is the REAL production wrapper,
    # not a raw mock.  Events go to ctx.emit → dashboard_events.jsonl.
    wrapped = binding.wrap_primitive("verify_against_rubric", _fake_verify, ctx)
    tools = {"verify_against_rubric": {"tool": wrapped, "description": "verify"}}

    # Drive only the verify step.
    summary = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text="paper",
        rubric_spec={"target_score": 0.7},
        start_stage="need_verification",
        emit=lambda e: None,   # driver's own emit sink; rubric events go via ctx.emit
    )

    # The driver should have run verify_against_rubric.
    assert "verify_against_rubric" in summary.get("driven", []), (
        f"verify_against_rubric was not driven. summary={summary}"
    )

    # Events are written to dashboard_events.jsonl via ctx.emit (make_emit →
    # DashboardEmitter._emit).  Read them back.
    events = _read_events(project_dir)
    rubric_events = [e for e in events if e.get("event") == "rubric_score"]

    assert rubric_events, (
        f"No rubric_score event found in dashboard_events.jsonl. "
        f"All events: {[e.get('event') for e in events]}"
    )

    # KEYSTONE: the iteration field must be >= 1.  Before Task 1, _step did NOT
    # advance ctx.current_iteration before calling the tool, so binding would
    # read ctx.current_iteration=0 and the event would carry iteration=0.
    first = rubric_events[0]
    assert first.get("iteration", 0) >= 1, (
        f"rubric_score event carries iteration={first.get('iteration')!r} — "
        f"expected >= 1.  This means the lifecycle driver is NOT advancing "
        f"ctx.current_iteration before calling binding-wrapped verify "
        f"(Task 1 regression). Full event: {first}"
    )

    # Also verify that ctx.latest_rubric_iteration was updated by binding.
    assert getattr(ctx, "latest_rubric_iteration", 0) >= 1, (
        f"ctx.latest_rubric_iteration={ctx.latest_rubric_iteration!r} — "
        f"expected >= 1 after a successful verify_against_rubric"
    )
