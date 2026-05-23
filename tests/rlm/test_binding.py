import json

import pytest

from backend.agents.rlm.binding import build_custom_tools

# ---------------------------------------------------------------------------
# Helpers shared across the event-emission tests
# ---------------------------------------------------------------------------

# Minimal valid ImprovementHypothesis JSON that propose_improvements will parse.
_HYP_WITH_TITLE = json.dumps({"hypotheses": [
    {"path_id": "p1", "hypothesis": "Tune the learning rate to 1e-4.",
     "rationale": "LR is highest-leverage.", "expected_outcome": "Higher reward.",
     "category": "optimizer", "title": "LR Tuning"},
    {"path_id": "p2", "hypothesis": "Swap backbone to ResNet-50.",
     "rationale": "Wider network fits better.", "expected_outcome": "Lower loss.",
     "category": "architecture", "title": "ResNet Backbone"},
]})

# Rubric-verifier LLM response for a passing run.
_RUBRIC_SCORES = json.dumps({"areas": [
    {"area": "code", "score": 0.9, "weak_points": []},
    {"area": "results", "score": 0.8, "weak_points": []},
], "confidence": 0.8})


def test_wrapped_primitive_emits_event_and_ledger_row(make_context, tmp_path):
    ctx = make_context(tmp_path)
    registry = {"echo": lambda value, *, ctx: {"echoed": value}}
    tools = build_custom_tools(ctx, registry=registry, descriptions={"echo": "echo a value"})

    assert set(tools["echo"]) == {"tool", "description"}
    result = tools["echo"]["tool"]("hi")
    assert result == {"echoed": "hi"}

    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    pe = [e for e in events if e.get("event") == "primitive_call"]
    assert [e["status"] for e in pe] == ["start", "ok"]
    assert pe[0]["primitive"] == "echo"

    ledger = [json.loads(ln) for ln in
              (ctx.project_dir / "cost_ledger.jsonl").read_text().splitlines() if ln]
    assert ledger[-1]["agent_id"] == "echo"


def test_wrapped_primitive_records_failsoft_failure(make_context, tmp_path, caplog):
    # Most primitives are fail-soft — on failure they RETURN a failure-shaped
    # dict instead of raising. wrap_primitive must mark that as an `error`
    # primitive_call and log a WARNING, not silently record it as a success.
    import logging

    ctx = make_context(tmp_path)

    def failsoft(*, ctx):
        return {"success": False, "error": "verify failed: truncated JSON"}

    tools = build_custom_tools(
        ctx, registry={"failsoft": failsoft}, descriptions={"failsoft": "fails soft"}
    )
    with caplog.at_level(logging.WARNING, logger="backend.agents.rlm.binding"):
        result = tools["failsoft"]["tool"]()

    assert result == {"success": False, "error": "verify failed: truncated JSON"}
    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    statuses = [e["status"] for e in events if e.get("event") == "primitive_call"]
    assert statuses == ["start", "error"]
    assert any("failsoft" in r.message for r in caplog.records)


def test_wrapped_primitive_records_failure(make_context, tmp_path):
    ctx = make_context(tmp_path)

    def boom(*, ctx):
        raise ValueError("bad")

    tools = build_custom_tools(ctx, registry={"boom": boom}, descriptions={"boom": "fails"})
    with pytest.raises(ValueError):
        tools["boom"]["tool"]()
    events = [json.loads(ln) for ln in
              (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines() if ln]
    statuses = [e["status"] for e in events if e.get("event") == "primitive_call"]
    assert statuses == ["start", "error"]
    # M1: the error summary is value-free — the exception MESSAGE ("bad") is
    # never leaked into the dashboard event, only the exception type.
    error_event = next(e for e in events if e.get("status") == "error")
    assert error_event["result_summary"] == "ValueError"
    assert "bad" not in str(error_event["result_summary"])
    ledger = [json.loads(ln) for ln in
              (ctx.project_dir / "cost_ledger.jsonl").read_text().splitlines() if ln]
    assert ledger[-1]["agent_id"] == "boom"  # ledger row appended on the error path too


# ---------------------------------------------------------------------------
# Phase 6 Task 13 — event emission tests
# ---------------------------------------------------------------------------


def _read_events(ctx) -> list[dict]:
    path = ctx.project_dir / "dashboard_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln]


def test_propose_improvements_emits_candidate_proposed_per_item(make_context, tmp_path):
    """wrap_primitive emits one candidate_proposed per returned hypothesis."""
    ctx = make_context(tmp_path, llm_responses=[_HYP_WITH_TITLE])
    tools = build_custom_tools(ctx)
    result = tools["propose_improvements"]["tool"](
        current_results={}, rubric_scores={}
    )
    proposed = [e for e in _read_events(ctx) if e.get("event") == "candidate_proposed"]
    assert len(proposed) == len(result)
    first = proposed[0]["candidate"]
    assert set(first) == {"id", "title", "category", "description", "reasoning"}
    # propose_round must have been incremented
    assert ctx.propose_round == 1


def test_verify_against_rubric_emits_rubric_score_on_success(make_context, tmp_path):
    """wrap_primitive emits rubric_score after a successful verify_against_rubric."""
    rubric = {
        "areas": [{"area": "code", "weight": 0.6}, {"area": "results", "weight": 0.4}],
        "source": "generated",
        "target_score": 0.7,
    }
    ctx = make_context(tmp_path, llm_responses=[_RUBRIC_SCORES])
    tools = build_custom_tools(ctx)
    tools["verify_against_rubric"]["tool"](
        results={"success": True, "metrics": {"acc": 0.9}}, rubric=rubric
    )
    assert any(e.get("event") == "rubric_score" for e in _read_events(ctx))


def test_verify_against_rubric_emits_nothing_on_failure(make_context, tmp_path):
    """wrap_primitive must NOT emit rubric_score when verification fails (success=False)."""
    rubric = {"areas": [], "source": "generated", "target_score": 0.7}
    # Force a parse failure so verify_against_rubric returns {success: False}
    ctx = make_context(tmp_path, llm_responses=["not json at all"])
    tools = build_custom_tools(ctx)
    tools["verify_against_rubric"]["tool"](results={}, rubric=rubric)
    assert not any(e.get("event") == "rubric_score" for e in _read_events(ctx))


def test_record_candidate_outcome_emits_candidate_outcome(make_context, tmp_path):
    """record_candidate_outcome wraps to emit a candidate_outcome event."""
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    tools["record_candidate_outcome"]["tool"](
        candidate_id="c1", outcome="promoted", parent_id="baseline"
    )
    out = [e for e in _read_events(ctx) if e.get("event") == "candidate_outcome"]
    assert len(out) == 1
    assert out[0]["candidate_id"] == "c1"
    assert out[0]["outcome"] == "promoted"
