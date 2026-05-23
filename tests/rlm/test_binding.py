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

# Rubric-verifier LLM response shape — leaf-id-keyed array, as the post-C2
# (2e1ce37 + fix/leaf-scorer-honesty) verify_against_rubric expects. The
# previous area-keyed shape was for the pre-2e1ce37 area-based verifier;
# verify_against_rubric now goes through `score_reproduction` (leaf scorer).
_RUBRIC_SCORES = json.dumps([
    {"leaf_id": "code", "score": 0.9, "justification": "code matches paper"},
    {"leaf_id": "results", "score": 0.8, "justification": "results within 5%"},
])

# Leaf-style rubric tree (PaperBench bundle shape) — what score_reproduction
# flattens via `flatten_leaves`. Used by the verify_against_rubric tests.
_LEAF_RUBRIC = {
    "id": "root",
    "requirements": "reproduce the paper",
    "weight": 1.0,
    "source": "generated",
    "target_score": 0.7,
    "sub_tasks": [
        {"id": "code", "requirements": "code matches", "weight": 0.6, "sub_tasks": []},
        {"id": "results", "requirements": "results match", "weight": 0.4, "sub_tasks": []},
    ],
}


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
    ctx = make_context(tmp_path, llm_responses=[_RUBRIC_SCORES])
    tools = build_custom_tools(ctx)
    tools["verify_against_rubric"]["tool"](
        results={"success": True, "metrics": {"acc": 0.9}}, rubric=_LEAF_RUBRIC
    )
    assert any(e.get("event") == "rubric_score" for e in _read_events(ctx))


def test_verify_against_rubric_emits_nothing_on_failure(make_context, tmp_path):
    """wrap_primitive must NOT emit rubric_score when verification fails.

    Two paths that should yield no event:
      - LLM grader output is unparseable on every batch → the C2 honesty guard
        in primitives.verify_against_rubric returns {"success": False, ...}
        (graded == 0 over a non-empty rubric is not a "scored 0.0" success;
        nothing should be shown to the UI).
      - Pre-existing fail-soft path on a malformed rubric / exception.
    """
    # Force the C2 honesty guard: well-formed rubric, garbage LLM output → 0 graded.
    ctx = make_context(tmp_path, llm_responses=["not json at all"])
    tools = build_custom_tools(ctx)
    tools["verify_against_rubric"]["tool"](results={}, rubric=_LEAF_RUBRIC)
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


# ---------------------------------------------------------------------------
# wrap_primitive auto-coercion (Fix B)
# ---------------------------------------------------------------------------


def test_wrap_primitive_coerces_int_to_str_when_str_expected(make_context, tmp_path):
    """wrap_primitive converts int arg to str when the primitive expects str; event shows coerced=True."""
    # understand_section expects text_slice: str
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    # Pass an int where str is expected — wrap_primitive should coerce int → str
    # and the primitive should succeed (or at least not raise a TypeError).
    result = tools["understand_section"]["tool"](42)
    # The primitive ran (returned a dict) — coercion succeeded
    assert isinstance(result, dict)
    # The completion event carries coerced=True
    events = _read_events(ctx)
    completion = [e for e in events if e.get("event") == "primitive_call" and e.get("status") in ("ok", "error")]
    assert completion, "no completion primitive_call event emitted"
    assert completion[0].get("coerced") is True, (
        f"expected coerced=True on completion event, got: {completion[0]}"
    )


def test_wrap_primitive_does_not_coerce_dict_to_str(make_context, tmp_path):
    """wrap_primitive does NOT coerce dict → str; the primitive receives the dict as-is."""
    # understand_section expects text_slice: str; passing a dict should NOT be coerced
    # (the spec says: if a dict is missing required keys — bail, don't try to guess).
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    # A dict is not a scalar — _coerce_args skips it because str(dict) would produce
    # Python repr that is semantically wrong as a paper text slice.
    # Instead the primitive should receive the raw dict and either handle it fail-soft
    # or raise. Either way coerced must be False.
    try:
        tools["understand_section"]["tool"]({"key": "value"})
    except Exception:
        pass  # raising is acceptable; we just need coerced=False
    events = _read_events(ctx)
    completion = [e for e in events if e.get("event") == "primitive_call" and e.get("status") in ("ok", "error")]
    assert completion, "no completion primitive_call event emitted"
    assert completion[0].get("coerced") is False, (
        f"coerced must be False when dict passed where str expected, got: {completion[0]}"
    )
