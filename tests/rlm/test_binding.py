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


def test_wrapped_primitive_summarizes_validation_error_without_values(make_context, tmp_path):
    from pydantic import BaseModel, ValidationError

    class Payload(BaseModel):
        count: int

    ctx = make_context(tmp_path)

    def validate(*, ctx):
        Payload.model_validate({"count": "not-an-int"})

    tools = build_custom_tools(ctx, registry={"validate": validate}, descriptions={"validate": "validates"})
    with pytest.raises(ValidationError):
        tools["validate"]["tool"]()

    events = _read_events(ctx)
    error_event = next(e for e in events if e.get("event") == "primitive_call" and e.get("status") == "error")
    summary = error_event["result_summary"]
    assert summary.startswith("ValidationError:")
    assert "count" in summary
    assert "int_parsing" in summary
    assert "not-an-int" not in summary


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
    # display_title is now included when available; allow it alongside the required keys.
    assert {"id", "title", "category", "description", "reasoning"}.issubset(set(first))
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


# ---------------------------------------------------------------------------
# _meta.hint → result_summary prefix tests (Option B in-band hint)
# ---------------------------------------------------------------------------


def test_result_summary_has_hint_prefix_when_meta_hint_present(make_context, tmp_path):
    """result_summary starts with '[hint] ' when the primitive result has _meta.hint."""
    from backend.agents.rlm.primitives import _HINT_THRESHOLD

    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    # Pass a slice larger than the threshold so understand_section injects _meta.
    large_slice = "z" * (_HINT_THRESHOLD + 1)
    tools["understand_section"]["tool"](large_slice)
    events = _read_events(ctx)
    completion = [
        e for e in events
        if e.get("event") == "primitive_call" and e.get("status") in ("ok", "error")
    ]
    assert completion, "no completion event found"
    summary = completion[0].get("result_summary", "")
    assert summary.startswith("[hint] "), (
        f"expected result_summary to start with '[hint] ' for a large slice, got: {summary!r}"
    )


def test_result_summary_no_hint_prefix_for_short_slice(make_context, tmp_path):
    """result_summary must NOT start with '[hint] ' for a short slice."""
    from backend.agents.rlm.primitives import _HINT_THRESHOLD

    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    short_slice = "z" * (_HINT_THRESHOLD - 1)
    tools["understand_section"]["tool"](short_slice)
    events = _read_events(ctx)
    completion = [
        e for e in events
        if e.get("event") == "primitive_call" and e.get("status") in ("ok", "error")
    ]
    assert completion, "no completion event found"
    summary = completion[0].get("result_summary", "")
    assert not summary.startswith("[hint] "), (
        f"result_summary must not carry [hint] for a short slice, got: {summary!r}"
    )


# ---------------------------------------------------------------------------
# 2026-05-23: candidate_id validation — pins the fix for the
# prj_6b9acbfd8afcd789 'candidate_id="None"' SSE-corruption bug.
# ---------------------------------------------------------------------------


def test_record_candidate_outcome_with_none_candidate_id_skips_emit(make_context, tmp_path):
    """Passing None as candidate_id must NOT produce a candidate_outcome
    event with candidate_id='None' (the str(None) corruption that broke the
    UI's candidate→outcome matching)."""
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    result = tools["record_candidate_outcome"]["tool"](
        candidate_id=None, outcome="declined"
    )
    # Primitive returns success=False with a clear error
    assert result.get("success") is False
    assert "candidate_id" in (result.get("error") or "")
    # No candidate_outcome event was emitted (binding skipped it)
    out = [e for e in _read_events(ctx) if e.get("event") == "candidate_outcome"]
    assert out == [], f"unexpected candidate_outcome event with bad input: {out}"


def test_record_candidate_outcome_with_literal_None_string_skips_emit(make_context, tmp_path):
    """The 2026-05-23 wire bug: candidate_id was the string 'None'. Pin that
    we now reject it too (defense in depth — the upstream str(None) ever
    re-appearing must not silently propagate)."""
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    for bad in ["None", "null", "", "   "]:
        result = tools["record_candidate_outcome"]["tool"](
            candidate_id=bad, outcome="declined"
        )
        assert result.get("success") is False, f"expected failure for bad cid={bad!r}"
    out = [e for e in _read_events(ctx) if e.get("event") == "candidate_outcome"]
    assert out == [], f"unexpected candidate_outcome events: {out}"


def test_record_candidate_outcome_canonicalizes_outcome_aliases(make_context, tmp_path):
    """2026-05-23 regression fix: previously the validator strictly rejected
    any outcome not in the literal 6-value set. C5 of the paper sweep had
    4 record_candidate_outcome calls with outcomes like 'success' / 'fail'
    that the model meant to be promotions or failures — all 4 were rejected,
    zero outcome events were emitted, the user's 'promoted candidate' gate
    became unreachable. The fix: canonicalize via case-insensitive alias map
    so the model can speak natural English and we still emit a valid event.
    """
    cases = [
        ("success",   "promoted"),   # most common alias
        ("PROMOTED",  "promoted"),   # case-insensitive
        ("Pass",      "promoted"),
        ("ok",        "promoted"),
        ("improved",  "promoted"),
        ("fail",      "failed"),
        ("error",     "failed"),
        ("regressed", "failed"),
        ("partial",   "marginal"),
        ("mixed",     "marginal"),
        ("skip",      "skipped"),
        ("decline",   "declined"),
        ("rejected",  "declined"),
        ("run",       "running"),
        ("in_progress", "running"),
        ("in progress", "running"),  # space becomes underscore
        ("In-Progress", "running"),   # hyphen becomes underscore
    ]
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    for raw, canonical in cases:
        result = tools["record_candidate_outcome"]["tool"](
            candidate_id=f"path_{raw.replace(' ', '_')}",
            outcome=raw,
        )
        assert result.get("success") is True, (
            f"expected success on alias outcome={raw!r}, got {result!r}"
        )
        assert result.get("outcome") == canonical, (
            f"alias {raw!r} should canonicalize to {canonical!r}, got {result.get('outcome')!r}"
        )
    out = [e for e in _read_events(ctx) if e.get("event") == "candidate_outcome"]
    assert len(out) == len(cases), (
        f"expected {len(cases)} candidate_outcome events, got {len(out)}"
    )


def test_record_candidate_outcome_unknown_outcome_falls_back_to_literal_not_reject(make_context, tmp_path):
    """Truly unknown outcomes are accepted-with-literal-value, NOT rejected.
    The UI can render 'unknown' / arbitrary strings as gray; dropping the
    event entirely is worse than showing an unfamiliar label."""
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    result = tools["record_candidate_outcome"]["tool"](
        candidate_id="path_99", outcome="totally-novel-outcome"
    )
    assert result.get("success") is True, (
        "unknown outcomes must NOT be rejected — the structural info is still useful"
    )
    # The literal value is preserved so the UI can render it as-is
    assert result.get("outcome") == "totally-novel-outcome"
    out = [e for e in _read_events(ctx) if e.get("event") == "candidate_outcome"]
    assert len(out) == 1


def test_record_candidate_outcome_empty_outcome_falls_back_to_unknown(make_context, tmp_path):
    """Empty / None outcomes get the literal string 'unknown' (and still emit)
    so the candidate appears in the UI tree even with no labeled outcome."""
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    for empty in ["", "   ", None]:
        result = tools["record_candidate_outcome"]["tool"](
            candidate_id=f"path_{id(empty)}", outcome=empty
        )
        assert result.get("success") is True
        assert result.get("outcome") == "unknown"


def test_record_candidate_outcome_happy_path_still_emits(make_context, tmp_path):
    """The good-input path is unchanged — emits a candidate_outcome with the
    real candidate_id and outcome string."""
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    result = tools["record_candidate_outcome"]["tool"](
        candidate_id="path_2", outcome="promoted", parent_id="baseline"
    )
    assert result.get("success") is True
    out = [e for e in _read_events(ctx) if e.get("event") == "candidate_outcome"]
    assert len(out) == 1
    assert out[0]["candidate_id"] == "path_2"
    assert out[0]["outcome"] == "promoted"


# ---------------------------------------------------------------------------
# _friendly_candidate_title — display title compression
# ---------------------------------------------------------------------------

from backend.agents.rlm.binding import _friendly_candidate_title


def test_friendly_candidate_title_short_passes_through():
    """A title that is ≤5 words AND ≤40 chars is returned unchanged."""
    short = "LR Tuning"
    assert _friendly_candidate_title(short) == "LR Tuning"


def test_friendly_candidate_title_long_is_compressed():
    """A long title should be condensed to ≤5 words."""
    long_title = (
        "Adaptive learning rate schedule with cosine annealing and warm restarts "
        "to improve convergence stability over 200 epochs"
    )
    result = _friendly_candidate_title(long_title)
    # Must be shorter than the original
    assert len(result) < len(long_title)
    # Should be ≤5 words (split on whitespace)
    assert len(result.split()) <= 5


def test_friendly_candidate_title_colon_split():
    """Titles with a colon are split and the prefix is returned."""
    title = "Architecture: Replace MLP with Transformer block"
    result = _friendly_candidate_title(title)
    assert result == "Architecture"


def test_friendly_candidate_title_em_dash_split():
    """Titles with an em-dash are split and the prefix is returned."""
    title = "Optimizer—Switch from SGD to AdamW for better convergence"
    result = _friendly_candidate_title(title)
    assert result == "Optimizer"


def test_friendly_candidate_title_five_word_boundary():
    """Exactly 5 words and ≤40 chars → pass through unchanged."""
    title = "Tune LR with cosine decay"  # 5 words, short
    assert len(title.split()) == 5
    assert len(title) <= 40
    assert _friendly_candidate_title(title) == title
