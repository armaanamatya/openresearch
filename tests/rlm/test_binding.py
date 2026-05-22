import json

import pytest

from backend.agents.rlm.binding import build_custom_tools


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
