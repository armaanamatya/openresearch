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
    ledger = [json.loads(ln) for ln in
              (ctx.project_dir / "cost_ledger.jsonl").read_text().splitlines() if ln]
    assert ledger[-1]["agent_id"] == "boom"  # ledger row appended on the error path too
