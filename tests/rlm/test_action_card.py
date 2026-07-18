"""Tests for the deterministic next-action card (feature #18).

Covers the ``OPENRESEARCH_ACTION_CARDS`` short-circuit in
``recommend_next_tool``:
1. flag OFF → LLM path used (the fake llm_client.complete WAS called);
2. flag ON + deterministic stage → card returned, LLM NOT called;
3. flag ON + ambiguous (can_finalize) → falls through to the LLM;
4. flag ON + card-build error → fail-soft to the LLM.

Hermetic: no network, no GPU, no real LLM (FakeLlmClient).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.rlm.primitives import recommend_next_tool


def _record_run_experiment(ctx) -> None:
    """Append an in-process ``run_experiment`` ledger row so the lifecycle
    ladder counts the experiment as having run (session-scoped count > 0)."""
    ctx.cost_ledger.append(
        CostLedgerEntry(
            timestamp=datetime.now(tz=timezone.utc),
            agent_id="run_experiment",
            attempt_index=0,
            provider="anthropic",
            model="test-model",
            outcome="ok",
        )
    )


def _seed_baseline(ctx) -> None:
    """Give the run a usable on-disk baseline so the lifecycle ladder advances
    past ``need_baseline`` (a non-empty commands.json + one runnable file)."""
    code_dir = ctx.project_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text(
        json.dumps(["python train.py"]), encoding="utf-8"
    )
    (code_dir / "train.py").write_text("print('hi')\n", encoding="utf-8")


def test_flag_off_uses_llm_path(make_context, tmp_path, monkeypatch):
    """Flag unset → deterministic short-circuit is dormant; the LLM is called."""
    monkeypatch.delenv("OPENRESEARCH_ACTION_CARDS", raising=False)
    ctx = make_context(tmp_path, llm_responses=['{"tool": "implement_baseline"}'])

    result = recommend_next_tool("what next?", ctx=ctx)

    assert ctx.llm_client.calls, "LLM must be called when the flag is OFF"
    assert result["tool"] == "implement_baseline"
    assert "deterministic" not in result


def test_flag_on_deterministic_stage_returns_card(make_context, tmp_path, monkeypatch):
    """Flag ON + unambiguous stage (no baseline → implement_baseline) → card,
    LLM NOT called."""
    monkeypatch.setenv("OPENRESEARCH_ACTION_CARDS", "1")
    # Scripted response would be a bug to use — assert it is NOT consumed.
    ctx = make_context(tmp_path, llm_responses=['{"tool": "WRONG"}'])

    result = recommend_next_tool("what next?", ctx=ctx)

    assert ctx.llm_client.calls == [], "LLM must NOT be called on a deterministic stage"
    assert result["tool"] == "implement_baseline"
    assert result.get("deterministic") is True
    assert result["card"]["stage"] == "need_baseline"
    assert result["outcome"] == "ok"


def test_flag_on_ambiguous_falls_through_to_llm(make_context, tmp_path, monkeypatch):
    """Flag ON but the run is scored (can_finalize) → ambiguous → LLM decides."""
    monkeypatch.setenv("OPENRESEARCH_ACTION_CARDS", "1")
    ctx = make_context(tmp_path, llm_responses=['{"tool": "propose_improvements"}'])
    # Advance the lifecycle all the way to can_finalize:
    _seed_baseline(ctx)  # past need_baseline
    # local sandbox → env build is a no-op success (past need_environment)
    ctx.sandbox_mode = "local"
    _record_run_experiment(ctx)  # past need_experiment
    ctx.latest_rubric_score = 0.8  # past need_verification → can_finalize

    result = recommend_next_tool("what next?", ctx=ctx)

    assert ctx.llm_client.calls, "ambiguous can_finalize must fall through to the LLM"
    assert result["tool"] == "propose_improvements"
    assert "deterministic" not in result


def test_flag_on_card_build_error_failsoft_to_llm(make_context, tmp_path, monkeypatch):
    """Flag ON but card-building raises → fail-soft to the LLM path."""
    monkeypatch.setenv("OPENRESEARCH_ACTION_CARDS", "1")
    ctx = make_context(tmp_path, llm_responses=['{"tool": "implement_baseline"}'])

    # Force build_action_card to blow up by making infer_required_stage explode.
    import backend.agents.rlm.action_card as ac

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic card-build failure")

    monkeypatch.setattr(ac, "_code_path_exists", _boom)

    result = recommend_next_tool("what next?", ctx=ctx)

    assert ctx.llm_client.calls, "a card-build error must fail-soft to the LLM"
    assert result["tool"] == "implement_baseline"
    assert "deterministic" not in result


def test_build_action_card_never_raises(make_context, tmp_path):
    """build_action_card is fail-soft even on a totally degenerate ctx."""
    from backend.agents.rlm.action_card import build_action_card

    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("everything is broken")

    assert build_action_card("x", _Broken()) is None
