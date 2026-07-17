"""Tests for the Foundry-routed root cost-ledger drain (money-safety fix).

Background: `cost_ledger.jsonl` / `tokens_total.json` were blind to
Foundry-routed root LLM spend (`opus-foundry`, `sonnet-foundry`,
`azure-foundry`/`grok`) because `run_pipeline_rlm`'s cost-drain block only
ever matched `root_model.rlm_backend == "anthropic-oauth"`. A campaign spent
$24.23 against a $19 cap while the ledger reported $0. This file proves:

1. `opus-foundry` root usage drains into the ledger, priced.
2. `sonnet-foundry` root usage drains into the ledger, priced.
3. Drained tokens surface in `tokens_total.json` (via `_aggregate_tokens_total`).
4. `final_report.cost.llm_usd` does NOT double-count the drained row (it is
   the row's SOLE contributor, since rlm's usage_summary.total_cost is never
   populated for a non-OpenRouter host).
5. An `anthropic-oauth` root is unaffected (both new helpers no-op for it) —
   no regression on the existing $0-is-correct OAuth behaviour.
6. A Foundry-routed model with no known price still gets its tokens ledgered,
   but fires a loud `unpriced_root_tokens` warning instead of a silent $0.
7. A root backend with no known drain path at all (anything other than
   anthropic-oauth / cred_provider=="azure-foundry") fires a loud
   `root_cost_unmetered` warning rather than staying silent.
"""

from __future__ import annotations

from types import SimpleNamespace

from rlm.core.types import ModelUsageSummary, UsageSummary

from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.models import ROOT_MODELS, RootModel
from backend.agents.rlm.report import _aggregate_tokens_total, _cost_dict
from backend.agents.rlm.run import (
    _drain_foundry_root_usage_to_ledger,
    _warn_unmetered_root_backend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(project_dir, *, with_path: bool = True) -> RunContext:
    project_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunCostLedger(
        project_id="prj_test",
        path=(project_dir / "cost_ledger.jsonl") if with_path else None,
    )
    return RunContext(
        project_id="prj_test",
        project_dir=project_dir,
        runs_root=project_dir.parent,
        dashboard=None,
        cost_ledger=ledger,
        llm_client=None,
        provider="anthropic",
        model="test-model",
    )


def _usage_result(models: dict[str, tuple[int, int]]):
    """Build a fake ``RLMChatCompletion``-shaped object with the given usage.

    ``models`` maps model name -> (input_tokens, output_tokens); total_calls
    is always 1 per model (sufficient for these tests).
    """
    summaries = {
        name: ModelUsageSummary(
            total_calls=1,
            total_input_tokens=inp,
            total_output_tokens=out,
            # rlm's AnthropicClient/OpenAIClient never populate total_cost for
            # a non-OpenRouter host — this is the crux of the no-double-count
            # invariant under test.
            total_cost=None,
        )
        for name, (inp, out) in models.items()
    }
    return SimpleNamespace(usage_summary=UsageSummary(model_usage_summaries=summaries))


def _make_emit():
    captured: list[dict] = []

    def emit(event: dict) -> None:
        captured.append(event)

    return emit, captured


# ---------------------------------------------------------------------------
# 1/2 — opus-foundry / sonnet-foundry drain into the ledger, priced
# ---------------------------------------------------------------------------


def test_opus_foundry_drains_usage_into_ledger(tmp_path):
    root_model = ROOT_MODELS["opus-foundry"]
    assert root_model.rlm_backend == "anthropic-foundry"
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"claude-opus-4-8": (100_000, 20_000)})
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    assert len(ctx.cost_ledger.entries) == 1
    entry = ctx.cost_ledger.entries[0]
    assert entry.agent_id == "rlm_root"
    assert entry.model == "claude-opus-4-8"
    assert entry.provider == "anthropic"
    assert entry.input_tokens == 100_000
    assert entry.output_tokens == 20_000
    # 100k*15/1e6 + 20k*75/1e6 = 1.5 + 1.5 = 3.0
    assert entry.estimated_usd == 3.0
    # A known rate — no "unpriced" warning should fire.
    assert not any(e.get("code") == "unpriced_root_tokens" for e in captured)


def test_sonnet_foundry_drains_usage_into_ledger(tmp_path):
    root_model = ROOT_MODELS["sonnet-foundry"]
    assert root_model.rlm_backend == "anthropic-foundry"
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"claude-sonnet-5": (50_000, 10_000)})
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    assert len(ctx.cost_ledger.entries) == 1
    entry = ctx.cost_ledger.entries[0]
    assert entry.model == "claude-sonnet-5"
    # 50k*3/1e6 + 10k*15/1e6 = 0.15 + 0.15 = 0.3
    assert entry.estimated_usd == 0.3
    assert not captured


def test_azure_foundry_grok_key_also_drains_when_priced(tmp_path):
    """The openai-transport azure-foundry (grok) root shares the same
    cred_provider family and must drain too — priced here via a model name
    that happens to have a rate, to isolate this from the unpriced case."""
    root_model = ROOT_MODELS["azure-foundry"]
    assert root_model.rlm_backend == "openai"
    assert root_model.cred_provider == "azure-foundry"
    ctx = _make_ctx(tmp_path / "run1")
    # gpt-4o has no reason to be deployed behind azure-foundry in practice, but
    # it exercises "a priced model reachable via the openai-compatible branch"
    # without inventing a new pricing entry for this test.
    result_obj = _usage_result({"gpt-4o": (1_000_000, 1_000_000)})
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    assert len(ctx.cost_ledger.entries) == 1
    entry = ctx.cost_ledger.entries[0]
    assert entry.provider == "openai"
    assert entry.model == "gpt-4o"
    assert entry.estimated_usd == 2.5 + 10.0  # 1M in @2.5/1M + 1M out @10/1M
    assert not captured


def test_zero_calls_are_skipped(tmp_path):
    """Mirrors the OAuth block's own `if _u.get("calls", 0) == 0: continue`."""
    root_model = ROOT_MODELS["opus-foundry"]
    ctx = _make_ctx(tmp_path / "run1")
    summaries = {
        "claude-opus-4-8": ModelUsageSummary(
            total_calls=0, total_input_tokens=0, total_output_tokens=0,
        )
    }
    result_obj = SimpleNamespace(
        usage_summary=UsageSummary(model_usage_summaries=summaries)
    )
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    assert ctx.cost_ledger.entries == []
    assert not captured


# ---------------------------------------------------------------------------
# 3 — drained tokens appear in tokens_total.json (via _aggregate_tokens_total)
# ---------------------------------------------------------------------------


def test_drained_tokens_appear_in_tokens_total(tmp_path):
    root_model = ROOT_MODELS["opus-foundry"]
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"claude-opus-4-8": (100_000, 20_000)})
    emit, _ = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    totals = _aggregate_tokens_total(ctx.project_dir)
    assert totals["grand_total"]["input_tokens"] == 100_000
    assert totals["grand_total"]["output_tokens"] == 20_000
    assert totals["by_model"]["claude-opus-4-8"]["input_tokens"] == 100_000
    assert totals["by_primitive"]["rlm_root"]["input_tokens"] == 100_000


# ---------------------------------------------------------------------------
# 4 — final_report.cost.llm_usd does not double-count
# ---------------------------------------------------------------------------


def test_final_report_cost_does_not_double_count(tmp_path):
    root_model = ROOT_MODELS["opus-foundry"]
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"claude-opus-4-8": (100_000, 20_000)})
    emit, _ = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    # usage_summary.total_cost is None for every model (as rlm's real clients
    # actually leave it for a non-OpenRouter host) -- so rlm_usd contributes 0
    # and the ledger row is the ONLY source of llm_usd.
    cost = _cost_dict(result_obj, ctx)
    assert cost["llm_usd"] == ctx.cost_ledger.total_usd()
    assert cost["llm_usd"] == 3.0  # real, nonzero -- no longer a silent $0
    assert cost["primitives"] == 3.0


# ---------------------------------------------------------------------------
# 5 — an anthropic-oauth root is unaffected (no regression)
# ---------------------------------------------------------------------------


def test_oauth_root_unaffected_by_new_helpers(tmp_path):
    root_model = ROOT_MODELS["claude-oauth"]
    assert root_model.rlm_backend == "anthropic-oauth"
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"claude-sonnet-4-6": (100_000, 20_000)})
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )
    _warn_unmetered_root_backend(root_model=root_model, result_obj=result_obj, emit=emit)

    # Both new helpers are no-ops for anthropic-oauth: that path is handled
    # exclusively by the pre-existing (untouched) drain_root_usage() branch,
    # where estimated_usd staying $0 is the CORRECT, documented behaviour.
    assert ctx.cost_ledger.entries == []
    assert captured == []


# ---------------------------------------------------------------------------
# 6 — unpriced Foundry model: tokens ledgered, loud warning, never silent $0
# ---------------------------------------------------------------------------


def test_unpriced_foundry_model_warns_loudly_not_silent_zero(tmp_path):
    # A synthetic azure-foundry (grok-family) root deployed to an arbitrary
    # model name with no MODEL_PRICING / resilience-pricing entry.
    root_model = RootModel(
        key="azure-foundry",
        rlm_backend="openai",
        backend_kwargs={"model_name": "some-unpriced-grok-deployment"},
    )
    assert root_model.cred_provider == "azure-foundry"
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"some-unpriced-grok-deployment": (40_000, 8_000)})
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )

    # Tokens are still ledgered -- never dropped.
    assert len(ctx.cost_ledger.entries) == 1
    entry = ctx.cost_ledger.entries[0]
    assert entry.model == "some-unpriced-grok-deployment"
    assert entry.input_tokens == 40_000
    assert entry.output_tokens == 8_000
    assert entry.estimated_usd is None  # genuinely unknown, not a computed 0.0
    assert entry.to_json()["cost_usd"] == 0.0  # serializes as 0.0 (unavoidable)...

    # ...but the loud warning makes it explicit that 0.0 != "$0 spent".
    # build_run_warning_event merges `data` flat into the event dict (not
    # nested under a "data" key) -- see sse_bridge.build_run_warning_event.
    warnings = [e for e in captured if e.get("code") == "unpriced_root_tokens"]
    assert len(warnings) == 1
    assert warnings[0]["unpriced_tokens"] == 48_000
    assert warnings[0]["model"] == "some-unpriced-grok-deployment"
    assert warnings[0]["level"] == "warn"
    assert "48000" in warnings[0]["message"] or "48,000" in warnings[0]["message"]


# ---------------------------------------------------------------------------
# 7 — a backend with no drain path at all fails loud, not silent
# ---------------------------------------------------------------------------


def test_unknown_backend_with_no_drain_path_warns_loudly(tmp_path):
    root_model = ROOT_MODELS["gpt-5"]
    assert root_model.rlm_backend == "openai"
    assert root_model.cred_provider == "openai"  # NOT azure-foundry, NOT oauth
    ctx = _make_ctx(tmp_path / "run1")
    result_obj = _usage_result({"gpt-5": (10_000, 2_000)})
    emit, captured = _make_emit()

    _drain_foundry_root_usage_to_ledger(
        root_model=root_model, result_obj=result_obj, ctx=ctx,
        project_dir=ctx.project_dir, emit=emit,
    )
    _warn_unmetered_root_backend(root_model=root_model, result_obj=result_obj, emit=emit)

    # No drain attempted for a backend outside the known set.
    assert ctx.cost_ledger.entries == []

    warnings = [e for e in captured if e.get("code") == "root_cost_unmetered"]
    assert len(warnings) == 1
    assert warnings[0]["backend"] == "openai"
    assert warnings[0]["root_key"] == "gpt-5"
    assert warnings[0]["tokens"] == 12_000
    assert warnings[0]["level"] == "warn"


def test_unmetered_warning_silent_when_no_usage(tmp_path):
    """No LLM calls made -> $0 IS correct -> no warning (avoid false-positive noise)."""
    root_model = ROOT_MODELS["gpt-5"]
    result_obj = _usage_result({})
    emit, captured = _make_emit()

    _warn_unmetered_root_backend(root_model=root_model, result_obj=result_obj, emit=emit)

    assert captured == []
