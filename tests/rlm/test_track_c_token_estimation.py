"""Tests for Track C — token estimation / optimization.

C1 — estimate_cost_usd resolves bare and provider-prefixed model keys.
C2 — equivalent_cost_usd returns hypothetical API cost for OAuth models.
C3 — ClaudeOauthClient.completion feeds the module-level root-usage sink;
      run.py drains it and ledgers cache tokens WITHOUT double-counting
      input/output in final_report.cost.llm_usd.
C4 — write_final_report_rlm calls recompute_calibration best-effort.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# C1 — key resolution in estimate_cost_usd
# ---------------------------------------------------------------------------


def test_c1_bare_key_resolves():
    """estimate_cost_usd resolves a bare 'claude-sonnet-4-6' key."""
    from backend.agents.resilience.pricing import estimate_cost_usd

    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    cost = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert cost is not None
    assert cost == pytest.approx(3.00, rel=1e-4)


def test_c1_bare_oauth_key_resolves():
    """estimate_cost_usd resolves 'claude-oauth' (zero-cost subscription)."""
    from backend.agents.resilience.pricing import estimate_cost_usd

    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    cost = estimate_cost_usd("claude-oauth", usage)
    assert cost is not None
    assert cost == pytest.approx(0.0)


def test_c1_provider_prefixed_model_resolves():
    """estimate_cost_usd resolves 'anthropic.claude-sonnet-4-6'."""
    from backend.agents.resilience.pricing import estimate_cost_usd, PRICING

    # Add a provider-prefixed key to PRICING for this test so the suffix-match
    # path is exercised even if the catalog doesn't have one.
    original = dict(PRICING)
    from backend.agents.resilience.pricing import ModelPricing
    PRICING["anthropic.claude-sonnet-4-6"] = ModelPricing(
        input_per_1m=3.00, output_per_1m=15.00
    )
    try:
        usage = {"input_tokens": 1_000_000, "output_tokens": 0}
        cost = estimate_cost_usd("claude-sonnet-4-6", usage)
        assert cost is not None
        assert cost == pytest.approx(3.00, rel=1e-4)
    finally:
        # Restore original state
        PRICING.clear()
        PRICING.update(original)


def test_c1_unknown_model_returns_none():
    """estimate_cost_usd returns None for a completely unknown model."""
    from backend.agents.resilience.pricing import estimate_cost_usd

    cost = estimate_cost_usd("totally-unknown-model-xyz", {"input_tokens": 1000})
    assert cost is None


def test_c1_resolve_pricing_suffix_match():
    """_resolve_pricing matches a bare name against suffixes of provider-prefixed keys."""
    from backend.agents.resilience.pricing import _resolve_pricing, PRICING, ModelPricing

    original = dict(PRICING)
    PRICING["myprovider.mymodel"] = ModelPricing(input_per_1m=5.0, output_per_1m=20.0)
    try:
        result = _resolve_pricing("mymodel")
        assert result is not None
        assert result.input_per_1m == pytest.approx(5.0)
    finally:
        PRICING.clear()
        PRICING.update(original)


def test_c1_resolve_pricing_strips_prefix_from_model():
    """_resolve_pricing strips 'provider.' prefix from model and looks up bare key."""
    from backend.agents.resilience.pricing import _resolve_pricing

    # 'claude-sonnet-4-6' is a bare key in PRICING
    result = _resolve_pricing("anthropic.claude-sonnet-4-6")
    assert result is not None
    assert result.input_per_1m == pytest.approx(3.00)


# ---------------------------------------------------------------------------
# C2 — equivalent_cost_usd for OAuth models
# ---------------------------------------------------------------------------


def test_c2_oauth_equivalent_cost_maps_to_sonnet():
    """equivalent_cost_usd for 'claude-oauth' returns claude-sonnet-4-6 pricing."""
    from backend.agents.resilience.pricing import equivalent_cost_usd, estimate_cost_usd

    usage = {"input_tokens": 500_000, "output_tokens": 100_000}
    equiv = equivalent_cost_usd("claude-oauth", usage)
    real_sonnet = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert equiv is not None
    assert equiv == pytest.approx(real_sonnet, rel=1e-6)


def test_c2_oauth_real_cost_stays_zero():
    """estimate_cost_usd for 'claude-oauth' remains $0 (actual billed cost)."""
    from backend.agents.resilience.pricing import estimate_cost_usd

    usage = {"input_tokens": 1_000_000, "output_tokens": 500_000}
    cost = estimate_cost_usd("claude-oauth", usage)
    assert cost is not None
    assert cost == pytest.approx(0.0)


def test_c2_non_oauth_equivalent_same_as_real():
    """equivalent_cost_usd for a non-OAuth model returns the same as estimate_cost_usd."""
    from backend.agents.resilience.pricing import equivalent_cost_usd, estimate_cost_usd

    usage = {"input_tokens": 100_000, "output_tokens": 50_000}
    equiv = equivalent_cost_usd("claude-sonnet-4-6", usage)
    real = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert equiv == pytest.approx(real, rel=1e-6)


def test_c2_unknown_model_returns_none():
    """equivalent_cost_usd returns None when neither the model nor its equivalent has pricing."""
    from backend.agents.resilience.pricing import equivalent_cost_usd

    # An unknown model that is not in OAUTH_EQUIVALENT_MODEL
    cost = equivalent_cost_usd("completely-unknown-xyz", {"input_tokens": 100})
    assert cost is None


# ---------------------------------------------------------------------------
# C3 — Root usage sink and double-count proof
# ---------------------------------------------------------------------------


def test_c3_drain_root_usage_returns_and_clears():
    """drain_root_usage() returns accumulated usage and clears the global dict."""
    import importlib
    import backend.agents.rlm.claude_oauth_client as mod

    # Force the CLI transport for this test so usage is actually accumulated.
    original_root_usage = dict(mod._ROOT_USAGE)
    # Manually inject usage into the sink.
    with mod._ROOT_USAGE_LOCK:
        mod._ROOT_USAGE.clear()
        mod._ROOT_USAGE["claude-sonnet-4-6"] = {
            "calls": 3,
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 30,
        }

    snapshot = mod.drain_root_usage()
    assert snapshot["claude-sonnet-4-6"]["calls"] == 3
    assert snapshot["claude-sonnet-4-6"]["input_tokens"] == 1000
    assert snapshot["claude-sonnet-4-6"]["cache_creation_input_tokens"] == 50

    # After draining the sink should be empty.
    assert mod._ROOT_USAGE == {}

    # Restore
    with mod._ROOT_USAGE_LOCK:
        mod._ROOT_USAGE.update(original_root_usage)


def test_c3_completion_cli_path_increments_sink(monkeypatch):
    """completion() on the CLI path increments _ROOT_USAGE in addition to per-instance counters."""
    import backend.agents.rlm.claude_oauth_client as mod
    from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

    monkeypatch.setenv("REPROLAB_RLM_ROOT_TRANSPORT", "cli")

    cli_usage = {
        "input_tokens": 800,
        "output_tokens": 150,
        "cache_creation_input_tokens": 60,
        "cache_read_input_tokens": 20,
    }

    # Patch _cli_complete to return a fixed result without real subprocess.
    monkeypatch.setattr(
        ClaudeOauthClient,
        "_cli_complete",
        lambda self, system, user, model: ("```repl\npass\n```", cli_usage),
    )

    # Clear the sink before the test.
    with mod._ROOT_USAGE_LOCK:
        mod._ROOT_USAGE.clear()

    client = ClaudeOauthClient(model_name="claude-sonnet-4-6")
    result = client.completion("test prompt")
    assert result  # non-empty

    # Per-instance counters updated.
    assert client.model_input_tokens["claude-sonnet-4-6"] == 800
    assert client.model_cache_creation_tokens["claude-sonnet-4-6"] == 60

    # Module-level sink also updated.
    snapshot = mod.drain_root_usage()
    assert "claude-sonnet-4-6" in snapshot
    assert snapshot["claude-sonnet-4-6"]["input_tokens"] == 800
    assert snapshot["claude-sonnet-4-6"]["cache_creation_input_tokens"] == 60
    assert snapshot["claude-sonnet-4-6"]["cache_read_input_tokens"] == 20
    assert snapshot["claude-sonnet-4-6"]["calls"] == 1


def test_c3_no_double_count_in_cost_dict():
    """
    Ledgering rlm_root rows does NOT change _cost_dict (which reads usage_summary).

    _cost_dict reads result.usage_summary — the rlm library's accumulated totals.
    The cost_ledger is read for primitives_usd only.  Therefore appending an
    rlm_root row to the ledger does not affect llm_usd — no double-count.
    """
    from backend.agents.rlm.report import _cost_dict
    from backend.agents.rlm.context import RunContext
    from backend.agents.resilience.cost import RunCostLedger, CostLedgerEntry

    # Build a minimal RunContext-like mock with a cost_ledger.
    tmp_dir = Path("/tmp/test_c3_no_double_count")
    tmp_dir.mkdir(exist_ok=True)
    ledger_path = tmp_dir / "cost_ledger.jsonl"

    ledger = RunCostLedger(project_id="test-dc", path=ledger_path)
    ctx_mock = MagicMock(spec=RunContext)
    ctx_mock.cost_ledger = ledger

    # Build a mock rlm result with usage_summary carrying $5.00 of root spend.
    from rlm.core.types import UsageSummary, ModelUsageSummary
    summary = ModelUsageSummary(
        total_calls=2,
        total_input_tokens=1_000_000,
        total_output_tokens=200_000,
        total_cost=5.00,
    )
    usage_summary = UsageSummary(model_usage_summaries={"claude-sonnet-4-6": summary})
    result_mock = MagicMock()
    result_mock.usage_summary = usage_summary

    # Before adding root ledger row.
    cost_before = _cost_dict(result_mock, ctx_mock)
    assert cost_before["llm_usd"] == pytest.approx(5.00)
    assert cost_before["primitives"] == pytest.approx(0.0)

    # Add an rlm_root ledger entry (as run.py's drain hook would do).
    # ProviderName is a Literal type alias — pass the string value directly.
    root_entry = CostLedgerEntry.from_usage(
        agent_id="rlm_root",
        attempt_index=0,
        provider="anthropic",  # type: ignore[arg-type]  # Literal["anthropic","openai"]
        model="claude-oauth",
        usage={
            "input_tokens": 1_000_000,
            "output_tokens": 200_000,
            "cache_creation_input_tokens": 50_000,
            "cache_read_input_tokens": 10_000,
            "reasoning_tokens": 0,
        },
    )
    ledger.append(root_entry)

    # After — llm_usd MUST be unchanged ($5.00); primitives_usd gains the ledger entry
    # (which is $0 for OAuth, so primitives also stays near 0).
    cost_after = _cost_dict(result_mock, ctx_mock)
    assert cost_after["llm_usd"] == pytest.approx(5.00), (
        "llm_usd must not change — it reads usage_summary, not the ledger"
    )
    # oauth cost is $0
    assert cost_after["primitives"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# C4 — Auto-calibration at finalize
# ---------------------------------------------------------------------------


def test_c4_calibration_auto_runs_at_finalize(tmp_path, monkeypatch):
    """write_final_report_rlm calls recompute_calibration for non-failed runs."""
    from backend.agents.rlm.report import write_final_report_rlm, RLMFinalReport
    from backend.agents.rlm.context import RunContext

    project_dir = tmp_path / "runs" / "test_run"
    project_dir.mkdir(parents=True)

    # Minimal ledger so tokens_total.json can be written.
    (project_dir / "cost_ledger.jsonl").write_text("", encoding="utf-8")

    # Evidence gate (ported 2026-06-09): seed a success+metrics row so the
    # partial verdict is legitimately earned (calibration only runs for
    # non-failed verdicts, and an evidence-less partial now downgrades).
    import json as _json
    (project_dir / "experiment_runs.jsonl").write_text(
        _json.dumps({"success": True, "metrics": {"accuracy": 0.9}}) + "\n",
        encoding="utf-8",
    )

    report = RLMFinalReport(verdict="partial", reproduction_summary="C4 test")

    # Patch recompute_calibration to verify it is called.
    called_with: list[Path] = []

    import backend.agents.rlm.report as report_mod
    original = getattr(report_mod, "recompute_calibration", None)

    # We need to ensure the env-var opt-out is NOT set.
    monkeypatch.delenv("REPROLAB_UPDATE_CALIBRATION", raising=False)

    # Patch via monkeypatch on the import inside the function
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(
        cal_mod,
        "recompute_calibration",
        lambda runs_root: called_with.append(Path(runs_root)),
    )

    # Also redirect calibration path to avoid writing to data/.
    cal_path = tmp_path / "calibration.json"
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    ctx_mock = MagicMock(spec=RunContext)
    ctx_mock.cost_ledger = None

    # write_final_report_rlm does NOT take a ctx arg — it takes report + project_dir.
    write_final_report_rlm(report=report, project_dir=project_dir)

    assert len(called_with) >= 1, "recompute_calibration should be called at finalize"
    assert called_with[0] == tmp_path / "runs"


def test_c4_calibration_skipped_for_failed_verdict(tmp_path, monkeypatch):
    """write_final_report_rlm skips calibration for failed runs."""
    from backend.agents.rlm.report import write_final_report_rlm, RLMFinalReport

    project_dir = tmp_path / "runs" / "test_run"
    project_dir.mkdir(parents=True)
    (project_dir / "cost_ledger.jsonl").write_text("", encoding="utf-8")

    report = RLMFinalReport(verdict="failed", reproduction_summary="failed run")

    monkeypatch.delenv("REPROLAB_UPDATE_CALIBRATION", raising=False)

    called_with: list[Path] = []

    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(
        cal_mod,
        "recompute_calibration",
        lambda runs_root: called_with.append(Path(runs_root)),
    )

    write_final_report_rlm(report=report, project_dir=project_dir)

    assert len(called_with) == 0, "recompute_calibration must NOT be called for failed runs"


def test_c4_calibration_skipped_when_opt_out(tmp_path, monkeypatch):
    """Setting REPROLAB_UPDATE_CALIBRATION=false suppresses the auto-call."""
    from backend.agents.rlm.report import write_final_report_rlm, RLMFinalReport

    project_dir = tmp_path / "runs" / "test_run"
    project_dir.mkdir(parents=True)
    (project_dir / "cost_ledger.jsonl").write_text("", encoding="utf-8")

    report = RLMFinalReport(verdict="partial", reproduction_summary="opt-out test")

    monkeypatch.setenv("REPROLAB_UPDATE_CALIBRATION", "false")

    called_with: list[Path] = []

    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(
        cal_mod,
        "recompute_calibration",
        lambda runs_root: called_with.append(Path(runs_root)),
    )

    write_final_report_rlm(report=report, project_dir=project_dir)

    assert len(called_with) == 0, "recompute_calibration must be suppressed when opt-out is set"


def test_c4_calibration_on_fixture_run_dir(tmp_path, monkeypatch):
    """recompute_calibration correctly aggregates a fixture run directory."""
    from backend.services.pricing.calibration import recompute_calibration

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_fixture"
    run_dir.mkdir(parents=True)
    (run_dir / ".preserved").write_text(
        json.dumps({"verdict": "partial", "schema_version": 1}), encoding="utf-8"
    )
    ledger_lines = [
        json.dumps({"primitive": "understand_section", "input_tokens": 8000, "output_tokens": 600}),
        json.dumps({"primitive": "implement_baseline", "input_tokens": 25000, "output_tokens": 7000}),
        json.dumps({"primitive": "rlm_root", "input_tokens": 5000, "output_tokens": 0,
                    "cache_creation_input_tokens": 3750, "cache_read_input_tokens": 1500}),
    ]
    (run_dir / "cost_ledger.jsonl").write_text("\n".join(ledger_lines), encoding="utf-8")

    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    result = recompute_calibration(runs_root)
    assert result["based_on_n_preserved_runs"] == 1
    per_prim = result["per_primitive"]
    assert "understand_section" in per_prim
    assert per_prim["understand_section"]["avg_input_tokens"] == pytest.approx(8000.0)
    assert "implement_baseline" in per_prim
    # rlm_root is now tracked in calibration too.
    assert "rlm_root" in per_prim
    assert per_prim["rlm_root"]["avg_input_tokens"] == pytest.approx(5000.0)
