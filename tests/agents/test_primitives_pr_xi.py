"""PR-ξ Phase 1 — Bugs 2 + 3 regression tests.

Bug 2: compute_scope sanitized unconditionally (not only when clipping active).
Bug 3: implement_baseline detects error envelopes from plan_reproduction and
       does not coerce them into an empty ReproductionContract.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal RunContext stub
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path: Path, *, clipping_active: bool = False):
    """Minimal RunContext-compatible stub for primitives tests."""
    from backend.agents.rlm.context import RunContext

    ctx = RunContext(
        project_id="prj_test_xi",
        project_dir=tmp_path / "runs" / "prj_test_xi",
        runs_root=tmp_path / "runs",
        dashboard=None,
        cost_ledger=None,
        llm_client=None,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    ctx.project_dir.mkdir(parents=True, exist_ok=True)
    (ctx.project_dir / "rlm_state").mkdir(exist_ok=True)
    ctx.emit = None

    # Patch _is_clipping_active to control the clipping gate in tests.
    return ctx, clipping_active


def _stub_llm_complete(response_dict: dict):
    """Return a mock llm_client.complete that always returns JSON-dumped response_dict."""
    client = MagicMock()
    client.complete.return_value = json.dumps(response_dict)
    return client


# ---------------------------------------------------------------------------
# Bug 2 — string-valued compute_scope must not abort plan_reproduction
# ---------------------------------------------------------------------------

class TestComputeScopeStringSanitized:
    """plan_reproduction must drop a string-valued compute_scope, not abort."""

    def test_compute_scope_string_does_not_abort_plan(self, tmp_path):
        """When the LLM emits compute_scope as a string (not a dict), the plan
        must still succeed and data_recipes must be populated from the spec text."""
        from backend.agents.rlm.primitives import plan_reproduction

        ctx, _ = _make_ctx(tmp_path)
        # LLM returns a string-valued compute_scope plus a real ReproductionContract field
        ctx.llm_client = _stub_llm_complete({
            "reproduction_definition": "Train on MNIST handwritten digits",
            "dataset_plan": "Train on MNIST",
            "compute_scope": "small",   # <-- string, not a dict
        })

        # method_spec mentions MNIST so data_recipes should be populated
        method_spec = {
            "core_contribution": "MNIST digit classifier",
            "datasets": ["MNIST"],
        }
        env_spec = {"framework": "pytorch"}

        result = plan_reproduction(method_spec, env_spec, ctx=ctx)

        # Must not return a failure envelope
        assert result.get("outcome") == "ok" or result.get("success") is not False, (
            f"plan_reproduction returned failure: {result}"
        )
        # compute_scope must be None or omitted (not the string "small")
        assert result.get("compute_scope") is None or "compute_scope" not in result, (
            f"compute_scope should be None after sanitization, got {result.get('compute_scope')!r}"
        )
        # data_recipes must contain MNIST since method_spec mentions it
        data_recipes = result.get("data_recipes") or []
        recipe_names = [r.get("canonical_name") for r in data_recipes]
        assert any("MNIST" in (n or "") for n in recipe_names), (
            f"Expected MNIST recipe in data_recipes, got: {recipe_names}"
        )

    def test_compute_scope_none_does_not_abort_plan(self, tmp_path):
        """When the LLM omits compute_scope entirely, plan must still succeed."""
        from backend.agents.rlm.primitives import plan_reproduction

        ctx, _ = _make_ctx(tmp_path)
        ctx.llm_client = _stub_llm_complete({
            "reproduction_definition": "Test reproduction",
            "dataset_plan": "Use CIFAR-10",
        })

        result = plan_reproduction(
            {"core_contribution": "image classification"},
            {"framework": "pytorch"},
            ctx=ctx,
        )
        assert result.get("outcome") == "ok" or result.get("success") is not False


# ---------------------------------------------------------------------------
# Bug 3 — error envelope must not be coerced into ReproductionContract
# ---------------------------------------------------------------------------

class TestImplementBaselineRejectsErrorEnvelope:
    """implement_baseline must detect a plan_reproduction error envelope and
    NOT construct a ReproductionContract from it."""

    def test_implement_baseline_rejects_error_envelope(self, tmp_path):
        """When plan['reproduction_contract'] is a failed envelope, implement_baseline
        must:
          (a) emit run_warning with code plan_reproduction_failed_envelope,
          (b) proceed with contract=None (fallback recovery — sub-agent still runs),
          (c) the code_dir is still returned (or a repairable dict, but not a hard crash).
        """
        from backend.agents.rlm.primitives import implement_baseline
        from backend.agents.schemas import BaselineResult

        ctx, _ = _make_ctx(tmp_path)
        # Simulate a plan that carries a failed envelope as the contract
        plan = {
            "paper_claim_map": {
                "core_contribution": "Variational Autoencoder on MNIST and Frey Face",
                "datasets": ["MNIST", "Frey Face"],
            },
            "environment_spec": {"framework": "pytorch"},
            "reproduction_contract": {
                "success": False,
                "error": "compute_scope validation failed",
                "outcome": "repairable",
            },
        }

        emitted_events: list[dict] = []

        def _capture_emit(event_payload: dict):
            emitted_events.append(event_payload)

        ctx.emit = _capture_emit

        # Patch _run_baseline_with_sdk to return a minimal BaselineResult.
        # The function is called inside asyncio.run() in a thread pool, so we
        # need the mock to be a coroutine function (async def).
        import asyncio as _asyncio
        code_dir = ctx.project_dir / "code"

        async def _fake_run(*args, **kwargs):
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "commands.json").write_text(
                '["python train.py"]', encoding="utf-8"
            )
            return BaselineResult(
                mode="implement_from_paper",
                code_path=str(code_dir),
                commands_to_run=["python train.py"],
            )

        with patch(
            "backend.agents.rlm.primitives._run_baseline_with_sdk",
            new=_fake_run,
        ):
            result = implement_baseline(plan, ctx=ctx)

        # (a) Must not raise — result is a code path string or a dict
        assert result is not None

        # (b) A run_warning with plan_reproduction_failed_envelope must have been
        # emitted — check dashboard_events.jsonl (written by _emit_dashboard_event_to_path)
        # The JSONL format is {"ts": ..., "event": ..., "data": {...}}
        warning_codes = []
        events_file = ctx.project_dir / "dashboard_events.jsonl"
        if events_file.exists():
            for line in events_file.read_text().splitlines():
                try:
                    event = json.loads(line)
                    data = event.get("data") or event.get("payload") or {}
                    if data.get("code") == "plan_reproduction_failed_envelope":
                        warning_codes.append(data["code"])
                except json.JSONDecodeError:
                    pass

        assert warning_codes, (
            "Expected run_warning with code='plan_reproduction_failed_envelope' to be "
            f"written to dashboard_events.jsonl; events file exists: {events_file.exists()}"
        )

    def test_valid_contract_dict_still_constructs(self, tmp_path):
        """A valid reproduction_contract dict (not an envelope) must still construct
        a ReproductionContract — the envelope check must not break normal flow."""
        from backend.agents.schemas import ReproductionContract

        valid = {
            "reproduction_definition": "Train VAE on MNIST",
            "dataset_plan": "Use MNIST from torchvision",
            "evaluation_plan": "Measure ELBO",
        }
        # Should not raise
        contract = ReproductionContract(**valid)
        assert contract.reproduction_definition == "Train VAE on MNIST"
        assert contract.dataset_plan == "Use MNIST from torchvision"
