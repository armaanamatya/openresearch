"""Tests for the /lab provider-picker wiring in live_runs.

Two surfaces, both previously inert (forwarded from the UI, never consumed):
  - ``apply_picker_overrides`` maps ``root_provider`` → ``request.model`` (the
    root-model token, which rides the config dict to the child's
    ``resolve_root_model``).
  - ``_subprocess_env`` maps ``subagent_auth`` → child env (it can't ride
    request.model — that selects the root, not the sub-agents).

Each ships the mandatory OFF (byte-identical) + ON (behavior-changing) pair.
"""

from __future__ import annotations

import json

import pytest

from backend.services.events.live_runs import (
    FileLiveRunService,
    StartRunRequest,
    apply_autonomous_profile_override,
    apply_picker_overrides,
)


def _svc(tmp_path):
    # Real instance; tmp_path has no .env so _subprocess_env loads nothing from
    # disk (mirrors tests/routes/test_advanced_field_forwarding.py::_svc).
    return FileLiveRunService(repo_root=tmp_path)


# --------------------------------------------------------------------------- #
# apply_picker_overrides — root_provider → request.model (the §1.6 matrix)
# --------------------------------------------------------------------------- #


def test_off_state_identity_when_root_provider_unset():
    """root_provider=None → the exact same request object (byte-identical)."""
    req = StartRunRequest()
    assert req.root_provider is None
    out = apply_picker_overrides(req)
    assert out is req
    assert out.model == "sonnet"


@pytest.mark.parametrize(
    ("root_provider", "in_model", "expected_model"),
    [
        ("foundry", "sonnet", "sonnet-foundry"),
        ("foundry", "opus", "opus-foundry"),
        ("foundry", "claude-opus-4-8", "opus-foundry"),  # opus substring → opus tier
        ("anthropic_oauth", "sonnet", "claude-oauth"),
        ("openai_api", "sonnet", "gpt-5"),
        ("azure_openai", "sonnet", "azure-gpt-4o"),
        ("featherless", "sonnet", "qwen3-coder-featherless"),
    ],
)
def test_root_provider_maps_to_root_token(root_provider, in_model, expected_model):
    req = StartRunRequest(root_provider=root_provider, model=in_model)
    out = apply_picker_overrides(req)
    assert out.model == expected_model
    # Only request.model is rewritten — the picker never touches provider/sandbox.
    assert out.provider == req.provider
    assert out.sandbox == req.sandbox


def test_anthropic_api_leaves_model_untouched():
    """anthropic_api rides the existing opus/sonnet → ANTHROPIC_API_KEY path."""
    req = StartRunRequest(root_provider="anthropic_api", model="opus")
    out = apply_picker_overrides(req)
    assert out is req
    assert out.model == "opus"


def test_run_spec_pin_wins_over_picker():
    """An explicit run-spec (which can pin root + per-role models) is not
    clobbered by a picker default — apply_picker_overrides is a no-op."""
    req = StartRunRequest(root_provider="foundry", model="sonnet", run_spec="configs/x.json")
    out = apply_picker_overrides(req)
    assert out is req
    assert out.model == "sonnet"


def test_unknown_provider_is_fail_safe():
    """A future/unknown root_provider value leaves the request untouched."""
    req = StartRunRequest(root_provider="some_future_provider", model="sonnet")
    out = apply_picker_overrides(req)
    assert out is req
    assert out.model == "sonnet"


def test_autonomous_override_still_wins_after_picker():
    """Precedence: picker runs first, autonomous runs last and overrides it."""
    req = StartRunRequest(root_provider="foundry", model="sonnet", autonomous=True)
    after_picker = apply_picker_overrides(req)
    assert after_picker.model == "sonnet-foundry"  # picker applied
    final = apply_autonomous_profile_override(after_picker)
    assert final.model == "opus-foundry"  # autonomous wins
    assert final.sandbox == "gcp"


# --------------------------------------------------------------------------- #
# _subprocess_env — subagent_auth → child env
# --------------------------------------------------------------------------- #


def test_subagent_auth_foundry_sets_executor_role_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_ROLE_MODELS", raising=False)
    req = StartRunRequest(subagent_auth="foundry")
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    parsed = json.loads(env["OPENRESEARCH_ROLE_MODELS"])
    assert parsed["executor"] == "sonnet-foundry"


def test_subagent_auth_foundry_merges_existing_role_models(tmp_path, monkeypatch):
    """A pre-existing planner/verifier pin survives the executor addition."""
    monkeypatch.setenv("OPENRESEARCH_ROLE_MODELS", '{"planner": "opus", "verifier": "sonnet"}')
    req = StartRunRequest(subagent_auth="foundry")
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    parsed = json.loads(env["OPENRESEARCH_ROLE_MODELS"])
    assert parsed["executor"] == "sonnet-foundry"
    assert parsed["planner"] == "opus"
    assert parsed["verifier"] == "sonnet"


def test_subagent_auth_foundry_merges_kv_form(tmp_path, monkeypatch):
    """The existing value may be the k=v CLI form, not JSON — still merged."""
    monkeypatch.setenv("OPENRESEARCH_ROLE_MODELS", "planner=opus,grader=o4-mini")
    req = StartRunRequest(subagent_auth="foundry")
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    parsed = json.loads(env["OPENRESEARCH_ROLE_MODELS"])
    assert parsed["executor"] == "sonnet-foundry"
    assert parsed["planner"] == "opus"
    assert parsed["grader"] == "o4-mini"


@pytest.mark.parametrize(
    ("subagent_auth", "expected_strategy"),
    [("anthropic_oauth", "oauth_only"), ("anthropic_api", "api_only")],
)
def test_subagent_auth_anthropic_sets_auth_strategy(tmp_path, monkeypatch, subagent_auth, expected_strategy):
    monkeypatch.delenv("OPENRESEARCH_LLM_AUTH_STRATEGY", raising=False)
    req = StartRunRequest(subagent_auth=subagent_auth)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    assert env["OPENRESEARCH_LLM_AUTH_STRATEGY"] == expected_strategy


def test_subagent_auth_unset_is_byte_identical(tmp_path, monkeypatch):
    """subagent_auth=None → neither knob is set (off-state invariant)."""
    monkeypatch.delenv("OPENRESEARCH_ROLE_MODELS", raising=False)
    monkeypatch.delenv("OPENRESEARCH_LLM_AUTH_STRATEGY", raising=False)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), StartRunRequest())
    assert "OPENRESEARCH_ROLE_MODELS" not in env
    assert "OPENRESEARCH_LLM_AUTH_STRATEGY" not in env
