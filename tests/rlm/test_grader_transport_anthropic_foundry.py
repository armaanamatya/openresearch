from unittest.mock import patch

import pytest

from backend.agents.rlm import grader_transport


def test_anthropic_foundry_backend_builds_scoped_client(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-42")
    with patch.object(grader_transport, "AnthropicMessagesClient") as mock_cls:
        grader_transport.build_transport_client(
            backend="anthropic-foundry",
            model="claude-sonnet-5",
            fallback_client=object(),
            fallback_label="fb",
            role_label="grader",
        )
    _, kwargs = mock_cls.call_args
    assert kwargs["base_url"] == "https://r.services.ai.azure.com/anthropic/v1"
    assert kwargs["api_key"] == "k-42"
    assert kwargs["model"] == "claude-sonnet-5"


def _neutralize_foundry_anthropic_settings(monkeypatch):
    """Delete the env + neutralize the Settings/.env fallback (this checkout's
    real .env may carry live Foundry credentials — see MEMORY.md — so
    os.environ deletion alone is not hermetic). Mirrors
    tests/agents/runtime/test_foundry_anthropic.py::test_unset_is_empty_and_absent.
    """
    from backend.agents.runtime import foundry_anthropic as fa

    for v in (
        "AZURE_FOUNDRY_ENDPOINT",
        "AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
        "AZURE_FOUNDRY_API_KEY",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(fa, "_env_or_settings", lambda *_a, **_kw: "")


def test_anthropic_foundry_missing_creds_raises_internally_and_falls_back(monkeypatch):
    # I2 (Wave-1 review): unset Foundry creds must NOT silently build a broken
    # client (empty api_key/base_url). The branch now gates on credentials and
    # raises ValueError internally — same idiom as the sibling `azure` branch
    # (`if not azure_endpoint: raise ValueError(...)`). build_transport_client's
    # blanket try/except (wrapping every branch, matching every sibling) catches
    # it and returns the fallback UNCHANGED — build_transport_client's contract
    # is "we NEVER raise" (the unguarded verifier call site in run.py depends on
    # this). The fail-closed behaviour lives one level up, in
    # build_validator_client (see test below) — exactly like the existing
    # azure/azure-foundry missing-creds coverage.
    _neutralize_foundry_anthropic_settings(monkeypatch)

    sentinel = object()
    client, label = grader_transport.build_transport_client(
        backend="anthropic-foundry",
        model="claude-sonnet-5",
        fallback_client=sentinel,
        fallback_label="fb",
        role_label="grader",
    )
    assert client is sentinel  # missing creds → fallback, NEVER a broken client
    assert label == "fb"


def test_anthropic_foundry_missing_creds_fail_closed_via_validator(monkeypatch):
    # The validator role deliberately converts the fallback-identity signal
    # above into a raise (spec 2026-06-20 §7.2) — matching the existing
    # test_validator_azure_foundry_missing_creds_fail_closed coverage for the
    # OpenAI-compatible azure-foundry backend.
    _neutralize_foundry_anthropic_settings(monkeypatch)
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "anthropic-foundry")
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_MODEL", raising=False)

    with pytest.raises(ValueError, match="anthropic-foundry"):
        grader_transport.build_validator_client(
            fallback_client=object(), fallback_label="exec"
        )
