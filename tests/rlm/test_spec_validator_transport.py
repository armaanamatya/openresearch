"""build_spec_validator_client — FAIL-CLOSED transport, VALIDATOR_* fallback.

Unit tests only; every underlying SDK client is mocked, no network. Mirrors
``tests/rlm/test_validator_transport.py`` (the ``build_validator_client``
suite) exactly, plus the SPEC-specific fallback/precedence cases: this
builder is the SPEC-role sibling of ``build_validator_client`` — it reads
``OPENRESEARCH_SPEC_VALIDATOR_BACKEND``/``_MODEL`` first, falling back to
``OPENRESEARCH_VALIDATOR_BACKEND``/``_MODEL`` when the SPEC-specific vars are
unset, so the rubric-vs-paper spec validator shares the external validator's
transport by default or can be configured independently. Same FAIL-CLOSED
contract: an explicitly-requested spec validator that cannot construct an
independent transport RAISES rather than silently riding the fallback
(planner/executor) client that WROTE the rubric.
"""
from __future__ import annotations

import types

import pytest

from backend.agents.rlm.grader_transport import (
    build_spec_validator_client,
    build_transport_client,
)


@pytest.fixture(autouse=True)
def _clear_spec_validator_env(monkeypatch):
    """Every test starts with BOTH the SPEC and VALIDATOR env unset (hermetic)."""
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", raising=False)
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_MODEL", raising=False)


# ---------------------------------------------------------------------------
# 1. Both backends unset → passthrough.
# ---------------------------------------------------------------------------
def test_spec_validator_passthrough_when_both_backends_unset():
    sentinel = object()
    client, label = build_spec_validator_client(fallback_client=sentinel, fallback_label="exec")
    assert client is sentinel  # unchanged → caller resolves separation=unavailable
    assert label == "exec"


def test_spec_validator_model_only_is_passthrough_spec_model(monkeypatch):
    # A model override with no backend cannot pick a transport → passthrough
    # (NOT a raise — the spec validator was not actually requested).
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "gpt-4.1")
    sentinel = object()
    client, label = build_spec_validator_client(fallback_client=sentinel, fallback_label="exec")
    assert client is sentinel
    assert label == "exec"


def test_spec_validator_model_only_is_passthrough_validator_fallback_model(monkeypatch):
    # Same, but the model-only value comes from the VALIDATOR_* fallback var.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4.1")
    sentinel = object()
    client, label = build_spec_validator_client(fallback_client=sentinel, fallback_label="exec")
    assert client is sentinel
    assert label == "exec"


# ---------------------------------------------------------------------------
# 2. FAIL-CLOSED — backend set but the transport can't build → ValueError.
# ---------------------------------------------------------------------------
def test_spec_validator_build_fail_closed_missing_creds(monkeypatch):
    # The canonical load-bearing case: backend=azure with no AZURE_OPENAI_*.
    # build_transport_client falls back internally; build_spec_validator_client
    # must convert that fallback into a raise rather than ride the fallback client.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError):
        build_spec_validator_client(fallback_client=object(), fallback_label="exec")


def test_spec_validator_build_fail_closed_unknown_backend(monkeypatch):
    # An unknown backend also falls back inside build_transport_client → raise.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "bananas")
    with pytest.raises(ValueError):
        build_spec_validator_client(fallback_client=object(), fallback_label="exec")


def test_spec_validator_build_fail_closed_construction_error(monkeypatch):
    # Force the azure client constructor to blow up → build_transport_client
    # fails soft to the fallback → build_spec_validator_client must RAISE.
    import backend.services.context.workspace.tools.azure_openai_client as azc

    def _boom(**kwargs):
        raise RuntimeError("no SDK")

    monkeypatch.setattr(azc, "AzureOpenAILlmClient", _boom)
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    with pytest.raises(ValueError):
        build_spec_validator_client(fallback_client=object(), fallback_label="exec")


# ---------------------------------------------------------------------------
# 3. The azure distinct-deployment case — SPEC_VALIDATOR_MODEL overrides
# AZURE_OPENAI_DEPLOYMENT (mirrors the validator distinct-deployment test).
# ---------------------------------------------------------------------------
def _patch_fake_azure(monkeypatch):
    import backend.services.context.workspace.tools.azure_openai_client as azc

    class _FakeAzure:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(azc, "AzureOpenAILlmClient", _FakeAzure)
    return _FakeAzure


def test_azure_spec_validator_uses_distinct_deployment(monkeypatch):
    fake = _patch_fake_azure(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "execA")  # the executor's deployment
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "specB")  # the spec validator's

    client, label = build_spec_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, fake)
    # The deployment ROUTES the request on Azure → it must be the override specB.
    assert client.kwargs["azure_deployment"] == "specB"
    assert client.kwargs["azure_deployment"] != "execA"
    assert client.kwargs["model"] == "specB"
    assert label == "spec_validator:azure:specB"


# ---------------------------------------------------------------------------
# 4. backend=oauth — the funded Claude transport, constructs without API creds.
# ---------------------------------------------------------------------------
def test_spec_validator_oauth_backend_constructs(monkeypatch):
    import backend.services.context.workspace.tools.rlm_query as _rq

    class _FakeOAuth:
        def __init__(self, model=None):
            self.model = model

    monkeypatch.setattr(_rq, "ClaudeLlmClient", _FakeOAuth)
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "oauth")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "claude-opus-4-8")

    client, label = build_spec_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, _FakeOAuth)
    assert client.model == "claude-opus-4-8"
    assert label == "spec_validator:oauth:claude-opus-4-8"


def test_spec_validator_azure_foundry_missing_creds_fail_closed(monkeypatch):
    # Foundry with no creds falls back inside build_transport_client → raise.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure-foundry")
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_DEPLOYMENT", raising=False)
    monkeypatch.setattr(
        "backend.config.get_settings", lambda *a, **k: types.SimpleNamespace()
    )
    with pytest.raises(ValueError):
        build_spec_validator_client(fallback_client=object(), fallback_label="exec")


# ---------------------------------------------------------------------------
# 5. NEW — the VALIDATOR_* fallback: SPEC unset, VALIDATOR set → used.
# ---------------------------------------------------------------------------
def test_spec_validator_falls_back_to_validator_backend_oauth(monkeypatch):
    import backend.services.context.workspace.tools.rlm_query as _rq

    class _FakeOAuth:
        def __init__(self, model=None):
            self.model = model

    monkeypatch.setattr(_rq, "ClaudeLlmClient", _FakeOAuth)
    # SPEC_VALIDATOR_BACKEND left unset; only the VALIDATOR_* fallback is set.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "oauth")

    client, label = build_spec_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, _FakeOAuth)
    # role_label="spec_validator" even though the BACKEND came from VALIDATOR_*.
    assert label.startswith("spec_validator:oauth:")


# ---------------------------------------------------------------------------
# 6. NEW — precedence: both set to DIFFERENT backends → SPEC wins.
# ---------------------------------------------------------------------------
def test_spec_validator_backend_takes_precedence_over_validator_backend(monkeypatch):
    import backend.services.context.workspace.tools.rlm_query as _rq

    class _FakeOAuth:
        def __init__(self, model=None):
            self.model = model

    monkeypatch.setattr(_rq, "ClaudeLlmClient", _FakeOAuth)
    # SPEC says oauth (cheap, no creds needed); VALIDATOR says azure (would need
    # AZURE_OPENAI_* creds it doesn't have here) — if VALIDATOR ever won this
    # would raise ValueError (fail-closed on the missing azure creds) instead of
    # constructing cleanly, so this also catches a precedence inversion.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "oauth")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    client, label = build_spec_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, _FakeOAuth)
    assert label == "spec_validator:oauth:claude-oauth"


# ---------------------------------------------------------------------------
# 7. NEW — model fallback: SPEC_VALIDATOR_MODEL unset, VALIDATOR_MODEL used.
# ---------------------------------------------------------------------------
def test_spec_validator_model_falls_back_to_validator_model(monkeypatch):
    fake = _patch_fake_azure(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    # OPENRESEARCH_SPEC_VALIDATOR_MODEL left unset; VALIDATOR_MODEL supplies it.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "valB")

    client, label = build_spec_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, fake)
    assert client.kwargs["model"] == "valB"
    assert client.kwargs["azure_deployment"] == "valB"
    assert label == "spec_validator:azure:valB"


# ---------------------------------------------------------------------------
# 8. build_transport_client role_label — spec_validator namespaces correctly
# (direct check of the shared dispatch core, mirroring test_grader_transport.py).
# ---------------------------------------------------------------------------
def test_build_transport_client_role_label_namespaces_spec_validator(monkeypatch):
    fake = _patch_fake_azure(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")

    client, label = build_transport_client(
        backend="azure",
        model="gpt-4o",
        fallback_client=object(),
        fallback_label="fb",
        role_label="spec_validator",
    )
    assert isinstance(client, fake)
    assert label == "spec_validator:azure:gpt-4o"
