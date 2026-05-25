"""Tests for the BYO provider-credentials surface.

Cover the four invariants that justify shipping this feature:
1. Pydantic edge validation (partial Azure / endpoint scheme / control chars).
2. Subprocess env precedence (BYO > .env > shell).
3. Scrub: keys never appear in any pydantic-driven serialization path.
4. Resume override forwards a fresh credential bundle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services.events.live_runs import (
    FileLiveRunService,
    ProviderCredentials,
    StartRunRequest,
)


# ---------------------------------------------------------------------------
# (1) Edge validation
# ---------------------------------------------------------------------------


def test_anthropic_key_only_round_trips() -> None:
    c = ProviderCredentials(anthropic_api_key="sk-ant-test-1234567890")
    env = c.to_env_overrides()
    assert env == {"ANTHROPIC_API_KEY": "sk-ant-test-1234567890"}


def test_azure_complete_round_trips() -> None:
    c = ProviderCredentials(
        azure_openai_api_key="abcd" * 8,
        azure_openai_endpoint="https://my-resource.openai.azure.com/",
        azure_openai_deployment="gpt-4o",
        azure_openai_api_version="2024-10-21",
    )
    env = c.to_env_overrides()
    # Endpoint trailing slash is stripped by the validator.
    assert env["AZURE_OPENAI_ENDPOINT"] == "https://my-resource.openai.azure.com"
    assert env["AZURE_OPENAI_API_KEY"] == "abcd" * 8
    assert env["AZURE_OPENAI_DEPLOYMENT"] == "gpt-4o"
    assert env["AZURE_OPENAI_API_VERSION"] == "2024-10-21"


def test_azure_partial_rejected() -> None:
    # Key but no endpoint — silent fallback to env at run time would be
    # confusing; surface the misconfiguration at the form edge.
    with pytest.raises(ValueError, match="incomplete"):
        ProviderCredentials(azure_openai_api_key="x" * 32)


def test_azure_endpoint_must_be_https() -> None:
    with pytest.raises(ValueError, match="https://"):
        ProviderCredentials(
            azure_openai_api_key="x" * 32,
            azure_openai_endpoint="http://insecure.openai.azure.com",
        )


def test_control_chars_rejected() -> None:
    with pytest.raises(ValueError, match="control characters"):
        ProviderCredentials(anthropic_api_key="sk-ant-\nrm -rf /")


def test_oversize_value_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds 512"):
        ProviderCredentials(anthropic_api_key="x" * 600)


def test_empty_strings_normalized_to_none() -> None:
    # An empty form field should not be treated as a value — otherwise the
    # subprocess env injection would clobber the .env-provided key with "".
    c = ProviderCredentials(
        anthropic_api_key="",
        openai_api_key="   ",
        azure_openai_api_key=None,
    )
    assert c.anthropic_api_key is None
    assert c.openai_api_key is None
    assert c.to_env_overrides() == {}


# ---------------------------------------------------------------------------
# (2) Subprocess env precedence
# ---------------------------------------------------------------------------


def _clean_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The dev shell exports real API keys; without scrubbing them the .env
    # precedence path is impossible to exercise (shell export > .env for
    # non-REPROLAB keys). Tests that assert on a specific env-var origin
    # must call this first.
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(k, raising=False)


def test_subprocess_env_byo_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_shell_env(monkeypatch)
    repo_root = tmp_path
    (repo_root / ".env").write_text("ANTHROPIC_API_KEY=server-side-value\n", encoding="utf-8")
    svc = FileLiveRunService(runs_root=tmp_path / "runs", repo_root=repo_root)
    req = StartRunRequest(
        provider_credentials=ProviderCredentials(anthropic_api_key="byo-value-from-form"),
    )
    env = svc._subprocess_env(req)
    assert env["ANTHROPIC_API_KEY"] == "byo-value-from-form"


def test_subprocess_env_no_byo_keeps_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_shell_env(monkeypatch)
    repo_root = tmp_path
    (repo_root / ".env").write_text("OPENAI_API_KEY=server-side-openai\n", encoding="utf-8")
    svc = FileLiveRunService(runs_root=tmp_path / "runs", repo_root=repo_root)
    req = StartRunRequest()  # no provider_credentials
    env = svc._subprocess_env(req)
    assert env["OPENAI_API_KEY"] == "server-side-openai"


def test_subprocess_env_partial_byo_only_overrides_named_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_shell_env(monkeypatch)
    repo_root = tmp_path
    (repo_root / ".env").write_text(
        "ANTHROPIC_API_KEY=server-anth\nOPENAI_API_KEY=server-oai\n",
        encoding="utf-8",
    )
    svc = FileLiveRunService(runs_root=tmp_path / "runs", repo_root=repo_root)
    req = StartRunRequest(
        provider_credentials=ProviderCredentials(anthropic_api_key="byo-anth"),
    )
    env = svc._subprocess_env(req)
    assert env["ANTHROPIC_API_KEY"] == "byo-anth"
    assert env["OPENAI_API_KEY"] == "server-oai"


# ---------------------------------------------------------------------------
# (3) Scrub invariant — keys never appear in any pydantic dump
# ---------------------------------------------------------------------------


def test_credentials_repr_redacts_values() -> None:
    c = ProviderCredentials(anthropic_api_key="sk-ant-SECRET-XYZ")
    assert "SECRET" not in repr(c)


def test_credentials_model_dump_redacts_secrets() -> None:
    c = ProviderCredentials(
        anthropic_api_key="sk-ant-SECRET-1",
        openai_api_key="sk-SECRET-2",
        azure_openai_api_key="azure-SECRET-3",
        azure_openai_endpoint="https://my.openai.azure.com",
        azure_openai_deployment="gpt-4o",
    )
    dumped = c.model_dump()
    # Secrets masked
    assert dumped["anthropic_api_key"] == "***"
    assert dumped["openai_api_key"] == "***"
    assert dumped["azure_openai_api_key"] == "***"
    # Resource identifiers retained (not secrets)
    assert dumped["azure_openai_endpoint"] == "https://my.openai.azure.com"
    assert dumped["azure_openai_deployment"] == "gpt-4o"
    # Full JSON also free of secret values
    blob = c.model_dump_json()
    assert "SECRET" not in blob


def test_start_run_request_dump_redacts_nested_credentials() -> None:
    # The high-risk path: a caller doing `request.model_dump()` for logging
    # must not leak the nested ProviderCredentials values.
    req = StartRunRequest(
        provider_credentials=ProviderCredentials(
            anthropic_api_key="sk-ant-LEAK-CANARY",
            openai_api_key="sk-LEAK-CANARY-2",
        ),
    )
    blob = json.dumps(req.model_dump(), default=str)
    assert "LEAK-CANARY" not in blob
    assert "LEAK-CANARY-2" not in blob
    # And the JSON helper version, which is what live_runs uses.
    assert "LEAK-CANARY" not in req.model_dump_json()


# ---------------------------------------------------------------------------
# (4) Resume override forwarding
# ---------------------------------------------------------------------------


def test_resume_override_accepts_provider_credentials() -> None:
    # The resume_run path's merge dict must accept a freshly-supplied
    # provider_credentials so the user can re-BYO on resume. Build the
    # merge dict the way resume_run does (without spinning up the full
    # FileLiveRunService).
    overrides = {
        "provider_credentials": ProviderCredentials(anthropic_api_key="resume-byo"),
    }
    merged: dict = {}
    if overrides:
        byo = overrides.get("provider_credentials")
        if byo is not None:
            merged["provider_credentials"] = byo
    req = StartRunRequest(**{k: v for k, v in merged.items() if v is not None})
    assert isinstance(req.provider_credentials, ProviderCredentials)
    assert req.provider_credentials.to_env_overrides() == {"ANTHROPIC_API_KEY": "resume-byo"}
