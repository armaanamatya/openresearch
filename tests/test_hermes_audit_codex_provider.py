from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.hermes_audit.client import NousHermesClient, _default_provider_chain
from backend.hermes_audit.models import HermesAuditScope, HermesAuditStatus
from backend.hermes_audit.providers import CodexCliProvider, OpenAIAuditProvider


def test_unavailable_when_binary_missing(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    provider = CodexCliProvider(auth_path_override=str(auth))

    with patch("backend.hermes_audit.providers.shutil.which", return_value=None):
        assert provider.is_available() is False


def test_unavailable_when_auth_json_missing(tmp_path: Path) -> None:
    provider = CodexCliProvider(auth_path_override=str(tmp_path / "missing.json"))

    with patch("backend.hermes_audit.providers.shutil.which", return_value="/usr/bin/codex"):
        assert provider.is_available() is False


def test_available_when_binary_and_auth_exist(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    provider = CodexCliProvider(auth_path_override=str(auth))

    with patch("backend.hermes_audit.providers.shutil.which", return_value="/usr/bin/codex"):
        assert provider.is_available() is True


def test_call_returns_last_message_file_on_success(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    provider = CodexCliProvider(
        cli_path="/usr/bin/codex",
        auth_path_override=str(auth),
        cli_timeout_seconds=10,
    )

    def fake_run(args, **kwargs):
        out_path = Path(args[args.index("--output-last-message") + 1])
        out_path.write_text('{"status":"verified"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="progress log", stderr="")

    with patch("backend.hermes_audit.providers.subprocess.run", side_effect=fake_run) as run:
        assert provider.call("audit this") == '{"status":"verified"}'

    call_kwargs = run.call_args.kwargs
    call_args = run.call_args.args[0]
    assert call_kwargs["timeout"] == 10
    assert "--skip-git-repo-check" in call_args
    assert "--ephemeral" in call_args
    assert "--ignore-user-config" in call_args
    assert "--ignore-rules" in call_args


def test_call_falls_back_to_stdout_when_last_message_missing(tmp_path: Path) -> None:
    provider = CodexCliProvider(cli_path="/usr/bin/codex")

    with patch(
        "backend.hermes_audit.providers.subprocess.run",
        return_value=SimpleNamespace(
            returncode=0,
            stdout='{"status":"verified"}\n',
            stderr="",
        ),
    ):
        assert provider.call("audit this") == '{"status":"verified"}'


def test_call_raises_on_nonzero_exit() -> None:
    provider = CodexCliProvider(cli_path="/usr/bin/codex")
    stderr = "x" * 600

    with patch(
        "backend.hermes_audit.providers.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr=stderr),
    ):
        with pytest.raises(RuntimeError) as exc:
            provider.call("audit this")

    assert "codex CLI exited 1" in str(exc.value)
    assert "x" * 500 in str(exc.value)
    assert "x" * 501 not in str(exc.value)


def test_call_raises_on_empty_output() -> None:
    provider = CodexCliProvider(cli_path="/usr/bin/codex")

    with patch(
        "backend.hermes_audit.providers.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ):
        with pytest.raises(RuntimeError, match="empty response"):
            provider.call("audit this")


def test_default_chain_appends_codex_cli() -> None:
    chain = _default_provider_chain("anthropic/claude-sonnet-4")

    assert isinstance(chain[-1], CodexCliProvider)


def test_chain_uses_codex_when_openai_key_path_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(OpenAIAuditProvider, "is_available", lambda self: False)
    monkeypatch.setattr(CodexCliProvider, "is_available", lambda self: True)
    monkeypatch.setattr(
        CodexCliProvider,
        "call",
        lambda self, prompt: (
            '{"target":"paper","scope":"step","status":"grounded",'
            '"summary":"ok","recommended_intervention":"annotate",'
            '"confidence":"high","provider":"codex_cli"}'
        ),
    )
    monkeypatch.setattr(
        "backend.hermes_audit.providers.NousHermesProvider.is_available",
        lambda self: False,
    )
    monkeypatch.setattr(
        "backend.hermes_audit.providers.ClaudeAuditProvider.is_available",
        lambda self: False,
    )
    monkeypatch.setattr(
        "backend.hermes_audit.providers.ClaudeCodeSdkProvider.is_available",
        lambda self: False,
    )

    report = NousHermesClient(runs_root=tmp_path).audit(
        scope=HermesAuditScope.step,
        target="paper",
        payload={},
    )

    assert report.status == HermesAuditStatus.grounded
    assert report.provider == "codex_cli"
