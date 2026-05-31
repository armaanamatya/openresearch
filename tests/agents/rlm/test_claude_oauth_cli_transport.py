"""RLM-root CLI transport (2026-05-31 harness-robustness fix).

The legacy claude-agent-sdk root path emptied ~80-90% of completions on a
contended host (nested-async-generator teardown race — the SDK's background
`_read_messages` task died against loop teardown before yielding any
AssistantMessage). The `claude` CLI is a synchronous single-shot subprocess with
none of that surface; `ClaudeOauthClient` now routes the root completion through
it (``--print --output-format json``), prompt piped via STDIN, falling back to
the SDK only if the CLI is missing/errors. These tests mock ``subprocess.run`` —
no live calls."""
from __future__ import annotations

import json
import subprocess

import pytest

from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient, _root_transport


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def _cli_json(result: str = "```repl\npass\n```", *, is_error: bool = False, subtype: str = "success", usage: dict | None = None) -> str:
    return json.dumps({
        "is_error": is_error,
        "subtype": subtype,
        "result": result,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 3},
    })


# --- _cli_complete success + invocation shape ---------------------------------

def test_cli_complete_success_and_stdin(monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        return _proc(stdout=_cli_json(result="```repl\nx=1\n```"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    c = ClaudeOauthClient(model_name="claude-sonnet-4-6")
    out = c._cli_complete(system="SYS", user="do the thing", model="claude-sonnet-4-6")

    assert out is not None
    text, usage = out
    assert text == "```repl\nx=1\n```"
    assert usage["input_tokens"] == 10 and usage["output_tokens"] == 5
    # user prompt goes via STDIN (ARG_MAX-safe), never argv
    assert captured["input"] == "do the thing"
    assert "do the thing" not in captured["cmd"]
    # invocation shape
    assert "--print" in captured["cmd"]
    assert "--output-format" in captured["cmd"] and "json" in captured["cmd"]
    assert "--model" in captured["cmd"] and "claude-sonnet-4-6" in captured["cmd"]
    assert "--append-system-prompt" in captured["cmd"] and "SYS" in captured["cmd"]
    assert "--disallowed-tools" in captured["cmd"] and "Bash" in captured["cmd"]


def test_cli_complete_omits_system_when_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _proc(stdout=_cli_json()))
    c = ClaudeOauthClient()
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _proc(stdout=_cli_json())

    monkeypatch.setattr(subprocess, "run", fake_run)
    c._cli_complete(system="", user="u", model="m")
    assert "--append-system-prompt" not in captured["cmd"]


@pytest.mark.parametrize("proc,desc", [
    (_proc(returncode=1, stderr="boom"), "nonzero exit"),
    (_proc(stdout="not json at all"), "non-JSON stdout"),
    (_proc(stdout=_cli_json(is_error=True)), "is_error true"),
    (_proc(stdout=_cli_json(subtype="error_max_turns")), "non-success subtype"),
])
def test_cli_complete_failure_modes_return_none(monkeypatch, proc, desc):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: proc)
    c = ClaudeOauthClient()
    assert c._cli_complete(system="", user="u", model="m") is None, desc


def test_cli_complete_timeout_returns_none(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1.0)
    monkeypatch.setattr(subprocess, "run", boom)
    assert ClaudeOauthClient()._cli_complete(system="", user="u", model="m") is None


def test_cli_complete_missing_binary_returns_none(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("claude")
    monkeypatch.setattr(subprocess, "run", boom)
    assert ClaudeOauthClient()._cli_complete(system="", user="u", model="m") is None


# --- completion() routing -----------------------------------------------------

def test_completion_uses_cli_and_records_usage(monkeypatch):
    c = ClaudeOauthClient(model_name="m")
    monkeypatch.setattr(c, "_cli_complete", lambda **kw: ("```repl\nok\n```", {"input_tokens": 7, "output_tokens": 3}))
    out = c.completion("hello")
    assert out == "```repl\nok\n```"
    assert c.model_call_counts["m"] == 1
    assert c.model_input_tokens["m"] == 7 and c.model_output_tokens["m"] == 3


def test_completion_empty_cli_text_yields_keepalive_fallback(monkeypatch):
    c = ClaudeOauthClient(model_name="m")
    monkeypatch.setattr(c, "_cli_complete", lambda **kw: ("   ", {}))
    out = c.completion("hello")
    assert "transient model-transport" in out  # _empty_root_turn_fallback keeps the loop alive


def test_completion_falls_back_to_sdk_when_cli_none(monkeypatch):
    c = ClaudeOauthClient(model_name="m")
    monkeypatch.setattr(c, "_cli_complete", lambda **kw: None)

    class FakeSdkClient:
        def complete(self, *, system, user):
            return "```repl\nfrom_sdk\n```"

    c._claude_clients["m"] = FakeSdkClient()  # pre-seed the lazy SDK cache
    out = c.completion("hello")
    assert out == "```repl\nfrom_sdk\n```"


def test_transport_sdk_skips_cli(monkeypatch):
    monkeypatch.setenv("REPROLAB_RLM_ROOT_TRANSPORT", "sdk")
    assert _root_transport() == "sdk"
    c = ClaudeOauthClient(model_name="m")

    def boom(**kw):
        raise AssertionError("CLI must not be called when transport=sdk")

    monkeypatch.setattr(c, "_cli_complete", boom)

    class FakeSdkClient:
        def complete(self, *, system, user):
            return "```repl\nsdk_only\n```"

    c._claude_clients["m"] = FakeSdkClient()
    assert c.completion("hi") == "```repl\nsdk_only\n```"


def test_root_transport_default_is_cli(monkeypatch):
    monkeypatch.delenv("REPROLAB_RLM_ROOT_TRANSPORT", raising=False)
    assert _root_transport() == "cli"
    monkeypatch.setenv("REPROLAB_RLM_ROOT_TRANSPORT", "garbage")
    assert _root_transport() == "cli"  # invalid → safe default
