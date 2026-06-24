"""Guard tests for the ``--effort`` flag on the Claude OAuth CLI root path.

Tests cover:
- ``_root_effort()`` default, valid override, and invalid-input fallback.
- ``_build_root_cli_cmd()`` structure: required flags always present, effort
  flag correct position, ``--append-system-prompt`` conditional on system.
- Round-trip: an overridden effort value appears in the argv produced by
  ``_build_root_cli_cmd``.

No real subprocess or network calls are made.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _root_effort() tests
# ---------------------------------------------------------------------------


class TestRootEffortDefault:
    """_root_effort() returns 'high' when env is unset."""

    def test_returns_high_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _root_effort

        assert _root_effort() == "high"


class TestRootEffortValidOverride:
    """A valid OPENRESEARCH_ROOT_EFFORT value is returned unchanged."""

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
    def test_valid_level_returned(self, monkeypatch, level):
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", level)
        from backend.agents.rlm.claude_oauth_client import _root_effort

        assert _root_effort() == level

    def test_xhigh_override(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", "xhigh")
        from backend.agents.rlm.claude_oauth_client import _root_effort

        assert _root_effort() == "xhigh"


class TestRootEffortInvalidFallback:
    """An invalid value is replaced with 'high' without raising."""

    def test_bogus_value_falls_back_to_high(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", "__bogus__")
        from backend.agents.rlm.claude_oauth_client import _root_effort

        assert _root_effort() == "high"

    def test_empty_value_falls_back_to_high(self, monkeypatch):
        # An empty string is not a valid level.
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", "")
        from backend.agents.rlm.claude_oauth_client import _root_effort

        assert _root_effort() == "high"

    def test_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", "INVALID")
        from backend.agents.rlm.claude_oauth_client import _root_effort

        # Must be fail-soft — never raise.
        result = _root_effort()
        assert result == "high"


# ---------------------------------------------------------------------------
# _build_root_cli_cmd() tests
# ---------------------------------------------------------------------------


class TestBuildRootCliCmdStructure:
    """_build_root_cli_cmd always includes the required core flags."""

    def test_contains_print_flag(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort="high")
        assert "--print" in cmd

    def test_contains_output_format_json(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort="high")
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "json"

    def test_contains_model_flag(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort="high")
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4-6"

    def test_contains_disallowed_tools(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import (
            _ROOT_DISALLOWED_TOOLS,
            _build_root_cli_cmd,
        )

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort="high")
        assert "--disallowed-tools" in cmd
        for tool in _ROOT_DISALLOWED_TOOLS:
            assert tool in cmd


class TestBuildRootCliCmdEffortFlag:
    """--effort flag is present and immediately followed by the effort value."""

    def test_effort_flag_present_default(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort="high")
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"

    def test_effort_flag_with_xhigh_override(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", "xhigh")
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd, _root_effort

        effort = _root_effort()
        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort=effort)
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "xhigh"

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
    def test_effort_flag_valid_values(self, monkeypatch, level):
        monkeypatch.setenv("OPENRESEARCH_ROOT_EFFORT", level)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort=level)
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == level


class TestBuildRootCliCmdSystemPrompt:
    """--system-prompt is present only when system is non-empty."""

    def test_system_prompt_included_when_non_empty(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(
            model="claude-sonnet-4-6", system="You are a researcher.", effort="high"
        )
        assert "--system-prompt" in cmd
        assert cmd[cmd.index("--system-prompt") + 1] == "You are a researcher."
        # Regression guard: the buggy --append-system-prompt flag must not appear.
        assert "--append-system-prompt" not in cmd

    def test_system_prompt_omitted_when_empty(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ROOT_EFFORT", raising=False)
        from backend.agents.rlm.claude_oauth_client import _build_root_cli_cmd

        cmd = _build_root_cli_cmd(model="claude-sonnet-4-6", system="", effort="high")
        assert "--system-prompt" not in cmd
        assert "--append-system-prompt" not in cmd
