"""Tests for the CLI root-validation gate (oauth-root-reliability plan, P2).

The gate is factored into the pure-ish helper ``backend.cli._root_validation_gate``
so it can be driven hermetically without the full ingest/store/workspace
machinery of ``cmd_reproduce`` (which needs a parsed paper, an open event
store, and workspace claims). The helper is the single decision point;
``cmd_reproduce`` consumes it inline (warn → print stderr, error → return 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.cli import _root_validation_gate


@dataclass(frozen=True)
class _FakeRoot:
    key: str
    rlm_backend: str
    paper_validated: bool


# ---------------------------------------------------------------------------
# Helper behaviour — default-OFF (flag unset)
# ---------------------------------------------------------------------------


def test_gate_flag_unset_oauth_warns_but_does_not_block(monkeypatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", raising=False)
    exit_code, warn, error = _root_validation_gate("claude-oauth")
    assert exit_code is None  # never blocks when flag unset
    assert error is None
    assert warn is not None
    assert "claude-oauth" in warn
    assert "[warn]" in warn


def test_gate_flag_unset_validated_no_warning_no_block(monkeypatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", raising=False)
    exit_code, warn, error = _root_validation_gate("gpt-5")
    assert exit_code is None
    assert warn is None
    assert error is None


# ---------------------------------------------------------------------------
# Helper behaviour — flag set (Variant A)
# ---------------------------------------------------------------------------


def test_gate_flag_set_oauth_blocks(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", "1")
    exit_code, warn, error = _root_validation_gate("claude-oauth")
    assert exit_code == 1
    assert error is not None
    assert "[error]" in error
    assert "claude-oauth" in error
    # The degenerate-loop warning still surfaces even when blocking.
    assert warn is not None and "[warn]" in warn


def test_gate_flag_set_validated_proceeds(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", "1")
    exit_code, warn, error = _root_validation_gate("gpt-5")
    assert exit_code is None
    assert error is None
    assert warn is None


def test_gate_flag_set_unvalidated_non_oauth_blocks(monkeypatch) -> None:
    # A credential-gated model (e.g. "claude") raises in resolve_root_model
    # without its API key and is correctly fail-soft skipped, so inject a
    # resolvable unvalidated non-oauth root to exercise the blocking path.
    monkeypatch.setattr(
        "backend.cli.resolve_root_model",
        lambda _name: _FakeRoot("kimi-k2.5", "openrouter", False),
        raising=False,
    )
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", "1")
    exit_code, warn, error = _root_validation_gate("kimi-k2.5")
    assert exit_code == 1
    assert error is not None and "[error]" in error
    # Non-oauth unvalidated root has no degenerate-loop warning.
    assert warn is None


@pytest.mark.parametrize("flag", ["1", "on", "true", "yes", "TRUE", "On"])
def test_gate_truthy_variants(monkeypatch, flag) -> None:
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", flag)
    exit_code, _warn, error = _root_validation_gate("claude-oauth")
    assert exit_code == 1 and error is not None


@pytest.mark.parametrize("flag", ["", "0", "off", "false", "no"])
def test_gate_falsey_variants(monkeypatch, flag) -> None:
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", flag)
    exit_code, _warn, _error = _root_validation_gate("claude-oauth")
    assert exit_code is None


# ---------------------------------------------------------------------------
# Fail-soft: an unresolvable model must NOT block the run.
# ---------------------------------------------------------------------------


def test_gate_unknown_model_fail_soft(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", "1")
    # resolve_root_model raises ValueError on an unknown key; the gate must
    # swallow it and skip (the real run surfaces the actual error).
    exit_code, warn, error = _root_validation_gate("definitely-not-a-real-model-xyz")
    assert exit_code is None
    assert warn is None
    assert error is None


def test_gate_resolution_exception_fail_soft(monkeypatch) -> None:
    def _boom(_name):
        raise RuntimeError("cred probe blew up")

    monkeypatch.setattr("backend.cli.resolve_root_model", _boom, raising=False)
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", "1")
    exit_code, warn, error = _root_validation_gate("claude-oauth")
    assert exit_code is None and warn is None and error is None


# ---------------------------------------------------------------------------
# End-to-end through cmd_reproduce: the gate fires BEFORE any ingest/dispatch.
# Relocating the gate above the mode dispatch (after rdr, before the PaperBench
# bundle dispatch) covers BOTH RLM run paths and makes this cheaply testable —
# the gate short-circuits before _make_services / the run, so no heavy mocking.
# ---------------------------------------------------------------------------

from argparse import Namespace  # noqa: E402

from backend.cli import cmd_reproduce  # noqa: E402


def test_cmd_reproduce_fail_fast_blocks_before_any_run(tmp_path, monkeypatch) -> None:
    """Flag set + unvalidated root -> cmd_reproduce returns 1 before ingest/dispatch."""
    monkeypatch.setenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", "1")
    monkeypatch.setattr(
        "backend.cli.resolve_root_model",
        lambda _m: _FakeRoot(key="kimi-k2.5", rlm_backend="openrouter", paper_validated=False),
        raising=False,
    )
    reached: list[str] = []
    monkeypatch.setattr("backend.cli._make_services", lambda *a, **k: reached.append("services"), raising=False)
    monkeypatch.setattr(
        "backend.cli._cmd_reproduce_rlm_paperbench",
        lambda *a, **k: (reached.append("paperbench"), 0)[1],
        raising=False,
    )

    rc = cmd_reproduce(Namespace(
        source="some-arxiv-id", mode="rlm", model="kimi-k2.5",
        sanity=False, runs_root=str(tmp_path),
    ))

    assert rc == 1
    assert reached == []  # gate short-circuited before any ingest/services/dispatch


def test_cmd_reproduce_warning_proceeds_and_covers_paperbench(tmp_path, monkeypatch, capsys) -> None:
    """Flag unset + claude-oauth -> loud warning prints AND the run proceeds (gate
    does not block); proves the gate also covers the PaperBench bundle path."""
    monkeypatch.delenv("OPENRESEARCH_REQUIRE_VALIDATED_ROOT", raising=False)
    monkeypatch.setattr(
        "backend.cli.resolve_root_model",
        lambda _m: _FakeRoot(key="claude-oauth", rlm_backend="anthropic-oauth", paper_validated=False),
        raising=False,
    )
    monkeypatch.setattr("backend.cli._is_paperbench_bundle_id", lambda *a, **k: True, raising=False)
    pb: list[int] = []
    monkeypatch.setattr(
        "backend.cli._cmd_reproduce_rlm_paperbench",
        lambda *a, **k: (pb.append(1), 0)[1],
        raising=False,
    )

    rc = cmd_reproduce(Namespace(
        source="some-bundle", mode="rlm", model="claude-oauth",
        sanity=False, runs_root=str(tmp_path),
    ))

    assert rc == 0
    assert pb == [1]  # reached the PaperBench dispatch — gate covers it and did not block
    err = capsys.readouterr().err
    assert "claude-oauth" in err and "[warn]" in err  # loud warning fired before the run
