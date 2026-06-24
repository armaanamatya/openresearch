"""Tests for the credential preflight gate in cmd_reproduce (cli.py).

Pinned guarantees:
- When validate_root_credentials returns (True, ...) → run proceeds (no early exit).
- When validate_root_credentials returns (False, actionable_msg) → CLI exits 1 with message.
- When OPENRESEARCH_SKIP_CRED_PREFLIGHT=1 → preflight is skipped entirely.
- When the preflight probe raises unexpectedly → fail-open (run proceeds).
- The preflight runs BEFORE ingest (no services opened, no subprocess spawned).

All tests mock the network and all agent sub-systems — no real I/O.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> argparse.Namespace:
    """Return a minimal Namespace that satisfies cmd_reproduce's early path."""
    defaults = {
        "source": "2605.15155",
        "source_kind": "auto",
        "mode": "rlm",
        "model": None,
        "provider": None,
        "verification_provider": None,
        "hints": None,
        "runs_root": "/tmp/test_runs",
        "database_url": "sqlite:///test.db",
        "agent": "default",
        "sanity": False,
        "resume_cells": False,
        "force_regen": False,
        "fresh": False,
        "paper_hint": None,
        "scope_spec": None,
        "blacklist": None,
        "models": None,
        "execution_mode": "max",
        "sandbox": "local",
        "gpu_mode": "auto",
        "command_timeout": None,
        "allow_sandbox_network": False,
        "sandbox_platform": None,
        "sandbox_memory": None,
        "sandbox_cpus": None,
        "max_usd": None,
        "max_wall_clock": None,
        "max_pod_seconds": None,
        "max_rlm_iterations": None,
        "max_invocations": None,
        "seed": None,
        "attempt_id": None,
        "run_group_id": None,
        "project_id": None,
        "minimize_compute": False,
        "run_spec": None,
        "dynamic_gpu": None,
        "force_single_gpu": None,
        "max_gpu_usd_per_hour": None,
        "max_run_gpu_usd": None,
        "dynamic_gpu_headroom": None,
        "vram_gb": None,
        "gpu_parallelism": None,
        "accelerator": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _stub_cmd_reproduce_early_exit(monkeypatch, exit_code: int | None = None) -> MagicMock:
    """Stub out everything AFTER the preflight so tests can call cmd_reproduce
    without needing real services.  When exit_code is not None, the stub raises
    SystemExit to simulate an early-return from _root_validation_gate."""
    # These helpers are called before the preflight; make them no-ops.
    monkeypatch.setattr("backend.cli._warn_on_shell_env_override", lambda: None)
    monkeypatch.setattr("backend.cli._install_termination_handlers", lambda: None)
    monkeypatch.setattr("backend.cli._set_active_project_id", lambda *a, **k: None)
    monkeypatch.setattr("backend.cli.configure_root_logger", lambda: None, raising=False)

    # Stub normalize_path_input so the arXiv id is unchanged.
    try:
        monkeypatch.setattr(
            "backend.services.paths.normalize_path_input",
            lambda x: x,
        )
    except Exception:
        pass

    # Stub paper_registry to return None (no bundled paper).
    mock_registry = MagicMock()
    mock_registry.resolve.return_value = None
    try:
        monkeypatch.setattr("backend.services.ingestion.paper_registry", mock_registry)
    except Exception:
        pass

    # _root_validation_gate: return (None, None, None) — no blocking, no warn.
    monkeypatch.setattr("backend.cli._root_validation_gate", lambda model: (None, None, None))

    # Stub observability module
    monkeypatch.setattr(
        "backend.observability.run_logging.configure_root_logger",
        lambda: None,
        raising=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCredPreflightGate:
    """Integration tests — call cmd_reproduce, assert early-exit or proceed."""

    def test_valid_cred_does_not_abort(self, monkeypatch, tmp_path):
        """validate_root_credentials returns (True, ...) → run proceeds past gate."""
        _stub_cmd_reproduce_early_exit(monkeypatch)

        # The preflight passes; downstream ingest is the next step.
        # We stub ingest to raise a sentinel exception so we can detect
        # that the preflight was passed (if it had aborted with return 1,
        # we'd never reach the sentinel).
        class _IngestSentinel(Exception):
            pass

        monkeypatch.setattr("backend.cli._make_services", MagicMock(side_effect=_IngestSentinel()))

        _mock_root_entry = MagicMock()
        _mock_root_entry.rlm_backend = "openai"
        _mock_root_entry.key = "gpt-5"

        # Patch the function at the module where it lives (local import in cli.py)
        with (
            patch("backend.agents.rlm.pre_flight_validator.validate_root_credentials",
                  return_value=(True, "[cred-preflight] ok")),
            patch("backend.agents.rlm.models.resolve_root_model", return_value=_mock_root_entry),
            pytest.raises(_IngestSentinel),
        ):
            from backend.cli import cmd_reproduce
            args = _make_args(runs_root=str(tmp_path))
            cmd_reproduce(args)

    def test_bad_cred_aborts_with_exit_1(self, monkeypatch, tmp_path):
        """validate_root_credentials returns (False, msg) → returns 1, message in stderr."""
        _stub_cmd_reproduce_early_exit(monkeypatch)

        _mock_root_entry = MagicMock()
        _mock_root_entry.rlm_backend = "openai"
        _mock_root_entry.key = "gpt-5"

        import io
        fake_stderr = io.StringIO()

        with (
            patch("backend.agents.rlm.pre_flight_validator.validate_root_credentials",
                  return_value=(
                      False,
                      "[cred-preflight] OPENAI_API_KEY rejected (HTTP 401).",
                  )),
            patch("backend.agents.rlm.models.resolve_root_model", return_value=_mock_root_entry),
            patch("sys.stderr", fake_stderr),
        ):
            from backend.cli import cmd_reproduce
            args = _make_args(runs_root=str(tmp_path))
            result = cmd_reproduce(args)

        assert result == 1
        err_text = fake_stderr.getvalue()
        assert "OPENAI_API_KEY" in err_text or "401" in err_text or "error" in err_text.lower()

    def test_skip_env_var_bypasses_gate(self, monkeypatch, tmp_path):
        """OPENRESEARCH_SKIP_CRED_PREFLIGHT=1 means validate_root_credentials is never called."""
        monkeypatch.setenv("OPENRESEARCH_SKIP_CRED_PREFLIGHT", "1")
        _stub_cmd_reproduce_early_exit(monkeypatch)

        class _IngestSentinel(Exception):
            pass

        monkeypatch.setattr("backend.cli._make_services", MagicMock(side_effect=_IngestSentinel()))

        # validate_root_credentials should NOT be called — wrap it to detect a call.
        cred_called = []
        with (
            patch("backend.agents.rlm.pre_flight_validator.validate_root_credentials",
                  side_effect=lambda *a, **k: cred_called.append(1) or (True, "ok")),
            pytest.raises(_IngestSentinel),
        ):
            from backend.cli import cmd_reproduce
            args = _make_args(runs_root=str(tmp_path))
            cmd_reproduce(args)

        assert cred_called == [], "validate_root_credentials was called despite SKIP flag"

    def test_preflight_probe_exception_fail_open(self, monkeypatch, tmp_path):
        """If the preflight probe itself raises, run proceeds (fail-open)."""
        _stub_cmd_reproduce_early_exit(monkeypatch)

        class _IngestSentinel(Exception):
            pass

        monkeypatch.setattr("backend.cli._make_services", MagicMock(side_effect=_IngestSentinel()))

        _mock_root_entry = MagicMock()
        _mock_root_entry.rlm_backend = "openai"
        _mock_root_entry.key = "gpt-5"

        # Even when the probe raises (any exception), the run must NOT exit 1.
        with (
            patch("backend.agents.rlm.pre_flight_validator.validate_root_credentials",
                  side_effect=RuntimeError("probe exploded")),
            patch("backend.agents.rlm.models.resolve_root_model", return_value=_mock_root_entry),
            pytest.raises(_IngestSentinel),
        ):
            from backend.cli import cmd_reproduce
            args = _make_args(runs_root=str(tmp_path))
            cmd_reproduce(args)
