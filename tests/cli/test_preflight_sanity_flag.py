"""P3 (2026-05-29): --preflight-sanity flag wires through cmd_reproduce.

These tests verify the flag is registered, defaults to off, and the
sandbox-gated branch in cmd_reproduce dispatches the sanity check only
for local/docker (not runpod, where transient capacity exhaustion would
otherwise terminate a real run that would have waited it out).

The sanity check itself runs ``run_experiment`` under the hood, which
talks to a real sandbox backend — we mock ``_run_sanity_check`` so these
tests stay fast and hermetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.cli import (
    _build_parser,
    _run_sanity_check,
    _with_reproduce_defaults,
    cmd_reproduce,
)


def test_preflight_sanity_flag_is_registered() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["reproduce", "foo.pdf", "--preflight-sanity"])
    assert parsed.preflight_sanity is True


def test_preflight_sanity_defaults_to_off() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["reproduce", "foo.pdf"])
    assert parsed.preflight_sanity is False


def test_preflight_sanity_default_via_backfill() -> None:
    """Programmatic callers may pass a Namespace without the flag — backfill
    must default it to False (P3 is opt-in)."""
    args = argparse.Namespace(source="foo.pdf")
    args = _with_reproduce_defaults(args)
    assert args.preflight_sanity is False


@pytest.mark.parametrize("sandbox", ["local", "docker"])
def test_preflight_dispatches_for_local_and_docker(sandbox: str, tmp_path: Path) -> None:
    """When the flag is set AND sandbox is local/docker, the sanity check
    fires and (when it returns success) the real run proceeds."""
    args = _with_reproduce_defaults(
        argparse.Namespace(
            source="mechanistic-understanding",
            sandbox=sandbox,
            preflight_sanity=True,
            runs_root=str(tmp_path),
            mode="rlm",
        )
    )
    with patch("backend.cli._run_sanity_check") as mock_sanity:
        mock_sanity.return_value = (True, "sanity ok", "prj_smoke")
        # Stub everything beyond the preflight so the test doesn't try to
        # spawn a real RLM. We only care that _run_sanity_check was called.
        with patch("backend.cli._cmd_reproduce_rlm_paperbench") as mock_run:
            mock_run.return_value = 0
            rc = cmd_reproduce(args)
    assert mock_sanity.called, "sanity preflight must fire for local/docker"
    # The call wrote_demo_status arg must be False (don't pollute UI with the
    # preflight project).
    _, kwargs = mock_sanity.call_args
    assert kwargs.get("write_demo_status") is False


def test_preflight_aborts_on_sanity_failure(tmp_path: Path) -> None:
    """When --preflight-sanity is set and the sanity check fails, the real
    run must NOT launch. Exit code must be 4 (preflight-failure), distinct
    from 3 (budget-exhausted)."""
    args = _with_reproduce_defaults(
        argparse.Namespace(
            source="mechanistic-understanding",
            sandbox="docker",
            preflight_sanity=True,
            runs_root=str(tmp_path),
            mode="rlm",
        )
    )
    with patch("backend.cli._run_sanity_check") as mock_sanity:
        mock_sanity.return_value = (False, "docker daemon down", "prj_smoke")
        with patch("backend.cli._cmd_reproduce_rlm_paperbench") as mock_run:
            rc = cmd_reproduce(args)
    assert rc == 4, "preflight failure must return exit code 4"
    assert not mock_run.called, "real run must not launch on preflight failure"


@pytest.mark.parametrize("sandbox", ["runpod", "auto"])
def test_preflight_skipped_for_runpod_and_auto(sandbox: str, tmp_path: Path) -> None:
    """RunPod is excluded by P3 design — transient capacity exhaustion would
    convert a temporary issue into a terminal one. 'auto' is treated the
    same since it commonly resolves to runpod."""
    args = _with_reproduce_defaults(
        argparse.Namespace(
            source="mechanistic-understanding",
            sandbox=sandbox,
            preflight_sanity=True,
            runs_root=str(tmp_path),
            mode="rlm",
        )
    )
    with patch("backend.cli._run_sanity_check") as mock_sanity:
        with patch("backend.cli._cmd_reproduce_rlm_paperbench") as mock_run:
            mock_run.return_value = 0
            cmd_reproduce(args)
    assert not mock_sanity.called, (
        f"sanity preflight must NOT fire for --sandbox={sandbox} (runpod-class)"
    )


def test_no_preflight_flag_means_no_dispatch(tmp_path: Path) -> None:
    """Backward compat: without the flag, cmd_reproduce must not call sanity."""
    args = _with_reproduce_defaults(
        argparse.Namespace(
            source="mechanistic-understanding",
            sandbox="docker",
            preflight_sanity=False,
            runs_root=str(tmp_path),
            mode="rlm",
        )
    )
    with patch("backend.cli._run_sanity_check") as mock_sanity:
        with patch("backend.cli._cmd_reproduce_rlm_paperbench") as mock_run:
            mock_run.return_value = 0
            cmd_reproduce(args)
    assert not mock_sanity.called
